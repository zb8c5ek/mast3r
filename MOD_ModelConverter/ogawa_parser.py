"""
Low-level Ogawa binary format parser.

Ogawa is a little-endian binary file format used by Alembic (.abc) files.
This module provides direct binary parsing without any external dependencies.

File Structure:
- Header (16 bytes):
  - Bytes 0-4: "Ogawa" magic signature
  - Byte 5: Frozen flag (0x00 = writing, 0xff = closed properly)
  - Bytes 6-7: Version (currently 1)
  - Bytes 8-15: Offset to root group (uint64)

- Groups:
  - Empty group: 8 zero bytes
  - Non-empty group: 8-byte child count, followed by 8 bytes per child
  - Child entry: top bit = 0 for Group, 1 for Data; remaining 63 bits = offset

- Data:
  - Empty data: top bit 1, bottom 63 bits = 0
  - Non-empty data: 8-byte size at offset, followed by n bytes of data
"""

import struct
import logging
from dataclasses import dataclass, field
from typing import List, Optional, BinaryIO, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
OGAWA_MAGIC = b'Ogawa'
OGAWA_FROZEN = 0xff
OGAWA_VERSION = 1
GROUP_FLAG_MASK = 0x8000000000000000  # Top bit mask for 64-bit value
OFFSET_MASK = 0x7FFFFFFFFFFFFFFF      # Bottom 63 bits mask


@dataclass
class OgawaHeader:
    """Ogawa file header information."""
    magic: bytes
    frozen: int
    version: int
    root_group_offset: int
    
    @property
    def is_valid(self) -> bool:
        return self.magic == OGAWA_MAGIC
    
    @property
    def is_frozen(self) -> bool:
        return self.frozen == OGAWA_FROZEN


@dataclass
class OgawaData:
    """Represents an Ogawa Data block."""
    offset: int
    size: int
    _reader: 'OgawaReader' = field(repr=False)
    _cached_data: Optional[bytes] = field(default=None, repr=False)
    
    def read(self) -> bytes:
        """Read the data bytes from the file."""
        if self._cached_data is None:
            self._cached_data = self._reader.read_data_bytes(self.offset, self.size)
        return self._cached_data
    
    def read_doubles(self) -> List[float]:
        """Read data as array of doubles (float64)."""
        data = self.read()
        count = len(data) // 8
        return list(struct.unpack(f'<{count}d', data[:count*8]))
    
    def read_floats(self) -> List[float]:
        """Read data as array of floats (float32)."""
        data = self.read()
        count = len(data) // 4
        return list(struct.unpack(f'<{count}f', data[:count*4]))
    
    def read_int32s(self) -> List[int]:
        """Read data as array of int32."""
        data = self.read()
        count = len(data) // 4
        return list(struct.unpack(f'<{count}i', data[:count*4]))
    
    def read_uint32s(self) -> List[int]:
        """Read data as array of uint32."""
        data = self.read()
        count = len(data) // 4
        return list(struct.unpack(f'<{count}I', data[:count*4]))
    
    def read_uint64s(self) -> List[int]:
        """Read data as array of uint64."""
        data = self.read()
        count = len(data) // 8
        return list(struct.unpack(f'<{count}Q', data[:count*8]))
    
    def read_string(self) -> str:
        """Read data as null-terminated string."""
        data = self.read()
        # Find null terminator or use entire data
        null_pos = data.find(b'\x00')
        if null_pos >= 0:
            return data[:null_pos].decode('utf-8', errors='replace')
        return data.decode('utf-8', errors='replace')
    
    def read_strings(self) -> List[str]:
        """Read data as multiple null-terminated strings."""
        data = self.read()
        strings = []
        current = b''
        for byte in data:
            if byte == 0:
                if current:
                    strings.append(current.decode('utf-8', errors='replace'))
                    current = b''
            else:
                current += bytes([byte])
        if current:
            strings.append(current.decode('utf-8', errors='replace'))
        return strings


@dataclass  
class OgawaGroup:
    """Represents an Ogawa Group node."""
    offset: int
    num_children: int
    child_entries: List[int]  # Raw 8-byte entries
    _reader: 'OgawaReader' = field(repr=False)
    _cached_children: Optional[List[Union['OgawaGroup', OgawaData]]] = field(default=None, repr=False)
    
    def is_child_data(self, index: int) -> bool:
        """Check if child at index is Data (vs Group)."""
        if index >= len(self.child_entries):
            return False
        return bool(self.child_entries[index] & GROUP_FLAG_MASK)
    
    def is_child_group(self, index: int) -> bool:
        """Check if child at index is a Group."""
        if index >= len(self.child_entries):
            return False
        return not bool(self.child_entries[index] & GROUP_FLAG_MASK)
    
    def get_child_offset(self, index: int) -> int:
        """Get the offset for a child."""
        if index >= len(self.child_entries):
            return 0
        return self.child_entries[index] & OFFSET_MASK
    
    def get_child(self, index: int) -> Optional[Union['OgawaGroup', OgawaData]]:
        """Get child at index, loading it if necessary."""
        if index >= self.num_children:
            return None
            
        entry = self.child_entries[index]
        offset = entry & OFFSET_MASK
        is_data = bool(entry & GROUP_FLAG_MASK)
        
        if is_data:
            if offset == 0:
                # Empty data
                return OgawaData(offset=0, size=0, _reader=self._reader)
            else:
                # Read data size from offset
                size = self._reader.read_uint64(offset)
                return OgawaData(offset=offset + 8, size=size, _reader=self._reader)
        else:
            # It's a group
            return self._reader.read_group(offset)
    
    def get_children(self) -> List[Union['OgawaGroup', OgawaData]]:
        """Get all children."""
        if self._cached_children is None:
            self._cached_children = []
            for i in range(self.num_children):
                child = self.get_child(i)
                if child is not None:
                    self._cached_children.append(child)
        return self._cached_children
    
    def get_data(self, index: int) -> Optional[OgawaData]:
        """Get data child at index (convenience method)."""
        child = self.get_child(index)
        if isinstance(child, OgawaData):
            return child
        return None
    
    def get_group(self, index: int) -> Optional['OgawaGroup']:
        """Get group child at index (convenience method)."""
        child = self.get_child(index)
        if isinstance(child, OgawaGroup):
            return child
        return None


class OgawaReader:
    """
    Low-level Ogawa file reader.
    
    Usage:
        reader = OgawaReader("file.abc")
        reader.open()
        root = reader.get_root_group()
        # ... traverse groups and data ...
        reader.close()
    """
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self._file: Optional[BinaryIO] = None
        self._header: Optional[OgawaHeader] = None
        self._root_group: Optional[OgawaGroup] = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def open(self):
        """Open the file and read header."""
        self._file = open(self.filepath, 'rb')
        self._header = self._read_header()
        
        if not self._header.is_valid:
            raise ValueError(f"Invalid Ogawa file: bad magic signature")
        
        logger.info(f"Opened Ogawa file: {self.filepath}")
        logger.debug(f"Header: frozen={self._header.is_frozen}, version={self._header.version}, "
                    f"root_offset={self._header.root_group_offset}")
    
    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None
    
    @property
    def header(self) -> Optional[OgawaHeader]:
        return self._header
    
    def _read_header(self) -> OgawaHeader:
        """Read the 16-byte Ogawa header."""
        self._file.seek(0)
        data = self._file.read(16)
        
        if len(data) < 16:
            raise ValueError("File too small to be valid Ogawa")
        
        magic = data[0:5]
        frozen = data[5]
        version = struct.unpack('<H', data[6:8])[0]
        root_offset = struct.unpack('<Q', data[8:16])[0]
        
        return OgawaHeader(
            magic=magic,
            frozen=frozen,
            version=version,
            root_group_offset=root_offset
        )
    
    def read_uint64(self, offset: int) -> int:
        """Read a uint64 at the given offset."""
        self._file.seek(offset)
        data = self._file.read(8)
        return struct.unpack('<Q', data)[0]
    
    def read_data_bytes(self, offset: int, size: int) -> bytes:
        """Read raw bytes from file."""
        if size == 0:
            return b''
        self._file.seek(offset)
        return self._file.read(size)
    
    def read_group(self, offset: int) -> OgawaGroup:
        """Read a group at the given offset."""
        self._file.seek(offset)
        
        # First 8 bytes could be zero (empty group) or child count
        first_8 = self._file.read(8)
        num_children = struct.unpack('<Q', first_8)[0]
        
        if num_children == 0:
            # Empty group
            return OgawaGroup(
                offset=offset,
                num_children=0,
                child_entries=[],
                _reader=self
            )
        
        # Read child entries (8 bytes each)
        child_data = self._file.read(num_children * 8)
        child_entries = list(struct.unpack(f'<{num_children}Q', child_data))
        
        return OgawaGroup(
            offset=offset,
            num_children=num_children,
            child_entries=child_entries,
            _reader=self
        )
    
    def get_root_group(self) -> OgawaGroup:
        """Get the root group of the file."""
        if self._root_group is None:
            if self._header is None:
                raise RuntimeError("File not opened")
            self._root_group = self.read_group(self._header.root_group_offset)
        return self._root_group
    
    def dump_structure(self, max_depth: int = 5):
        """Debug: print the file structure."""
        root = self.get_root_group()
        self._dump_group(root, 0, max_depth)
    
    def _dump_group(self, group: OgawaGroup, depth: int, max_depth: int):
        """Recursively dump group structure."""
        indent = "  " * depth
        print(f"{indent}Group @ {group.offset}: {group.num_children} children")
        
        if depth >= max_depth:
            print(f"{indent}  ... (max depth reached)")
            return
        
        for i in range(group.num_children):
            child = group.get_child(i)
            if isinstance(child, OgawaData):
                preview = ""
                if child.size > 0 and child.size < 100:
                    data = child.read()
                    # Try to decode as string
                    try:
                        preview = f' = "{data[:50].decode("utf-8", errors="replace")}"'
                    except:
                        preview = f" = {data[:50].hex()}"
                print(f"{indent}  [{i}] Data @ {child.offset}: {child.size} bytes{preview}")
            elif isinstance(child, OgawaGroup):
                print(f"{indent}  [{i}] ", end="")
                self._dump_group(child, depth + 1, max_depth)


def read_ogawa_file(filepath: Union[str, Path]) -> OgawaReader:
    """Convenience function to open an Ogawa file."""
    reader = OgawaReader(filepath)
    reader.open()
    return reader


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        with OgawaReader(sys.argv[1]) as reader:
            print(f"File: {reader.filepath}")
            print(f"Valid: {reader.header.is_valid}")
            print(f"Frozen: {reader.header.is_frozen}")
            print(f"Version: {reader.header.version}")
            print()
            reader.dump_structure(max_depth=3)
