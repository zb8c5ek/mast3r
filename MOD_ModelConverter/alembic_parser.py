"""
Alembic semantic layer parser.

This module builds on top of the Ogawa binary parser to interpret
Alembic's semantic structure: objects, properties, and metadata.

Alembic Structure (on top of Ogawa):
- Archive: root container
  - Time samplings
  - Indexed metadata
  - Root object
    - Child objects (Xform, Camera, PolyMesh, etc.)
    - Properties (Compound, Scalar, Array)

Key insights from AbcCoreOgawa C++ source:
- Object groups contain:
  - Child 0: Compound property group (if exists)
  - Child 1+: Child object groups
  - Last child (if data): Object headers
- Property headers are encoded with bitmasks for type, pod, extent, etc.
- Metadata can be indexed for space efficiency
"""

import struct
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import IntEnum

from .ogawa_parser import OgawaReader, OgawaGroup, OgawaData

logger = logging.getLogger(__name__)


class PropertyType(IntEnum):
    """Alembic property types."""
    COMPOUND = 0
    SCALAR = 1
    ARRAY = 2


class PlainOldDataType(IntEnum):
    """Alembic POD types (Plain Old Data)."""
    BOOLEAN = 0
    UINT8 = 1
    INT8 = 2
    UINT16 = 3
    INT16 = 4
    UINT32 = 5
    INT32 = 6
    UINT64 = 7
    INT64 = 8
    FLOAT16 = 9
    FLOAT32 = 10
    FLOAT64 = 11
    STRING = 12
    WSTRING = 13


# Sizes for POD types (in bytes)
POD_SIZES = {
    PlainOldDataType.BOOLEAN: 1,
    PlainOldDataType.UINT8: 1,
    PlainOldDataType.INT8: 1,
    PlainOldDataType.UINT16: 2,
    PlainOldDataType.INT16: 2,
    PlainOldDataType.UINT32: 4,
    PlainOldDataType.INT32: 4,
    PlainOldDataType.UINT64: 8,
    PlainOldDataType.INT64: 8,
    PlainOldDataType.FLOAT16: 2,
    PlainOldDataType.FLOAT32: 4,
    PlainOldDataType.FLOAT64: 8,
    PlainOldDataType.STRING: 0,  # Variable
    PlainOldDataType.WSTRING: 0,  # Variable
}


@dataclass
class MetaData:
    """Alembic metadata - key-value pairs."""
    data: Dict[str, str] = field(default_factory=dict)
    
    def get(self, key: str, default: str = "") -> str:
        return self.data.get(key, default)
    
    @property
    def schema(self) -> str:
        """Get the schema type (e.g., 'AbcGeom_Camera_v1')."""
        return self.data.get('schema', '')
    
    @property
    def schema_base_type(self) -> str:
        """Get the base schema type (e.g., 'AbcGeom_Camera')."""
        schema = self.schema
        if '_v' in schema:
            return schema.rsplit('_v', 1)[0]
        return schema
    
    def deserialize(self, text: str):
        """Parse metadata from serialized string format."""
        # Format is typically: key=value;key2=value2;...
        self.data.clear()
        if not text:
            return
        for part in text.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                self.data[key.strip()] = value.strip()


@dataclass
class DataType:
    """Alembic data type descriptor."""
    pod: PlainOldDataType
    extent: int = 1  # Number of POD elements per item (e.g., 3 for Vec3)
    
    @property
    def num_bytes(self) -> int:
        """Total bytes per item."""
        return POD_SIZES.get(self.pod, 0) * self.extent


@dataclass
class PropertyHeader:
    """Header information for a property."""
    name: str
    property_type: PropertyType
    data_type: Optional[DataType] = None
    metadata: MetaData = field(default_factory=MetaData)
    is_scalar_like: bool = False
    is_homogeneous: bool = False
    num_samples: int = 0
    first_changed_index: int = 0
    last_changed_index: int = 0
    time_sampling_index: int = 0


@dataclass
class ObjectHeader:
    """Header information for an object."""
    name: str
    full_name: str
    metadata: MetaData = field(default_factory=MetaData)
    
    @property
    def schema(self) -> str:
        return self.metadata.schema
    
    def is_camera(self) -> bool:
        return 'Camera' in self.metadata.schema
    
    def is_xform(self) -> bool:
        return 'Xform' in self.metadata.schema


class AlembicProperty:
    """
    Represents an Alembic property (Compound, Scalar, or Array).
    """
    
    def __init__(self, header: PropertyHeader, group: OgawaGroup, archive: 'AlembicArchive'):
        self.header = header
        self._group = group
        self._archive = archive
        self._sub_properties: Optional[Dict[str, 'AlembicProperty']] = None
        self._samples: Optional[List[Any]] = None
    
    @property
    def name(self) -> str:
        return self.header.name
    
    @property
    def is_compound(self) -> bool:
        return self.header.property_type == PropertyType.COMPOUND
    
    @property
    def is_scalar(self) -> bool:
        return self.header.property_type == PropertyType.SCALAR
    
    @property
    def is_array(self) -> bool:
        return self.header.property_type == PropertyType.ARRAY
    
    def get_sub_properties(self) -> Dict[str, 'AlembicProperty']:
        """Get sub-properties for a compound property."""
        if not self.is_compound:
            return {}
        
        if self._sub_properties is None:
            self._sub_properties = {}
            # Parse sub-properties from the group
            # Implementation depends on how properties are stored
        
        return self._sub_properties
    
    def get_sample(self, index: int = 0) -> Any:
        """Get sample value at index."""
        if self._samples is None:
            self._load_samples()
        
        if index < len(self._samples):
            return self._samples[index]
        return None
    
    def get_all_samples(self) -> List[Any]:
        """Get all sample values."""
        if self._samples is None:
            self._load_samples()
        return self._samples
    
    def _load_samples(self):
        """Load sample data from the group."""
        self._samples = []
        
        if self._group is None or self._group.num_children == 0:
            return
        
        # For scalar/array properties, samples are stored in data children
        for i in range(self._group.num_children):
            if self._group.is_child_data(i):
                data = self._group.get_data(i)
                if data and data.size > 0:
                    value = self._read_sample_data(data)
                    if value is not None:
                        self._samples.append(value)
    
    def _read_sample_data(self, data: OgawaData) -> Any:
        """Read and interpret sample data based on data type."""
        if self.header.data_type is None:
            return data.read()
        
        dt = self.header.data_type
        raw = data.read()
        
        # Skip the 16-byte key/hash at the start if present
        if len(raw) > 16:
            raw = raw[16:]
        
        if dt.pod == PlainOldDataType.FLOAT64:
            count = len(raw) // 8
            values = list(struct.unpack(f'<{count}d', raw[:count*8]))
            if dt.extent > 1 and count >= dt.extent:
                # Group into tuples based on extent
                return [tuple(values[i:i+dt.extent]) for i in range(0, count, dt.extent)]
            return values
        elif dt.pod == PlainOldDataType.FLOAT32:
            count = len(raw) // 4
            values = list(struct.unpack(f'<{count}f', raw[:count*4]))
            if dt.extent > 1:
                return [tuple(values[i:i+dt.extent]) for i in range(0, count, dt.extent)]
            return values
        elif dt.pod == PlainOldDataType.INT32:
            count = len(raw) // 4
            return list(struct.unpack(f'<{count}i', raw[:count*4]))
        elif dt.pod == PlainOldDataType.UINT32:
            count = len(raw) // 4
            return list(struct.unpack(f'<{count}I', raw[:count*4]))
        elif dt.pod == PlainOldDataType.UINT8:
            return list(raw)
        elif dt.pod == PlainOldDataType.STRING:
            return data.read_strings()
        else:
            return raw


class AlembicObject:
    """
    Represents an Alembic object in the hierarchy.
    
    Objects can be geometric types (Camera, Xform, PolyMesh, etc.)
    or organizational containers.
    """
    
    def __init__(self, header: ObjectHeader, group: OgawaGroup, archive: 'AlembicArchive'):
        self.header = header
        self._group = group
        self._archive = archive
        self._children: Optional[Dict[str, 'AlembicObject']] = None
        self._properties: Optional[Dict[str, AlembicProperty]] = None
    
    @property
    def name(self) -> str:
        return self.header.name
    
    @property
    def full_name(self) -> str:
        return self.header.full_name
    
    @property
    def schema(self) -> str:
        return self.header.schema
    
    def is_camera(self) -> bool:
        return self.header.is_camera()
    
    def is_xform(self) -> bool:
        return self.header.is_xform()
    
    def get_children(self) -> Dict[str, 'AlembicObject']:
        """Get all child objects."""
        if self._children is None:
            self._load_children()
        return self._children
    
    def get_child(self, name: str) -> Optional['AlembicObject']:
        """Get a child by name."""
        return self.get_children().get(name)
    
    def get_properties(self) -> Dict[str, AlembicProperty]:
        """Get all properties on this object."""
        if self._properties is None:
            self._load_properties()
        return self._properties
    
    def get_property(self, name: str) -> Optional[AlembicProperty]:
        """Get a property by name."""
        return self.get_properties().get(name)
    
    def find_cameras(self) -> List['AlembicObject']:
        """Recursively find all camera objects in this subtree."""
        cameras = []
        if self.is_camera():
            cameras.append(self)
        for child in self.get_children().values():
            cameras.extend(child.find_cameras())
        return cameras
    
    def find_xforms(self) -> List['AlembicObject']:
        """Recursively find all xform objects in this subtree."""
        xforms = []
        if self.is_xform():
            xforms.append(self)
        for child in self.get_children().values():
            xforms.extend(child.find_xforms())
        return xforms
    
    def _load_children(self):
        """Load child objects from the group."""
        self._children = {}
        
        if self._group is None or self._group.num_children == 0:
            return
        
        # Read object headers from the last data child
        headers = self._read_object_headers()
        
        # Create child objects (starting from index 1, index 0 is properties)
        for i, header in enumerate(headers):
            child_idx = i + 1  # Skip index 0 (properties group)
            if child_idx < self._group.num_children and self._group.is_child_group(child_idx):
                child_group = self._group.get_group(child_idx)
                child_obj = AlembicObject(header, child_group, self._archive)
                self._children[header.name] = child_obj
    
    def _load_properties(self):
        """Load properties from the group."""
        self._properties = {}
        
        if self._group is None or self._group.num_children == 0:
            return
        
        # Properties are in child 0 if it's a group
        if self._group.is_child_group(0):
            props_group = self._group.get_group(0)
            if props_group and props_group.num_children > 0:
                self._parse_property_group(props_group)
    
    def _parse_property_group(self, group: OgawaGroup):
        """Parse a compound property group."""
        if group.num_children == 0:
            return
        
        # Read property headers
        headers = self._read_property_headers(group)
        
        # Create properties (property data comes before headers)
        for i, header in enumerate(headers):
            # Property data group/data location varies
            prop_group = None
            if i < group.num_children - 1:  # -1 because last is headers
                child = group.get_child(i)
                if isinstance(child, OgawaGroup):
                    prop_group = child
            
            prop = AlembicProperty(header, prop_group, self._archive)
            self._properties[header.name] = prop
    
    def _read_object_headers(self) -> List[ObjectHeader]:
        """Read object headers from the group's last data child."""
        headers = []
        
        if self._group.num_children == 0:
            return headers
        
        # Find last data child
        last_idx = self._group.num_children - 1
        if not self._group.is_child_data(last_idx):
            return headers
        
        data = self._group.get_data(last_idx)
        if data is None or data.size <= 32:
            return headers
        
        buf = data.read()
        # Skip the last 32 bytes (hashes)
        buf = buf[:-32] if len(buf) > 32 else buf
        
        pos = 0
        while pos < len(buf):
            if pos + 4 > len(buf):
                break
            
            # Name size (4 bytes)
            name_size = struct.unpack('<I', buf[pos:pos+4])[0]
            pos += 4
            
            if name_size == 0 or pos + name_size + 1 > len(buf):
                break
            
            # Name
            name = buf[pos:pos+name_size].decode('utf-8', errors='replace')
            pos += name_size
            
            # Metadata index (1 byte)
            meta_index = buf[pos]
            pos += 1
            
            metadata = MetaData()
            
            if meta_index == 0xff:
                # Inline metadata
                if pos + 4 > len(buf):
                    break
                meta_size = struct.unpack('<I', buf[pos:pos+4])[0]
                pos += 4
                if pos + meta_size <= len(buf):
                    meta_str = buf[pos:pos+meta_size].decode('utf-8', errors='replace')
                    metadata.deserialize(meta_str)
                    pos += meta_size
            elif meta_index < len(self._archive.indexed_metadata):
                metadata = self._archive.indexed_metadata[meta_index]
            
            header = ObjectHeader(
                name=name,
                full_name=self.full_name + "/" + name,
                metadata=metadata
            )
            headers.append(header)
        
        return headers
    
    def _read_property_headers(self, group: OgawaGroup) -> List[PropertyHeader]:
        """Read property headers from a compound property group."""
        headers = []
        
        # Find the data child containing headers (usually last)
        data_idx = -1
        for i in range(group.num_children - 1, -1, -1):
            if group.is_child_data(i):
                data_idx = i
                break
        
        if data_idx < 0:
            return headers
        
        data = group.get_data(data_idx)
        if data is None or data.size == 0:
            return headers
        
        buf = data.read()
        pos = 0
        
        while pos < len(buf):
            if pos + 4 > len(buf):
                break
            
            # Info word (4 bytes)
            info = struct.unpack('<I', buf[pos:pos+4])[0]
            pos += 4
            
            # Extract property type
            ptype = info & 0x0003
            prop_type = PropertyType.COMPOUND if ptype == 0 else (
                PropertyType.SCALAR if ptype == 1 else PropertyType.ARRAY
            )
            
            # Size hint for variable-length fields
            size_hint = (info & 0x000c) >> 2
            
            # Data type info (for non-compound)
            data_type = None
            num_samples = 0
            first_changed = 0
            last_changed = 0
            ts_index = 0
            is_scalar_like = bool(ptype & 1)
            is_homogeneous = False
            
            if prop_type != PropertyType.COMPOUND:
                # POD type (bits 4-7)
                pod_val = (info & 0x00f0) >> 4
                pod = PlainOldDataType(pod_val) if pod_val <= 13 else PlainOldDataType.FLOAT64
                
                # Extent (bits 12-19)
                extent = (info & 0xff000) >> 12
                if extent == 0:
                    extent = 1
                
                data_type = DataType(pod=pod, extent=extent)
                is_homogeneous = bool(info & 0x400)
                
                # Number of samples
                num_samples = self._read_uint_with_hint(buf, size_hint, pos)
                pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
                
                # First/last changed indices
                if info & 0x0200:
                    first_changed = self._read_uint_with_hint(buf, size_hint, pos)
                    pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
                    last_changed = self._read_uint_with_hint(buf, size_hint, pos)
                    pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
                elif info & 0x800:
                    first_changed = 0
                    last_changed = 0
                else:
                    first_changed = 1
                    last_changed = num_samples - 1 if num_samples > 0 else 0
                
                # Time sampling index
                if info & 0x0100:
                    ts_index = self._read_uint_with_hint(buf, size_hint, pos)
                    pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
            
            # Name size
            if pos >= len(buf):
                break
            name_size = self._read_uint_with_hint(buf, size_hint, pos)
            pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
            
            if name_size == 0 or pos + name_size > len(buf):
                break
            
            # Name
            name = buf[pos:pos+name_size].decode('utf-8', errors='replace')
            pos += name_size
            
            # Metadata
            metadata = MetaData()
            meta_index = (info & 0xff00000) >> 20
            
            if meta_index == 0xff:
                meta_size = self._read_uint_with_hint(buf, size_hint, pos)
                pos += [1, 2, 4][size_hint] if size_hint < 3 else 4
                if meta_size > 0 and pos + meta_size <= len(buf):
                    meta_str = buf[pos:pos+meta_size].decode('utf-8', errors='replace')
                    metadata.deserialize(meta_str)
                    pos += meta_size
            elif meta_index < len(self._archive.indexed_metadata):
                metadata = self._archive.indexed_metadata[meta_index]
            
            header = PropertyHeader(
                name=name,
                property_type=prop_type,
                data_type=data_type,
                metadata=metadata,
                is_scalar_like=is_scalar_like,
                is_homogeneous=is_homogeneous,
                num_samples=num_samples,
                first_changed_index=first_changed,
                last_changed_index=last_changed,
                time_sampling_index=ts_index
            )
            headers.append(header)
        
        return headers
    
    def _read_uint_with_hint(self, buf: bytes, hint: int, pos: int) -> int:
        """Read an unsigned int with size determined by hint."""
        if hint == 0 and pos + 1 <= len(buf):
            return buf[pos]
        elif hint == 1 and pos + 2 <= len(buf):
            return struct.unpack('<H', buf[pos:pos+2])[0]
        elif pos + 4 <= len(buf):
            return struct.unpack('<I', buf[pos:pos+4])[0]
        return 0


class AlembicArchive:
    """
    Represents an Alembic archive (.abc file).
    
    This is the top-level container that provides access to the
    entire Alembic file structure.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._reader: Optional[OgawaReader] = None
        self._root_object: Optional[AlembicObject] = None
        self.indexed_metadata: List[MetaData] = [MetaData()]  # Index 0 = empty
        self.time_samplings: List[Any] = []
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def open(self):
        """Open the archive for reading."""
        self._reader = OgawaReader(self.filepath)
        self._reader.open()
        self._load_archive_info()
        self._load_root_object()
        logger.info(f"Opened Alembic archive: {self.filepath}")
    
    def close(self):
        """Close the archive."""
        if self._reader:
            self._reader.close()
            self._reader = None
    
    def get_root(self) -> Optional[AlembicObject]:
        """Get the root object."""
        return self._root_object
    
    def find_cameras(self) -> List[AlembicObject]:
        """Find all camera objects in the archive."""
        if self._root_object is None:
            return []
        return self._root_object.find_cameras()
    
    def find_xforms(self) -> List[AlembicObject]:
        """Find all xform objects in the archive."""
        if self._root_object is None:
            return []
        return self._root_object.find_xforms()
    
    def _load_archive_info(self):
        """Load archive-level information."""
        root_group = self._reader.get_root_group()
        
        if root_group.num_children == 0:
            return
        
        # Archive info is typically in specific children of root
        # Child 0: Usually root object group
        # Other children may contain time samplings, metadata, etc.
        
        for i in range(root_group.num_children):
            if root_group.is_child_data(i):
                data = root_group.get_data(i)
                if data and data.size > 0:
                    # Try to parse as indexed metadata
                    self._try_load_indexed_metadata(data)
    
    def _try_load_indexed_metadata(self, data: OgawaData):
        """Try to load indexed metadata from a data block."""
        if data.size > 65536:  # Metadata blocks shouldn't be huge
            return
        
        try:
            buf = data.read()
            pos = 0
            while pos < len(buf):
                if pos + 1 > len(buf):
                    break
                
                size = buf[pos]
                pos += 1
                
                if size == 0 or pos + size > len(buf):
                    break
                
                meta_str = buf[pos:pos+size].decode('utf-8', errors='replace')
                pos += size
                
                md = MetaData()
                md.deserialize(meta_str)
                self.indexed_metadata.append(md)
        except Exception as e:
            logger.debug(f"Failed to parse metadata: {e}")
    
    def _load_root_object(self):
        """Load the root object from the archive."""
        root_group = self._reader.get_root_group()
        
        if root_group.num_children == 0:
            return
        
        # Find the root object group (usually first group child)
        root_obj_group = None
        for i in range(root_group.num_children):
            if root_group.is_child_group(i):
                root_obj_group = root_group.get_group(i)
                break
        
        if root_obj_group is None:
            return
        
        header = ObjectHeader(name="ABC", full_name="", metadata=MetaData())
        self._root_object = AlembicObject(header, root_obj_group, self)


def open_archive(filepath: str) -> AlembicArchive:
    """Convenience function to open an Alembic archive."""
    archive = AlembicArchive(filepath)
    archive.open()
    return archive


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with AlembicArchive(sys.argv[1]) as archive:
            print(f"Archive: {archive.filepath}")
            
            cameras = archive.find_cameras()
            print(f"Found {len(cameras)} cameras:")
            for cam in cameras:
                print(f"  - {cam.full_name}")
            
            xforms = archive.find_xforms()
            print(f"Found {len(xforms)} xforms:")
            for xf in xforms:
                print(f"  - {xf.full_name}")
