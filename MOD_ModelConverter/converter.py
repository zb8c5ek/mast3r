"""
Main converter interface for Alembic to COLMAP conversion.

This module provides a high-level API for converting Alembic (.abc) files
to COLMAP format, as well as command-line interface.

Usage:
    # Python API
    from MOD_ModelConverter import abc_to_colmap
    result = abc_to_colmap("input.abc", "output_dir/")
    
    # Command line
    python -m MOD_ModelConverter.converter input.abc output_dir/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List, Union

from .ogawa_parser import OgawaReader
from .alembic_parser import AlembicArchive
from .camera_extractor import CameraData, extract_cameras, extract_cameras_with_animation
from .colmap_exporter import (
    export_to_colmap_txt, 
    export_cameras_debug,
    CameraModel
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO):
    """Configure logging for the converter."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def inspect_abc_file(abc_path: Union[str, Path], max_depth: int = 5) -> Dict:
    """
    Inspect an Alembic file and return its structure.
    
    Args:
        abc_path: Path to .abc file
        max_depth: Maximum depth to traverse
    
    Returns:
        Dictionary with file information
    """
    abc_path = Path(abc_path)
    
    if not abc_path.exists():
        raise FileNotFoundError(f"File not found: {abc_path}")
    
    info = {
        'filepath': str(abc_path),
        'filesize': abc_path.stat().st_size,
        'ogawa': {},
        'objects': [],
        'cameras': [],
        'xforms': []
    }
    
    # Inspect Ogawa layer
    with OgawaReader(abc_path) as reader:
        info['ogawa'] = {
            'valid': reader.header.is_valid,
            'frozen': reader.header.is_frozen,
            'version': reader.header.version,
            'root_offset': reader.header.root_group_offset
        }
        
        root = reader.get_root_group()
        info['ogawa']['root_children'] = root.num_children
    
    # Inspect Alembic layer
    with AlembicArchive(str(abc_path)) as archive:
        root = archive.get_root()
        if root:
            info['objects'] = _collect_objects(root, max_depth)
        
        cameras = archive.find_cameras()
        info['cameras'] = [
            {'name': c.name, 'full_name': c.full_name, 'schema': c.schema}
            for c in cameras
        ]
        
        xforms = archive.find_xforms()
        info['xforms'] = [
            {'name': x.name, 'full_name': x.full_name, 'schema': x.schema}
            for x in xforms
        ]
    
    return info


def _collect_objects(obj, max_depth: int, depth: int = 0) -> List[Dict]:
    """Recursively collect object information."""
    if depth > max_depth:
        return []
    
    result = [{
        'name': obj.name,
        'full_name': obj.full_name,
        'schema': obj.schema,
        'is_camera': obj.is_camera(),
        'is_xform': obj.is_xform(),
        'properties': list(obj.get_properties().keys()),
        'children': []
    }]
    
    for child in obj.get_children().values():
        result[0]['children'].extend(_collect_objects(child, max_depth, depth + 1))
    
    return result


def abc_to_colmap(abc_path: Union[str, Path],
                  output_dir: Union[str, Path],
                  image_width: int = 1920,
                  image_height: int = 1080,
                  camera_model: str = CameraModel.PINHOLE,
                  image_name_format: str = "{name}.png",
                  share_intrinsics: bool = False,
                  frame_range: Optional[tuple] = None,
                  debug: bool = False) -> Dict[str, Path]:
    """
    Convert Alembic file to COLMAP format.
    
    Args:
        abc_path: Path to input .abc file
        output_dir: Directory to write COLMAP files
        image_width: Image width in pixels (default: 1920)
        image_height: Image height in pixels (default: 1080)
        camera_model: COLMAP camera model (default: PINHOLE)
        image_name_format: Format string for image names
        share_intrinsics: If True, all cameras share the same intrinsics
        frame_range: Optional (start, end) frame range for animated cameras
        debug: If True, also write debug information
    
    Returns:
        Dictionary with paths to created files
    """
    abc_path = Path(abc_path)
    output_dir = Path(output_dir)
    
    logger.info(f"Converting {abc_path} to COLMAP format")
    logger.info(f"Output directory: {output_dir}")
    
    # Open and parse Alembic file
    with AlembicArchive(str(abc_path)) as archive:
        # Extract cameras
        if frame_range:
            cameras = extract_cameras_with_animation(
                archive, 
                frame_range=frame_range,
                image_width=image_width,
                image_height=image_height
            )
        else:
            cameras = extract_cameras(
                archive,
                image_width=image_width,
                image_height=image_height
            )
        
        if not cameras:
            logger.warning("No cameras found in Alembic file")
            # Create empty output
            output_dir.mkdir(parents=True, exist_ok=True)
            return export_to_colmap_txt(
                [], 
                output_dir, 
                camera_model=camera_model
            )
        
        logger.info(f"Found {len(cameras)} camera(s)")
        
        # Export to COLMAP
        result = export_to_colmap_txt(
            cameras,
            output_dir,
            camera_model=camera_model,
            image_name_format=image_name_format,
            share_intrinsics=share_intrinsics
        )
        
        # Write debug info if requested
        if debug:
            debug_path = output_dir / "cameras_debug.txt"
            export_cameras_debug(cameras, debug_path)
            result['debug'] = debug_path
        
        return result


def print_inspection(info: Dict):
    """Pretty-print inspection results."""
    print(f"\n{'='*60}")
    print(f"Alembic File Inspection")
    print(f"{'='*60}")
    print(f"File: {info['filepath']}")
    print(f"Size: {info['filesize']} bytes")
    print()
    
    print("Ogawa Layer:")
    ogawa = info['ogawa']
    print(f"  Valid: {ogawa['valid']}")
    print(f"  Frozen: {ogawa['frozen']}")
    print(f"  Version: {ogawa['version']}")
    print(f"  Root children: {ogawa['root_children']}")
    print()
    
    print(f"Cameras ({len(info['cameras'])}):")
    for cam in info['cameras']:
        print(f"  - {cam['full_name']} [{cam['schema']}]")
    print()
    
    print(f"Transforms ({len(info['xforms'])}):")
    for xf in info['xforms']:
        print(f"  - {xf['full_name']} [{xf['schema']}]")
    print()
    
    print("Object Hierarchy:")
    for obj in info['objects']:
        _print_object(obj, indent=2)


def _print_object(obj: Dict, indent: int = 0):
    """Print object info with indentation."""
    prefix = " " * indent
    markers = []
    if obj['is_camera']:
        markers.append("CAM")
    if obj['is_xform']:
        markers.append("XF")
    marker_str = f" [{', '.join(markers)}]" if markers else ""
    
    print(f"{prefix}{obj['name']}{marker_str}")
    
    if obj['properties']:
        print(f"{prefix}  Properties: {', '.join(obj['properties'][:5])}")
        if len(obj['properties']) > 5:
            print(f"{prefix}  ... and {len(obj['properties'])-5} more")
    
    for child in obj['children']:
        _print_object(child, indent + 4)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Alembic (.abc) camera data to COLMAP format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python -m MOD_ModelConverter.converter input.abc output/
  
  # With custom image size
  python -m MOD_ModelConverter.converter input.abc output/ --width 4096 --height 2160
  
  # Inspect file without converting
  python -m MOD_ModelConverter.converter input.abc --inspect
  
  # With debug output
  python -m MOD_ModelConverter.converter input.abc output/ --debug
        """
    )
    
    parser.add_argument('input', help='Input Alembic (.abc) file')
    parser.add_argument('output', nargs='?', help='Output directory for COLMAP files')
    
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect file structure without converting')
    parser.add_argument('--width', type=int, default=1920,
                       help='Image width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Image height in pixels (default: 1080)')
    parser.add_argument('--camera-model', choices=['SIMPLE_PINHOLE', 'PINHOLE', 'OPENCV'],
                       default='PINHOLE', help='COLMAP camera model (default: PINHOLE)')
    parser.add_argument('--image-format', default='{name}.png',
                       help='Image name format string (default: {name}.png)')
    parser.add_argument('--share-intrinsics', action='store_true',
                       help='Share camera intrinsics across all images')
    parser.add_argument('--frame-start', type=int, help='Start frame for animated cameras')
    parser.add_argument('--frame-end', type=int, help='End frame for animated cameras')
    parser.add_argument('--debug', action='store_true',
                       help='Write additional debug information')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    try:
        if args.inspect:
            # Inspect mode
            info = inspect_abc_file(args.input)
            print_inspection(info)
        else:
            # Convert mode
            if not args.output:
                parser.error("Output directory is required for conversion")
            
            frame_range = None
            if args.frame_start is not None or args.frame_end is not None:
                frame_range = (
                    args.frame_start if args.frame_start is not None else 0,
                    args.frame_end if args.frame_end is not None else 999999
                )
            
            result = abc_to_colmap(
                args.input,
                args.output,
                image_width=args.width,
                image_height=args.height,
                camera_model=args.camera_model,
                image_name_format=args.image_format,
                share_intrinsics=args.share_intrinsics,
                frame_range=frame_range,
                debug=args.debug
            )
            
            print(f"\nConversion complete!")
            print(f"Output files:")
            for key, path in result.items():
                if key != 'output_dir':
                    print(f"  {key}: {path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
