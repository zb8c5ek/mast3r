"""
MOD_ModelConverter - Low-level Alembic (.abc) to COLMAP converter

This module provides a from-scratch implementation of an Alembic parser
that reads the Ogawa binary format and converts camera data to COLMAP format.

The implementation is based on:
- Ogawa Specification: https://github.com/alembic/alembic/wiki/Ogawa-Specification
- Alembic C++ source: https://github.com/alembic/alembic

Usage:
    # Simple conversion
    from MOD_ModelConverter import abc_to_colmap
    result = abc_to_colmap("input.abc", "output_dir/")
    
    # Inspect file structure
    from MOD_ModelConverter import inspect_abc_file
    info = inspect_abc_file("input.abc")
    
    # Low-level access
    from MOD_ModelConverter import AlembicArchive, extract_cameras
    with AlembicArchive("input.abc") as archive:
        cameras = extract_cameras(archive)

Author: Generated for mast3r project
"""

# Low-level Ogawa binary layer
from .ogawa_parser import OgawaReader, OgawaGroup, OgawaData

# Alembic semantic layer
from .alembic_parser import AlembicArchive, AlembicObject, AlembicProperty

# Camera extraction
from .camera_extractor import (
    CameraData, 
    CameraIntrinsics,
    TransformData, 
    extract_cameras,
    extract_cameras_with_animation
)

# COLMAP export
from .colmap_exporter import (
    export_to_colmap_txt,
    export_cameras_debug,
    CameraModel,
    COLMAPCamera,
    COLMAPImage
)

# High-level conversion API
from .converter import abc_to_colmap, inspect_abc_file

__all__ = [
    # Ogawa layer
    'OgawaReader',
    'OgawaGroup', 
    'OgawaData',
    
    # Alembic layer
    'AlembicArchive',
    'AlembicObject',
    'AlembicProperty',
    
    # Camera extraction
    'CameraData',
    'CameraIntrinsics',
    'TransformData',
    'extract_cameras',
    'extract_cameras_with_animation',
    
    # COLMAP export
    'export_to_colmap_txt',
    'export_cameras_debug',
    'CameraModel',
    'COLMAPCamera',
    'COLMAPImage',
    
    # High-level API
    'abc_to_colmap',
    'inspect_abc_file',
]
