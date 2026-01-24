"""
COLMAP .txt format exporter.

This module exports camera data to COLMAP's text format, which consists of:
- cameras.txt: Camera intrinsic parameters
- images.txt: Camera poses and image associations
- points3D.txt: 3D points (empty for camera-only export)

COLMAP text file formats:
https://colmap.github.io/format.html

cameras.txt format:
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: N
<camera_id> <model> <width> <height> <params>

Supported camera models:
- SIMPLE_PINHOLE: f, cx, cy
- PINHOLE: fx, fy, cx, cy
- SIMPLE_RADIAL: f, cx, cy, k
- RADIAL: f, cx, cy, k1, k2
- OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
- FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

images.txt format:
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: N, mean observations per image: X
<image_id> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <camera_id> <name>
<x1> <y1> <point3d_id1> <x2> <y2> <point3d_id2> ...

points3D.txt format:
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: N, mean track length: X
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from .camera_extractor import CameraData, CameraIntrinsics, TransformData

logger = logging.getLogger(__name__)


# COLMAP camera model IDs
class CameraModel:
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    PINHOLE = "PINHOLE"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    RADIAL = "RADIAL"
    OPENCV = "OPENCV"
    FULL_OPENCV = "FULL_OPENCV"


@dataclass
class COLMAPCamera:
    """COLMAP camera representation."""
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]
    
    def to_line(self) -> str:
        """Convert to COLMAP cameras.txt line format."""
        params_str = ' '.join(f'{p:.12g}' for p in self.params)
        return f"{self.camera_id} {self.model} {self.width} {self.height} {params_str}"


@dataclass
class COLMAPImage:
    """COLMAP image representation."""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Tuple[float, float, int]] = None  # (x, y, point3d_id)
    
    def to_lines(self) -> List[str]:
        """Convert to COLMAP images.txt format (2 lines)."""
        line1 = f"{self.image_id} {self.qw:.12g} {self.qx:.12g} {self.qy:.12g} {self.qz:.12g} " \
                f"{self.tx:.12g} {self.ty:.12g} {self.tz:.12g} {self.camera_id} {self.name}"
        
        # Second line: 2D point observations (empty if no points)
        if self.points2d:
            points_str = ' '.join(f'{x:.6g} {y:.6g} {pid}' for x, y, pid in self.points2d)
            line2 = points_str
        else:
            line2 = ""
        
        return [line1, line2]


def convert_intrinsics_to_colmap(intrinsics: CameraIntrinsics, 
                                  width: int, 
                                  height: int,
                                  model: str = CameraModel.PINHOLE) -> COLMAPCamera:
    """
    Convert Alembic camera intrinsics to COLMAP camera format.
    
    Args:
        intrinsics: Alembic camera intrinsics
        width: Image width in pixels
        height: Image height in pixels
        model: COLMAP camera model to use
    
    Returns:
        COLMAPCamera object
    """
    fx, fy = intrinsics.get_focal_length_pixels(width, height)
    cx, cy = intrinsics.get_principal_point_pixels(width, height)
    
    if model == CameraModel.SIMPLE_PINHOLE:
        # Use average focal length
        f = (fx + fy) / 2
        params = [f, cx, cy]
    elif model == CameraModel.PINHOLE:
        params = [fx, fy, cx, cy]
    elif model == CameraModel.SIMPLE_RADIAL:
        f = (fx + fy) / 2
        params = [f, cx, cy, 0.0]  # k = 0 (no distortion)
    elif model == CameraModel.RADIAL:
        f = (fx + fy) / 2
        params = [f, cx, cy, 0.0, 0.0]  # k1 = k2 = 0
    elif model == CameraModel.OPENCV:
        params = [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0]  # No distortion
    else:
        # Default to PINHOLE
        params = [fx, fy, cx, cy]
        model = CameraModel.PINHOLE
    
    return COLMAPCamera(
        camera_id=0,  # Will be assigned later
        model=model,
        width=width,
        height=height,
        params=params
    )


def convert_transform_to_colmap(transform: TransformData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Alembic transform to COLMAP format.
    
    COLMAP uses camera-to-world convention for storage but internally
    works with world-to-camera. The images.txt stores the camera pose
    as the transformation from world to camera coordinates.
    
    Args:
        transform: Alembic transform (typically world-to-local)
    
    Returns:
        (quaternion, translation) in COLMAP format
        quaternion is (w, x, y, z)
        translation is (tx, ty, tz)
    """
    # Get the 4x4 matrix
    matrix = transform.matrix.copy()
    
    # COLMAP coordinate system: +X right, +Y down, +Z forward
    # Alembic/Maya: +X right, +Y up, -Z forward
    # Need to apply coordinate system conversion
    
    # Coordinate transform matrix (Alembic to COLMAP)
    # Flip Y and Z
    coord_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)
    
    # Apply coordinate transform
    colmap_matrix = coord_transform @ matrix @ np.linalg.inv(coord_transform)
    
    # For COLMAP images.txt, we need world-to-camera transform
    # If Alembic gives us camera-to-world, we need to invert
    # The convention varies by exporter, so we'll assume camera-to-world
    # and invert it
    w2c = np.linalg.inv(colmap_matrix)
    
    # Extract rotation and translation
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    
    # Convert rotation matrix to quaternion
    q = rotation_matrix_to_quaternion(R)
    
    return q, t


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    
    Uses the algorithm from:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Normalize quaternion
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    
    return q


def write_cameras_txt(cameras: List[COLMAPCamera], filepath: Union[str, Path]):
    """
    Write cameras.txt file.
    
    Args:
        cameras: List of COLMAPCamera objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        
        for cam in cameras:
            f.write(cam.to_line() + "\n")
    
    logger.info(f"Wrote {len(cameras)} cameras to {filepath}")


def write_images_txt(images: List[COLMAPImage], filepath: Union[str, Path]):
    """
    Write images.txt file.
    
    Args:
        images: List of COLMAPImage objects
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        
        for img in images:
            lines = img.to_lines()
            f.write(lines[0] + "\n")
            f.write(lines[1] + "\n")
    
    logger.info(f"Wrote {len(images)} images to {filepath}")


def write_points3d_txt(filepath: Union[str, Path]):
    """
    Write empty points3D.txt file.
    
    Args:
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")
    
    logger.info(f"Wrote empty points3D.txt to {filepath}")


def export_to_colmap_txt(cameras: List[CameraData],
                         output_dir: Union[str, Path],
                         camera_model: str = CameraModel.PINHOLE,
                         image_name_format: str = "{name}.png",
                         share_intrinsics: bool = False) -> Dict[str, Path]:
    """
    Export camera data to COLMAP text format.
    
    Args:
        cameras: List of CameraData from Alembic
        output_dir: Directory to write COLMAP files
        camera_model: COLMAP camera model to use
        image_name_format: Format string for image names (can use {name}, {frame}, {index})
        share_intrinsics: If True, all cameras share the same intrinsic parameters
    
    Returns:
        Dictionary with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colmap_cameras = []
    colmap_images = []
    
    # Camera intrinsics mapping
    # If share_intrinsics is True, we'll create one camera and reference it
    # Otherwise, each camera gets its own intrinsic parameters
    
    intrinsics_map: Dict[str, int] = {}  # hash -> camera_id
    camera_id_counter = 1
    
    for idx, cam in enumerate(cameras):
        # Convert intrinsics
        colmap_cam = convert_intrinsics_to_colmap(
            cam.intrinsics,
            cam.image_width,
            cam.image_height,
            model=camera_model
        )
        
        if share_intrinsics:
            # All cameras share the same intrinsics
            if len(colmap_cameras) == 0:
                colmap_cam.camera_id = camera_id_counter
                colmap_cameras.append(colmap_cam)
            camera_id = 1
        else:
            # Create hash for intrinsics to detect duplicates
            params_hash = f"{cam.image_width}_{cam.image_height}_" + \
                         "_".join(f"{p:.6f}" for p in colmap_cam.params)
            
            if params_hash in intrinsics_map:
                camera_id = intrinsics_map[params_hash]
            else:
                colmap_cam.camera_id = camera_id_counter
                colmap_cameras.append(colmap_cam)
                intrinsics_map[params_hash] = camera_id_counter
                camera_id = camera_id_counter
                camera_id_counter += 1
        
        # Convert transform
        q, t = convert_transform_to_colmap(cam.transform)
        
        # Generate image name
        image_name = image_name_format.format(
            name=cam.name,
            frame=cam.frame_index,
            index=idx
        )
        
        colmap_image = COLMAPImage(
            image_id=idx + 1,
            qw=q[0],
            qx=q[1],
            qy=q[2],
            qz=q[3],
            tx=t[0],
            ty=t[1],
            tz=t[2],
            camera_id=camera_id,
            name=image_name,
            points2d=[]
        )
        colmap_images.append(colmap_image)
    
    # Write files
    cameras_path = output_dir / "cameras.txt"
    images_path = output_dir / "images.txt"
    points3d_path = output_dir / "points3D.txt"
    
    write_cameras_txt(colmap_cameras, cameras_path)
    write_images_txt(colmap_images, images_path)
    write_points3d_txt(points3d_path)
    
    logger.info(f"Exported COLMAP data to {output_dir}")
    logger.info(f"  - {len(colmap_cameras)} unique camera models")
    logger.info(f"  - {len(colmap_images)} images")
    
    return {
        'cameras': cameras_path,
        'images': images_path,
        'points3D': points3d_path,
        'output_dir': output_dir
    }


def export_cameras_debug(cameras: List[CameraData], output_file: Union[str, Path]):
    """
    Export camera data to a debug-friendly format for inspection.
    
    Args:
        cameras: List of CameraData
        output_file: Output file path
    """
    output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        f.write("# Alembic Camera Debug Export\n")
        f.write(f"# Number of cameras: {len(cameras)}\n\n")
        
        for idx, cam in enumerate(cameras):
            f.write(f"=== Camera {idx}: {cam.name} ===\n")
            f.write(f"Frame Index: {cam.frame_index}\n")
            f.write(f"Image Size: {cam.image_width} x {cam.image_height}\n")
            f.write(f"\n")
            
            # Intrinsics
            intr = cam.intrinsics
            f.write("Intrinsics (Alembic native):\n")
            f.write(f"  Focal Length: {intr.focal_length} mm\n")
            f.write(f"  Horizontal Aperture: {intr.horizontal_aperture} cm\n")
            f.write(f"  Vertical Aperture: {intr.vertical_aperture} cm\n")
            f.write(f"  Horizontal Film Offset: {intr.horizontal_film_offset} cm\n")
            f.write(f"  Vertical Film Offset: {intr.vertical_film_offset} cm\n")
            f.write(f"  Lens Squeeze Ratio: {intr.lens_squeeze_ratio}\n")
            f.write(f"  Near/Far Clip: {intr.near_clipping_plane} / {intr.far_clipping_plane} cm\n")
            f.write(f"\n")
            
            # Computed values
            fx, fy = intr.get_focal_length_pixels(cam.image_width, cam.image_height)
            cx, cy = intr.get_principal_point_pixels(cam.image_width, cam.image_height)
            f.write("Computed (pixels):\n")
            f.write(f"  fx: {fx:.4f}\n")
            f.write(f"  fy: {fy:.4f}\n")
            f.write(f"  cx: {cx:.4f}\n")
            f.write(f"  cy: {cy:.4f}\n")
            f.write(f"  FOV (H): {intr.get_field_of_view_horizontal():.2f} deg\n")
            f.write(f"  FOV (V): {intr.get_field_of_view_vertical():.2f} deg\n")
            f.write(f"\n")
            
            # Transform
            xf = cam.transform
            f.write("Transform:\n")
            f.write(f"  Translation: [{xf.translation[0]:.6f}, {xf.translation[1]:.6f}, {xf.translation[2]:.6f}]\n")
            f.write(f"  Quaternion (wxyz): [{xf.rotation_quaternion[0]:.6f}, {xf.rotation_quaternion[1]:.6f}, {xf.rotation_quaternion[2]:.6f}, {xf.rotation_quaternion[3]:.6f}]\n")
            f.write(f"  Scale: [{xf.scale[0]:.6f}, {xf.scale[1]:.6f}, {xf.scale[2]:.6f}]\n")
            f.write(f"  Matrix:\n")
            for row in xf.matrix:
                f.write(f"    [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}, {row[3]:12.6f}]\n")
            f.write(f"\n")
            
            # COLMAP format
            q, t = convert_transform_to_colmap(xf)
            f.write("COLMAP Format:\n")
            f.write(f"  Quaternion (wxyz): [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]\n")
            f.write(f"  Translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")
            f.write(f"\n\n")
    
    logger.info(f"Wrote debug export to {output_file}")
