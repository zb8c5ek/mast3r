"""
Camera and Transform data extractor for Alembic files.

This module extracts camera intrinsic parameters and world transforms
from Alembic Camera and Xform objects.

Camera Intrinsics (from Alembic CameraSample):
- focalLength: in millimeters
- horizontalAperture: horizontal film back in centimeters  
- verticalAperture: vertical film back in centimeters
- horizontalFilmOffset: film offset in centimeters
- verticalFilmOffset: film offset in centimeters
- lensSqueezeRatio: anamorphic lens squeeze
- nearClippingPlane: in centimeters
- farClippingPlane: in centimeters

Transform Operations (from Alembic XformSample):
- Translate (3 channels)
- Rotate (4 channels: x, y, z, angle)
- Scale (3 channels)
- Matrix (16 channels: 4x4 matrix)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .alembic_parser import AlembicArchive, AlembicObject, AlembicProperty

logger = logging.getLogger(__name__)


# Default camera values (from CameraSample.h defaults)
DEFAULT_FOCAL_LENGTH = 35.0  # mm
DEFAULT_HORIZONTAL_APERTURE = 3.6  # cm (36mm full frame)
DEFAULT_VERTICAL_APERTURE = 2.4  # cm (24mm full frame)
DEFAULT_NEAR_CLIP = 0.1  # cm
DEFAULT_FAR_CLIP = 100000.0  # cm


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    focal_length: float = DEFAULT_FOCAL_LENGTH  # mm
    horizontal_aperture: float = DEFAULT_HORIZONTAL_APERTURE  # cm
    vertical_aperture: float = DEFAULT_VERTICAL_APERTURE  # cm
    horizontal_film_offset: float = 0.0  # cm
    vertical_film_offset: float = 0.0  # cm
    lens_squeeze_ratio: float = 1.0
    near_clipping_plane: float = DEFAULT_NEAR_CLIP  # cm
    far_clipping_plane: float = DEFAULT_FAR_CLIP  # cm
    
    # Overscan (for rendering)
    overscan_left: float = 0.0
    overscan_right: float = 0.0
    overscan_top: float = 0.0
    overscan_bottom: float = 0.0
    
    # Other parameters
    f_stop: float = 5.6
    focus_distance: float = 5.0  # cm
    shutter_open: float = 0.0
    shutter_close: float = 0.020833333333333332
    
    def get_field_of_view_horizontal(self) -> float:
        """Calculate horizontal field of view in degrees."""
        # FOV = 2 * atan(aperture / (2 * focal_length))
        # Note: aperture is in cm, focal length in mm
        aperture_mm = self.horizontal_aperture * 10  # Convert cm to mm
        fov_rad = 2 * math.atan(aperture_mm / (2 * self.focal_length))
        return math.degrees(fov_rad)
    
    def get_field_of_view_vertical(self) -> float:
        """Calculate vertical field of view in degrees."""
        aperture_mm = self.vertical_aperture * 10
        fov_rad = 2 * math.atan(aperture_mm / (2 * self.focal_length))
        return math.degrees(fov_rad)
    
    def get_focal_length_pixels(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert focal length to pixel units.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            (fx, fy) focal lengths in pixels
        """
        # Sensor dimensions in mm
        sensor_width_mm = self.horizontal_aperture * 10
        sensor_height_mm = self.vertical_aperture * 10
        
        # Pixels per mm
        px_per_mm_x = image_width / sensor_width_mm
        px_per_mm_y = image_height / sensor_height_mm
        
        # Focal length in pixels
        fx = self.focal_length * px_per_mm_x
        fy = self.focal_length * px_per_mm_y / self.lens_squeeze_ratio
        
        return fx, fy
    
    def get_principal_point_pixels(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Get principal point in pixel coordinates.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            (cx, cy) principal point in pixels
        """
        # Sensor dimensions in mm
        sensor_width_mm = self.horizontal_aperture * 10
        sensor_height_mm = self.vertical_aperture * 10
        
        # Film offset in mm
        offset_x_mm = self.horizontal_film_offset * 10
        offset_y_mm = self.vertical_film_offset * 10
        
        # Principal point = center + offset
        cx = image_width / 2 + (offset_x_mm / sensor_width_mm) * image_width
        cy = image_height / 2 + (offset_y_mm / sensor_height_mm) * image_height
        
        return cx, cy


@dataclass
class TransformData:
    """Camera/object transform (extrinsics)."""
    # 4x4 world-to-local transformation matrix
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # Decomposed transform components (for debugging/reference)
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # w, x, y, z
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'TransformData':
        """Create from a 4x4 transformation matrix."""
        td = cls()
        td.matrix = matrix.copy()
        td._decompose()
        return td
    
    def _decompose(self):
        """Decompose the matrix into translation, rotation, scale."""
        # Extract translation
        self.translation = self.matrix[:3, 3].copy()
        
        # Extract rotation and scale from upper 3x3
        m = self.matrix[:3, :3]
        
        # Scale is the length of each column
        self.scale = np.array([
            np.linalg.norm(m[:, 0]),
            np.linalg.norm(m[:, 1]),
            np.linalg.norm(m[:, 2])
        ])
        
        # Remove scale to get rotation matrix
        rot = m.copy()
        for i in range(3):
            if self.scale[i] > 1e-10:
                rot[:, i] /= self.scale[i]
        
        # Convert rotation matrix to quaternion
        self.rotation_quaternion = self._matrix_to_quaternion(rot)
    
    @staticmethod
    def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        # Based on: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def get_camera_position(self) -> np.ndarray:
        """Get camera position in world coordinates."""
        return self.translation.copy()
    
    def get_camera_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return self.matrix[:3, :3].copy()
    
    def get_inverse(self) -> 'TransformData':
        """Get the inverse transform."""
        inv_matrix = np.linalg.inv(self.matrix)
        return TransformData.from_matrix(inv_matrix)


@dataclass
class CameraData:
    """Complete camera data including intrinsics and extrinsics."""
    name: str
    intrinsics: CameraIntrinsics
    transform: TransformData
    frame_index: int = 0
    
    # Associated image info (if available)
    image_path: Optional[str] = None
    image_width: int = 1920
    image_height: int = 1080


def extract_camera_intrinsics(camera_obj: AlembicObject, sample_index: int = 0) -> CameraIntrinsics:
    """
    Extract camera intrinsic parameters from an Alembic Camera object.
    
    Args:
        camera_obj: Alembic camera object
        sample_index: Time sample index (for animated cameras)
    
    Returns:
        CameraIntrinsics with extracted parameters
    """
    intrinsics = CameraIntrinsics()
    
    props = camera_obj.get_properties()
    logger.debug(f"Camera {camera_obj.name} has properties: {list(props.keys())}")
    
    # Core camera properties are typically stored as a scalar array
    # with 16 double values in this order (from CameraSample.h):
    # 0: focalLength (mm)
    # 1: horizontalAperture (cm)
    # 2: horizontalFilmOffset (cm)
    # 3: verticalAperture (cm)
    # 4: verticalFilmOffset (cm)
    # 5: lensSqueezeRatio
    # 6: overscanLeft
    # 7: overscanRight
    # 8: overscanTop
    # 9: overscanBottom
    # 10: fStop
    # 11: focusDistance (cm)
    # 12: shutterOpen
    # 13: shutterClose
    # 14: nearClippingPlane (cm)
    # 15: farClippingPlane (cm)
    
    # Look for the core values property
    for prop_name in ['.vals', '.coreVals', 'coreValues', '.camera']:
        if prop_name in props:
            prop = props[prop_name]
            samples = prop.get_all_samples()
            if samples and len(samples) > sample_index:
                values = samples[sample_index]
                if isinstance(values, (list, tuple)) and len(values) >= 16:
                    # Extract all 16 core values
                    if isinstance(values[0], (list, tuple)):
                        # Nested structure
                        values = [v[0] if isinstance(v, (list, tuple)) else v for v in values]
                    
                    intrinsics.focal_length = float(values[0])
                    intrinsics.horizontal_aperture = float(values[1])
                    intrinsics.horizontal_film_offset = float(values[2])
                    intrinsics.vertical_aperture = float(values[3])
                    intrinsics.vertical_film_offset = float(values[4])
                    intrinsics.lens_squeeze_ratio = float(values[5])
                    intrinsics.overscan_left = float(values[6])
                    intrinsics.overscan_right = float(values[7])
                    intrinsics.overscan_top = float(values[8])
                    intrinsics.overscan_bottom = float(values[9])
                    intrinsics.f_stop = float(values[10])
                    intrinsics.focus_distance = float(values[11])
                    intrinsics.shutter_open = float(values[12])
                    intrinsics.shutter_close = float(values[13])
                    intrinsics.near_clipping_plane = float(values[14])
                    intrinsics.far_clipping_plane = float(values[15])
                    
                    logger.debug(f"Extracted camera intrinsics: focal={intrinsics.focal_length}mm, "
                               f"aperture={intrinsics.horizontal_aperture}x{intrinsics.vertical_aperture}cm")
                    break
    
    # Also check for individual properties (some exporters use this format)
    if 'focalLength' in props:
        samples = props['focalLength'].get_all_samples()
        if samples:
            intrinsics.focal_length = float(samples[min(sample_index, len(samples)-1)])
    
    if 'horizontalAperture' in props:
        samples = props['horizontalAperture'].get_all_samples()
        if samples:
            intrinsics.horizontal_aperture = float(samples[min(sample_index, len(samples)-1)])
    
    if 'verticalAperture' in props:
        samples = props['verticalAperture'].get_all_samples()
        if samples:
            intrinsics.vertical_aperture = float(samples[min(sample_index, len(samples)-1)])
    
    return intrinsics


def extract_transform(xform_obj: AlembicObject, sample_index: int = 0) -> TransformData:
    """
    Extract transform data from an Alembic Xform object.
    
    Args:
        xform_obj: Alembic xform object
        sample_index: Time sample index
    
    Returns:
        TransformData with the transformation matrix
    """
    transform = TransformData()
    
    props = xform_obj.get_properties()
    logger.debug(f"Xform {xform_obj.name} has properties: {list(props.keys())}")
    
    # Xform operations are stored in specific properties:
    # - .ops: Operation types (encoded bytes)
    # - .vals: Operation values (doubles)
    # - .inherits: Whether to inherit parent transform (bool)
    
    # Try to get the transform matrix directly
    for prop_name in ['.vals', 'vals', '.xform', 'matrix']:
        if prop_name in props:
            prop = props[prop_name]
            samples = prop.get_all_samples()
            if samples and len(samples) > sample_index:
                values = samples[sample_index]
                
                # Check if it's a 4x4 matrix (16 values)
                if isinstance(values, (list, tuple)):
                    flat_values = []
                    for v in values:
                        if isinstance(v, (list, tuple)):
                            flat_values.extend(v)
                        else:
                            flat_values.append(v)
                    
                    if len(flat_values) >= 16:
                        # Interpret as row-major 4x4 matrix
                        matrix = np.array(flat_values[:16], dtype=np.float64).reshape(4, 4)
                        transform = TransformData.from_matrix(matrix)
                        logger.debug(f"Extracted 4x4 transform matrix from {prop_name}")
                        return transform
    
    # If no matrix found, try to compose from individual operations
    # This is more complex and depends on the specific operation encoding
    
    # Try individual translate/rotate/scale properties
    translation = np.zeros(3)
    rotation = np.eye(3)
    scale = np.ones(3)
    
    for prop_name in ['translate', '.translate', 'translation']:
        if prop_name in props:
            samples = props[prop_name].get_all_samples()
            if samples:
                val = samples[min(sample_index, len(samples)-1)]
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    translation = np.array(val[:3], dtype=np.float64)
    
    for prop_name in ['scale', '.scale']:
        if prop_name in props:
            samples = props[prop_name].get_all_samples()
            if samples:
                val = samples[min(sample_index, len(samples)-1)]
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    scale = np.array(val[:3], dtype=np.float64)
    
    # Build matrix from components
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * scale.reshape(1, 3)
    matrix[:3, 3] = translation
    
    transform = TransformData.from_matrix(matrix)
    return transform


def get_world_transform(obj: AlembicObject, sample_index: int = 0) -> TransformData:
    """
    Get the world transform for an object by traversing the hierarchy.
    
    This accumulates transforms from root to the object.
    """
    # Build the transform chain from root to object
    # (This requires access to parent objects, which our current implementation
    #  doesn't directly support - we'd need to traverse from root)
    
    # For now, just return the local transform
    if obj.is_xform():
        return extract_transform(obj, sample_index)
    
    return TransformData()


def extract_cameras(archive: AlembicArchive, 
                   image_width: int = 1920, 
                   image_height: int = 1080) -> List[CameraData]:
    """
    Extract all cameras from an Alembic archive.
    
    Args:
        archive: Opened AlembicArchive
        image_width: Default image width for cameras
        image_height: Default image height for cameras
    
    Returns:
        List of CameraData objects
    """
    cameras = []
    
    # Find all camera objects
    camera_objects = archive.find_cameras()
    
    for cam_obj in camera_objects:
        logger.info(f"Processing camera: {cam_obj.full_name}")
        
        # Extract intrinsics
        intrinsics = extract_camera_intrinsics(cam_obj)
        
        # Get transform from parent xform (if exists)
        # For now, use identity transform
        transform = TransformData()
        
        # Try to find parent xform
        # The path might be like /Root/CameraXform/Camera
        # We need to find the Xform parent
        parent_path = '/'.join(cam_obj.full_name.split('/')[:-1])
        root = archive.get_root()
        
        if root:
            # Search for xform in parent path
            xforms = archive.find_xforms()
            for xf in xforms:
                if xf.full_name in cam_obj.full_name:
                    transform = extract_transform(xf)
                    logger.debug(f"Found parent xform: {xf.full_name}")
        
        camera = CameraData(
            name=cam_obj.name,
            intrinsics=intrinsics,
            transform=transform,
            image_width=image_width,
            image_height=image_height
        )
        cameras.append(camera)
    
    logger.info(f"Extracted {len(cameras)} cameras from archive")
    return cameras


def extract_cameras_with_animation(archive: AlembicArchive,
                                  frame_range: Optional[Tuple[int, int]] = None,
                                  image_width: int = 1920,
                                  image_height: int = 1080) -> List[CameraData]:
    """
    Extract cameras with animation samples.
    
    Args:
        archive: Opened AlembicArchive
        frame_range: Optional (start_frame, end_frame) tuple
        image_width: Default image width
        image_height: Default image height
    
    Returns:
        List of CameraData objects, one per camera per frame
    """
    cameras = []
    camera_objects = archive.find_cameras()
    
    for cam_obj in camera_objects:
        # Determine number of samples
        props = cam_obj.get_properties()
        num_samples = 1
        
        for prop in props.values():
            if prop.header.num_samples > num_samples:
                num_samples = prop.header.num_samples
        
        # Apply frame range
        start_frame = 0
        end_frame = num_samples
        if frame_range:
            start_frame = max(0, frame_range[0])
            end_frame = min(num_samples, frame_range[1])
        
        for frame_idx in range(start_frame, end_frame):
            intrinsics = extract_camera_intrinsics(cam_obj, frame_idx)
            
            # Get transform
            transform = TransformData()
            xforms = archive.find_xforms()
            for xf in xforms:
                if xf.full_name in cam_obj.full_name:
                    transform = extract_transform(xf, frame_idx)
                    break
            
            camera = CameraData(
                name=f"{cam_obj.name}_{frame_idx:04d}",
                intrinsics=intrinsics,
                transform=transform,
                frame_index=frame_idx,
                image_width=image_width,
                image_height=image_height
            )
            cameras.append(camera)
    
    return cameras
