"""
kern_trajectory_analyze - Extract and Analyze Camera Trajectories
=================================================================
Core kernel functions for extracting camera trajectories from reconstructions.

Trajectory data includes:
- Timestamps (extracted from filename as strings)
- Translation (camera position in world coordinates)
- Orientation (rotation matrix or quaternion)

Filename format expected: XXXXXX_TIMESTAMP1_TIMESTAMP2_camX.ext
- XXXXXX: frame number
- TIMESTAMP1: first timestamp string (e.g., '9223372199204787635')
- TIMESTAMP2: second timestamp string (e.g., '1768981867560696696')
- camX: camera identifier
"""
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation


@dataclass
class CameraPose:
    """Single camera pose at a specific timestamp."""
    frame_idx: int              # Frame index from filename
    timestamp_str1: str         # First timestamp string from filename
    timestamp_str2: str         # Second timestamp string from filename
    image_name: str             # Original image filename
    camera_id: str              # Camera identifier (e.g., 'cam0')
    
    # Position (translation)
    position: np.ndarray        # 3D position in world coordinates [x, y, z]
    
    # Orientation
    rotation_matrix: np.ndarray # 3x3 rotation matrix (world-to-camera)
    quaternion: np.ndarray      # [w, x, y, z] quaternion
    euler_angles: np.ndarray    # [roll, pitch, yaw] in degrees
    
    # Full transform
    cam_from_world: np.ndarray  # 4x4 transformation matrix
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'frame_idx': self.frame_idx,
            'timestamp_str1': self.timestamp_str1,
            'timestamp_str2': self.timestamp_str2,
            'image_name': self.image_name,
            'camera_id': self.camera_id,
            'position': self.position.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist(),
            'quaternion': self.quaternion.tolist(),
            'euler_angles': self.euler_angles.tolist(),
            'cam_from_world': self.cam_from_world.tolist(),
        }


@dataclass
class CameraTrajectory:
    """Complete trajectory for a single camera over time."""
    camera_id: str
    poses: List[CameraPose] = field(default_factory=list)
    
    # Computed trajectory statistics
    path_length: float = 0.0
    num_poses: int = 0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'camera_id': self.camera_id,
            'num_poses': self.num_poses,
            'path_length': self.path_length,
            'first_frame': self.poses[0].frame_idx if self.poses else -1,
            'last_frame': self.poses[-1].frame_idx if self.poses else -1,
            'poses': [p.to_dict() for p in self.poses],
        }


def parse_timestamp_from_filename(filename: str) -> Tuple[int, str, str, str]:
    """
    Parse frame index, timestamps, and camera ID from filename.
    
    Expected format: XXXXXX_TIMESTAMP1_TIMESTAMP2_camX.ext
    
    Args:
        filename: Image filename (e.g., '000000_9223372199054785510_1768981867363745461_cam0.jpg')
    
    Returns:
        (frame_idx, timestamp_str1, timestamp_str2, camera_id)
        Returns (-1, '', '', '') if parsing fails
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try to parse: XXXXXX_TIMESTAMP1_TIMESTAMP2_camX
    parts = name.split('_')
    
    if len(parts) >= 4:
        try:
            frame_idx = int(parts[0])
            timestamp_str1 = parts[1]  # First timestamp string
            timestamp_str2 = parts[2]  # Second timestamp string
            camera_id = parts[3]
            return frame_idx, timestamp_str1, timestamp_str2, camera_id
        except (ValueError, IndexError):
            pass
    
    # Fallback for different formats
    if len(parts) >= 2:
        try:
            frame_idx = int(parts[0])
            return frame_idx, '', '', parts[-1] if len(parts) > 1 else ''
        except ValueError:
            pass
    
    return -1, '', '', ''


def ensure_4x4_matrix(T: np.ndarray) -> np.ndarray:
    """
    Ensure transformation matrix is 4x4 (convert 3x4 if needed).
    
    Args:
        T: 3x4 or 4x4 transformation matrix
    
    Returns:
        4x4 transformation matrix
    """
    T = np.asarray(T)
    if T.shape == (3, 4):
        T_4x4 = np.eye(4)
        T_4x4[:3, :] = T
        return T_4x4
    elif T.shape == (4, 4):
        return T
    else:
        raise ValueError(f"Invalid transform shape: {T.shape}")


def extract_pose_from_transform(
    T: np.ndarray,
    image_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract position and orientation from a 3x4 or 4x4 transformation matrix.
    
    Args:
        T: 3x4 or 4x4 cam_from_world transformation matrix
        image_name: Image filename for context
    
    Returns:
        (position, rotation_matrix, quaternion, euler_angles)
    """
    # Ensure 4x4 matrix
    T = ensure_4x4_matrix(T)
    
    # T is cam_from_world, so:
    # R = T[:3, :3]  (rotation)
    # t = T[:3, 3]   (translation)
    # Camera position in world = -R^T @ t
    
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Camera position in world coordinates
    position = -R.T @ t
    
    # Rotation matrix (keep as-is for cam_from_world)
    rotation_matrix = R
    
    # Convert to quaternion and euler using scipy
    try:
        rot = Rotation.from_matrix(R)
        quaternion = rot.as_quat()  # [x, y, z, w] format from scipy
        # Convert to [w, x, y, z] format (more common convention)
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        euler_angles = rot.as_euler('xyz', degrees=True)
    except Exception:
        quaternion = np.array([1, 0, 0, 0])
        euler_angles = np.array([0, 0, 0])
    
    return position, rotation_matrix, quaternion, euler_angles


def extract_trajectory_from_reconstruction(
    recon_path: Path,
    pycolmap_recon=None
) -> Dict[str, CameraTrajectory]:
    """
    Extract camera trajectories from a COLMAP/GLOMAP reconstruction.
    
    Args:
        recon_path: Path to reconstruction folder
        pycolmap_recon: Optional pre-loaded pycolmap.Reconstruction
    
    Returns:
        Dict mapping camera_id to CameraTrajectory
    """
    import pycolmap
    
    recon_path = Path(recon_path)
    
    # Handle path variations
    if not (recon_path / 'cameras.bin').exists():
        if (recon_path / '0').exists():
            recon_path = recon_path / '0'
        elif (recon_path / 'sparse' / '0').exists():
            recon_path = recon_path / 'sparse' / '0'
    
    # Load reconstruction
    if pycolmap_recon is None:
        recon = pycolmap.Reconstruction(str(recon_path))
    else:
        recon = pycolmap_recon
    
    # Group poses by camera ID
    trajectories: Dict[str, List[CameraPose]] = {}
    
    num_reg_images = recon.num_reg_images()
    for idx, (img_id, image) in enumerate(recon.images.items()):
        # Parse filename
        frame_idx, ts_str1, ts_str2, camera_id = parse_timestamp_from_filename(image.name)
        
        if frame_idx < 0:
            frame_idx = idx
        
        if not camera_id:
            camera_id = f"cam_{image.camera_id}"
        
        # Get transformation matrix
        if callable(image.cam_from_world):
            rigid = image.cam_from_world()
            T = rigid.matrix() if callable(rigid.matrix) else rigid.matrix
        elif hasattr(image.cam_from_world, 'matrix'):
            T = image.cam_from_world.matrix() if callable(image.cam_from_world.matrix) else image.cam_from_world.matrix
        else:
            T = np.array(image.cam_from_world)
        
        # Ensure 4x4 matrix
        T = ensure_4x4_matrix(T)
        
        # Extract pose components
        position, rotation_matrix, quaternion, euler_angles = extract_pose_from_transform(T, image.name)
        
        # Create pose object
        pose = CameraPose(
            frame_idx=frame_idx,
            timestamp_str1=ts_str1,
            timestamp_str2=ts_str2,
            image_name=image.name,
            camera_id=camera_id,
            position=position,
            rotation_matrix=rotation_matrix,
            quaternion=quaternion,
            euler_angles=euler_angles,
            cam_from_world=T,
        )
        
        if camera_id not in trajectories:
            trajectories[camera_id] = []
        trajectories[camera_id].append(pose)
        
        if idx + 1 >= num_reg_images:
            break
    
    # Sort poses by frame index and compute trajectory statistics
    result = {}
    for camera_id, poses in trajectories.items():
        # Sort by frame index (which corresponds to temporal order)
        poses.sort(key=lambda p: p.frame_idx)
        
        # Compute path length
        path_length = 0.0
        if len(poses) > 1:
            for i in range(1, len(poses)):
                dist = np.linalg.norm(poses[i].position - poses[i-1].position)
                path_length += dist
        
        traj = CameraTrajectory(
            camera_id=camera_id,
            poses=poses,
            path_length=path_length,
            num_poses=len(poses),
        )
        result[camera_id] = traj
    
    return result


def trajectories_to_arrays(
    trajectory: CameraTrajectory
) -> Dict[str, np.ndarray]:
    """
    Convert a CameraTrajectory to numpy arrays for easy processing.
    
    Args:
        trajectory: CameraTrajectory object
    
    Returns:
        Dict with:
            'frame_indices': (N,) array of frame indices
            'positions': (N, 3) array of positions
            'quaternions': (N, 4) array of quaternions [w, x, y, z]
            'euler_angles': (N, 3) array of euler angles [roll, pitch, yaw]
            'timestamp_str1': list of first timestamp strings
            'timestamp_str2': list of second timestamp strings
    """
    n = len(trajectory.poses)
    
    frame_indices = np.zeros(n, dtype=int)
    positions = np.zeros((n, 3))
    quaternions = np.zeros((n, 4))
    euler_angles = np.zeros((n, 3))
    timestamp_str1 = []
    timestamp_str2 = []
    
    for i, pose in enumerate(trajectory.poses):
        frame_indices[i] = pose.frame_idx
        positions[i] = pose.position
        quaternions[i] = pose.quaternion
        euler_angles[i] = pose.euler_angles
        timestamp_str1.append(pose.timestamp_str1)
        timestamp_str2.append(pose.timestamp_str2)
    
    return {
        'frame_indices': frame_indices,
        'positions': positions,
        'quaternions': quaternions,
        'euler_angles': euler_angles,
        'timestamp_str1': timestamp_str1,
        'timestamp_str2': timestamp_str2,
    }


def compute_trajectory_stats(trajectory: CameraTrajectory) -> Dict:
    """
    Compute detailed statistics for a camera trajectory.
    
    Args:
        trajectory: CameraTrajectory object
    
    Returns:
        Dict with trajectory statistics
    """
    if not trajectory.poses:
        return {'error': 'No poses in trajectory'}
    
    arrays = trajectories_to_arrays(trajectory)
    positions = arrays['positions']
    frame_indices = arrays['frame_indices']
    
    # Position statistics
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_range = pos_max - pos_min
    pos_mean = positions.mean(axis=0)
    
    # Compute step distances
    step_distances = []
    if len(positions) > 1:
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            step_distances.append(dist)
    
    avg_step = np.mean(step_distances) if step_distances else 0
    max_step = np.max(step_distances) if step_distances else 0
    
    return {
        'camera_id': trajectory.camera_id,
        'num_poses': trajectory.num_poses,
        'first_frame': int(frame_indices[0]) if len(frame_indices) > 0 else -1,
        'last_frame': int(frame_indices[-1]) if len(frame_indices) > 0 else -1,
        'path_length': trajectory.path_length,
        'position_min': pos_min.tolist(),
        'position_max': pos_max.tolist(),
        'position_range': pos_range.tolist(),
        'position_mean': pos_mean.tolist(),
        'avg_step_distance': avg_step,
        'max_step_distance': max_step,
    }


# =============================================================================
# 2D VIEW DATA PREPARATION
# =============================================================================
def prepare_trajectory_2d_views(trajectory: CameraTrajectory) -> Dict:
    """
    Prepare trajectory data for 2D visualization (front view + top view).
    
    Front view: X-Z plane (looking from Y direction)
    Top view: X-Y plane (looking from Z direction)
    
    Args:
        trajectory: CameraTrajectory object
    
    Returns:
        Dict with 2D coordinates for visualization
    """
    arrays = trajectories_to_arrays(trajectory)
    positions = arrays['positions']
    
    return {
        'camera_id': trajectory.camera_id,
        'num_points': len(positions),
        'front_view': {  # X-Z plane
            'x': positions[:, 0].tolist(),
            'y': positions[:, 2].tolist(),  # Z becomes Y in 2D
            'axis_labels': ['X', 'Z'],
        },
        'top_view': {  # X-Y plane
            'x': positions[:, 0].tolist(),
            'y': positions[:, 1].tolist(),
            'axis_labels': ['X', 'Y'],
        },
        'side_view': {  # Y-Z plane
            'x': positions[:, 1].tolist(),
            'y': positions[:, 2].tolist(),
            'axis_labels': ['Y', 'Z'],
        },
    }


# =============================================================================
# RIG RELATIVE POSE ANALYSIS (Multi-camera)
# =============================================================================
def compute_relative_pose(T1: np.ndarray, T2: np.ndarray) -> Dict:
    """
    Compute relative pose from camera 1 to camera 2.
    
    Args:
        T1: 4x4 cam1_from_world transform
        T2: 4x4 cam2_from_world transform
    
    Returns:
        Dict with relative translation and rotation
    """
    # T_rel = T2 @ T1^(-1)  -> cam2_from_cam1
    T1_inv = np.linalg.inv(T1)
    T_rel = T2 @ T1_inv
    
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    
    # Convert rotation to angle-axis representation
    try:
        rot = Rotation.from_matrix(R_rel)
        angle = np.linalg.norm(rot.as_rotvec())  # rotation angle in radians
        euler = rot.as_euler('xyz', degrees=True)
        quat = rot.as_quat()  # [x, y, z, w]
        quat = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
    except:
        angle = 0
        euler = [0, 0, 0]
        quat = [1, 0, 0, 0]
    
    return {
        'translation': t_rel.tolist(),
        'translation_norm': float(np.linalg.norm(t_rel)),
        'rotation_angle_deg': float(np.degrees(angle)),
        'euler_angles_deg': euler if isinstance(euler, list) else euler.tolist(),
        'quaternion': quat if isinstance(quat, list) else quat,
    }


def analyze_rig_relative_poses(trajectories: Dict[str, CameraTrajectory]) -> Dict:
    """
    Analyze relative poses between cameras in a rig (multi-camera setup).
    
    For each pair of cameras, compute relative pose at each synchronized frame.
    
    Args:
        trajectories: Dict mapping camera_id to CameraTrajectory
    
    Returns:
        Dict with relative pose analysis
    """
    cam_ids = sorted(trajectories.keys())
    
    if len(cam_ids) < 2:
        return {'is_rig': False, 'num_cameras': len(cam_ids)}
    
    # Build frame-indexed poses for each camera
    cam_poses_by_frame: Dict[str, Dict[int, CameraPose]] = {}
    all_frames = set()
    
    for cam_id, traj in trajectories.items():
        cam_poses_by_frame[cam_id] = {p.frame_idx: p for p in traj.poses}
        all_frames.update(p.frame_idx for p in traj.poses)
    
    # Find common frames (synchronized)
    common_frames = sorted(all_frames)
    for cam_id in cam_ids:
        common_frames = [f for f in common_frames if f in cam_poses_by_frame[cam_id]]
    
    # Compute relative poses for each camera pair at each common frame
    pair_analysis = {}
    
    for i, cam1 in enumerate(cam_ids):
        for cam2 in cam_ids[i+1:]:
            pair_key = f"{cam1}_to_{cam2}"
            
            rel_translations = []
            rel_angles = []
            frame_data = []
            
            for frame_idx in common_frames:
                pose1 = cam_poses_by_frame[cam1][frame_idx]
                pose2 = cam_poses_by_frame[cam2][frame_idx]
                
                rel = compute_relative_pose(pose1.cam_from_world, pose2.cam_from_world)
                rel_translations.append(rel['translation_norm'])
                rel_angles.append(rel['rotation_angle_deg'])
                
                frame_data.append({
                    'frame_idx': frame_idx,
                    'timestamp_str1': pose1.timestamp_str1,
                    'timestamp_str2': pose1.timestamp_str2,
                    **rel
                })
            
            # Statistics of relative pose changes
            trans_arr = np.array(rel_translations)
            angle_arr = np.array(rel_angles)
            
            pair_analysis[pair_key] = {
                'camera_pair': [cam1, cam2],
                'num_common_frames': len(common_frames),
                'translation_stats': {
                    'mean': float(trans_arr.mean()) if len(trans_arr) > 0 else 0,
                    'std': float(trans_arr.std()) if len(trans_arr) > 0 else 0,
                    'min': float(trans_arr.min()) if len(trans_arr) > 0 else 0,
                    'max': float(trans_arr.max()) if len(trans_arr) > 0 else 0,
                },
                'rotation_stats': {
                    'mean': float(angle_arr.mean()) if len(angle_arr) > 0 else 0,
                    'std': float(angle_arr.std()) if len(angle_arr) > 0 else 0,
                    'min': float(angle_arr.min()) if len(angle_arr) > 0 else 0,
                    'max': float(angle_arr.max()) if len(angle_arr) > 0 else 0,
                },
                'per_frame': frame_data,
            }
    
    return {
        'is_rig': True,
        'num_cameras': len(cam_ids),
        'camera_ids': cam_ids,
        'num_common_frames': len(common_frames),
        'common_frames': common_frames,
        'pair_analysis': pair_analysis,
    }


# =============================================================================
# FULL TRAJECTORY ANALYSIS (Single reconstruction)
# =============================================================================
def analyze_single_reconstruction(recon_path: Path) -> Dict:
    """
    Full analysis of a single reconstruction's trajectories.
    
    Args:
        recon_path: Path to reconstruction folder
    
    Returns:
        Dict with full analysis including trajectories, 2D views, and rig analysis
    """
    recon_path = Path(recon_path)
    
    # Extract trajectories
    trajectories = extract_trajectory_from_reconstruction(recon_path)
    
    num_cameras = len(trajectories)
    is_rig = num_cameras > 1
    
    # Per-camera analysis
    cameras_data = {}
    for cam_id, traj in trajectories.items():
        stats = compute_trajectory_stats(traj)
        views_2d = prepare_trajectory_2d_views(traj)
        
        cameras_data[cam_id] = {
            'stats': stats,
            'views_2d': views_2d,
            'trajectory': traj.to_dict(),
        }
    
    # Rig analysis (if multi-camera)
    rig_analysis = None
    if is_rig:
        rig_analysis = analyze_rig_relative_poses(trajectories)
    
    return {
        'path': str(recon_path),
        'num_cameras': num_cameras,
        'is_rig': is_rig,
        'cameras': cameras_data,
        'rig_analysis': rig_analysis,
    }


# =============================================================================
# HIERARCHICAL ANALYSIS (Folder -> Group -> Reconstruction)
# =============================================================================
def analyze_folder_hierarchical(folder_path: Path) -> Dict:
    """
    Hierarchical analysis of a mapping results folder.
    
    Structure:
        folder_root/
        ├── group_000/
        │   ├── cam0/ or bubble_00/
        │   │   └── sparse/0, 1, ...
        │   └── ...
        └── ...
    
    Args:
        folder_path: Root folder path (e.g., mapping3r_individual_xxx)
    
    Returns:
        Hierarchical dict with analysis at each level
    """
    folder_path = Path(folder_path)
    
    result = {
        'folder_name': folder_path.name,
        'folder_path': str(folder_path),
        'groups': {},
        'summary': {
            'total_groups': 0,
            'total_reconstructions': 0,
            'total_cameras': 0,
        }
    }
    
    # Iterate through groups
    for group_dir in sorted(folder_path.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.startswith('group_'):
            continue
        
        group_name = group_dir.name
        group_data = {
            'group_name': group_name,
            'reconstructions': {},
            'summary': {
                'total_reconstructions': 0,
                'total_cameras': 0,
            }
        }
        
        # Iterate through cam/bubble folders
        for recon_dir in sorted(group_dir.iterdir()):
            if not recon_dir.is_dir():
                continue
            
            # Check for sparse folder
            sparse_dir = recon_dir / 'sparse'
            if not sparse_dir.exists():
                continue
            
            recon_name = recon_dir.name  # e.g., 'cam0' or 'bubble_00'
            
            # Find best model in sparse (prefer model 0, or the one with most files)
            best_model = None
            for model_dir in sorted(sparse_dir.iterdir()):
                if model_dir.is_dir() and (model_dir / 'cameras.bin').exists():
                    best_model = model_dir
                    break  # Take first valid model (usually '0')
            
            if best_model is None:
                continue
            
            # Analyze this reconstruction
            try:
                analysis = analyze_single_reconstruction(best_model)
                group_data['reconstructions'][recon_name] = analysis
                group_data['summary']['total_reconstructions'] += 1
                group_data['summary']['total_cameras'] += analysis['num_cameras']
            except Exception as e:
                group_data['reconstructions'][recon_name] = {'error': str(e)}
        
        if group_data['reconstructions']:
            result['groups'][group_name] = group_data
            result['summary']['total_groups'] += 1
            result['summary']['total_reconstructions'] += group_data['summary']['total_reconstructions']
            result['summary']['total_cameras'] += group_data['summary']['total_cameras']
    
    return result


def save_analysis_json(analysis: Dict, output_path: Path) -> Path:
    """Save analysis to JSON file."""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to: {output_path}")
    return output_path


# =============================================================================
# Quick usage example
# =============================================================================
if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python kern_trajectory_analyze.py <path> [--hierarchical]")
        print("\nExtracts camera trajectories from a reconstruction or folder.")
        print("\nExamples:")
        print("  python kern_trajectory_analyze.py <sparse/0>              # Single reconstruction")
        print("  python kern_trajectory_analyze.py <mapping3r_xxx> --hierarchical  # Full folder")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    hierarchical = '--hierarchical' in sys.argv or '-h' in sys.argv[2:]
    
    if hierarchical:
        # Full hierarchical analysis
        print(f"Analyzing folder hierarchically: {path}")
        analysis = analyze_folder_hierarchical(path)
        
        print(f"\n=== Summary ===")
        print(f"Folder: {analysis['folder_name']}")
        print(f"Groups: {analysis['summary']['total_groups']}")
        print(f"Reconstructions: {analysis['summary']['total_reconstructions']}")
        print(f"Total cameras: {analysis['summary']['total_cameras']}")
        
        for group_name, group_data in analysis['groups'].items():
            print(f"\n  {group_name}:")
            for recon_name, recon_data in group_data['reconstructions'].items():
                if 'error' in recon_data:
                    print(f"    {recon_name}: ERROR - {recon_data['error']}")
                else:
                    is_rig = recon_data['is_rig']
                    n_cams = recon_data['num_cameras']
                    cam_ids = list(recon_data['cameras'].keys())
                    print(f"    {recon_name}: {n_cams} cam(s) {cam_ids} {'[RIG]' if is_rig else ''}")
                    
                    if is_rig and recon_data['rig_analysis']:
                        rig = recon_data['rig_analysis']
                        for pair_key, pair_data in rig['pair_analysis'].items():
                            ts = pair_data['translation_stats']
                            rs = pair_data['rotation_stats']
                            print(f"      {pair_key}: T={ts['mean']:.3f}m (std={ts['std']:.4f}), R={rs['mean']:.2f}deg (std={rs['std']:.3f})")
        
        # Save JSON
        output_path = path / 'trajectory_analysis.json'
        save_analysis_json(analysis, output_path)
    
    else:
        # Single reconstruction analysis
        print(f"Loading reconstruction: {path}")
        
        trajectories = extract_trajectory_from_reconstruction(path)
        
        print(f"\nFound {len(trajectories)} camera trajectories:")
        for cam_id, traj in trajectories.items():
            stats = compute_trajectory_stats(traj)
            print(f"\n  {cam_id}:")
            print(f"    Poses: {stats['num_poses']}")
            print(f"    Frames: {stats['first_frame']} -> {stats['last_frame']}")
            print(f"    Path length: {stats['path_length']:.2f} m")
            print(f"    Avg step: {stats['avg_step_distance']:.3f} m")
            
            if traj.poses:
                p0 = traj.poses[0]
                p1 = traj.poses[-1]
                print(f"    First: frame={p0.frame_idx}, ts1={p0.timestamp_str1}, ts2={p0.timestamp_str2}")
                print(f"    Last:  frame={p1.frame_idx}, ts1={p1.timestamp_str1}, ts2={p1.timestamp_str2}")
        
        # Rig analysis
        if len(trajectories) > 1:
            print(f"\n=== Rig Analysis ===")
            rig = analyze_rig_relative_poses(trajectories)
            print(f"Common frames: {rig['num_common_frames']}")
            for pair_key, pair_data in rig['pair_analysis'].items():
                ts = pair_data['translation_stats']
                rs = pair_data['rotation_stats']
                print(f"  {pair_key}:")
                print(f"    Translation: mean={ts['mean']:.4f}m, std={ts['std']:.6f}m")
                print(f"    Rotation: mean={rs['mean']:.3f}deg, std={rs['std']:.5f}deg")
