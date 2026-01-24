"""
ESSN_ReconstructionAnalyze - Core Functions for Reconstruction Analysis
=======================================================================
Essential/core functions for loading and analyzing COLMAP/GLOMAP reconstructions.

For full workflow with HTML/JSON reports, use:
    python _BURNSCRIPT_Result_Analyze/BURN_ReconstructionAnalyze.py <paths...>

Core Functions:
    - load_reconstruction_to_dict: Load a reconstruction into a dict
    - get_reconstruction_stats: Get statistics from a reconstruction
    - count_images_in_folder: Count image files in a folder
"""
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
import pycolmap


def count_images_in_folder(folder_path: Path) -> int:
    """
    Count image files in a folder.
    
    Args:
        folder_path: Path to folder containing images
    
    Returns:
        Number of image files found
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return sum(1 for f in folder_path.iterdir() 
               if f.is_file() and f.suffix.lower() in image_extensions)


def find_images_folder(recon_path: Path) -> Tuple[Path, int]:
    """
    Find the images folder relative to a reconstruction path.
    
    Structure is always:
        parent_folder/
        ├── sparse/
        │   ├── 0/   <- recon_path points here
        │   └── ...
        └── images/  <- we want this
    
    Args:
        recon_path: Path to reconstruction (e.g., .../sparse/0)
    
    Returns:
        (images_folder_path, image_count) or (None, 0) if not found
    """
    recon_path = Path(recon_path)
    images_folder = recon_path.parent.parent / 'images'
    
    if images_folder.exists() and images_folder.is_dir():
        count = count_images_in_folder(images_folder)
        return images_folder, count
    
    return None, 0


def load_reconstruction_to_dict(recon_path: Path) -> Dict:
    """
    Load a COLMAP/GLOMAP reconstruction into a dictionary.
    
    Args:
        recon_path: Path to reconstruction folder (containing cameras.bin, images.bin)
    
    Returns:
        Dict with reconstruction data:
            {
                'path': str,
                'images_folder': str,
                'num_images_total': int,
                'num_images_registered': int,
                'registration_ratio': float,
                'cameras': {camera_id: {...}},
                'images': {image_id: {...}},
                'points3D_count': int,
                'mean_reproj_error': float,
            }
    """
    recon_path = Path(recon_path)
    
    # Handle path variations
    if not (recon_path / 'cameras.bin').exists():
        if (recon_path / '0').exists() and (recon_path / '0' / 'cameras.bin').exists():
            recon_path = recon_path / '0'
        elif (recon_path / 'sparse' / '0').exists():
            recon_path = recon_path / 'sparse' / '0'
    
    recon = pycolmap.Reconstruction(str(recon_path))
    
    # Find actual images folder
    images_folder, total_images = find_images_folder(recon_path)
    if total_images == 0:
        total_images = len(recon.images)
    
    num_reg = recon.num_reg_images()
    
    # Extract cameras
    cameras = {}
    for cam_id, cam in recon.cameras.items():
        cameras[cam_id] = {
            'model': str(cam.model),
            'width': cam.width,
            'height': cam.height,
            'fx': cam.focal_length_x,
            'fy': cam.focal_length_y,
            'cx': cam.principal_point_x,
            'cy': cam.principal_point_y,
        }
    
    # Extract images with poses
    images = {}
    for idx, (img_id, image) in enumerate(recon.images.items()):
        # Get transform matrix
        if callable(image.cam_from_world):
            rigid = image.cam_from_world()
            T = rigid.matrix() if callable(rigid.matrix) else rigid.matrix
        elif hasattr(image.cam_from_world, 'matrix'):
            T = image.cam_from_world.matrix() if callable(image.cam_from_world.matrix) else image.cam_from_world.matrix
        else:
            T = image.cam_from_world
        
        # Camera position: -R^T @ t
        R, t = T[:3, :3], T[:3, 3]
        position = -R.T @ t
        
        images[img_id] = {
            'name': image.name,
            'camera_id': image.camera_id,
            'cam_from_world': T.tolist(),
            'position': position.tolist(),
        }
        
        if idx + 1 >= num_reg:
            break
    
    # Compute 3D points stats
    reproj_errors = []
    num_pts = recon.num_points3D()
    for idx, (_, pt) in enumerate(recon.points3D.items()):
        reproj_errors.append(pt.error)
        if idx + 1 >= num_pts:
            break
    
    return {
        'path': str(recon_path),
        'images_folder': str(images_folder) if images_folder else None,
        'num_images_total': total_images,
        'num_images_registered': num_reg,
        'registration_ratio': num_reg / max(1, total_images),
        'cameras': cameras,
        'images': images,
        'points3D_count': num_pts,
        'mean_reproj_error': np.mean(reproj_errors) if reproj_errors else 0,
    }


def get_reconstruction_stats(recon_path: Path) -> dict:
    """
    Get statistics from a COLMAP/GLOMAP reconstruction.
    
    Args:
        recon_path: Path to reconstruction folder
    
    Returns:
        Dict with reconstruction statistics
    """
    data = load_reconstruction_to_dict(recon_path)
    
    return {
        'num_images_total': data['num_images_total'],
        'num_images_registered': data['num_images_registered'],
        'registration_ratio': data['registration_ratio'],
        'num_cameras': len(data['cameras']),
        'num_points3D': data['points3D_count'],
        'mean_reproj_error': data['mean_reproj_error'],
    }


# =============================================================================
# Quick usage example
# =============================================================================
if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python ESSN_ReconstructionAnalyze.py <recon_path>")
        print("\nFor full analysis with HTML/JSON reports, use:")
        print("  python _BURNSCRIPT_Result_Analyze/BURN_ReconstructionAnalyze.py <paths...>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    print(f"Loading: {path}")
    
    data = load_reconstruction_to_dict(path)
    stats = get_reconstruction_stats(path)
    
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    print(f"\nImages loaded: {len(data['images'])}")
    print(f"Registration: {data['num_images_registered']}/{data['num_images_total']} "
          f"({data['registration_ratio']*100:.1f}%)")
