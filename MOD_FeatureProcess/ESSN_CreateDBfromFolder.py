"""ESSN_CreateDBfromFolder - Create COLMAP database from images using MASt3R."""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union
from time import time

import pycolmap
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import kapture_to_colmap

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import kapture_import_image_folder_or_list
from mast3r.image_pairs import make_pairs
import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r.utils.image import load_images
from .ESSN_FeatureProcess import run_mast3r_matching

MatchingStrategy = Union[Tuple[str, Union[float, int]], List[Tuple[str, Union[float, int]]]]

# COLMAP camera model IDs
CAMERA_MODEL_IDS = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
    'OPENCV_FISHEYE': 5,
    'FULL_OPENCV': 6,
}


def _update_camera_model(db, camera_model: str = 'PINHOLE'):
    """Update all cameras in DB to use specified camera model with appropriate params."""
    import numpy as np
    model_id = CAMERA_MODEL_IDS.get(camera_model, 1)  # default PINHOLE
    
    # Read existing cameras
    rows = db.execute("SELECT camera_id, width, height, params FROM cameras").fetchall()
    for camera_id, width, height, params_blob in rows:
        # Default focal estimate and principal point
        focal = 1.2 * max(width, height)
        cx, cy = width / 2.0, height / 2.0
        
        if camera_model == 'SIMPLE_PINHOLE':
            params = np.array([focal, cx, cy], dtype=np.float64)
        elif camera_model == 'PINHOLE':
            params = np.array([focal, focal, cx, cy], dtype=np.float64)
        elif camera_model == 'SIMPLE_RADIAL':
            params = np.array([focal, cx, cy, 0.0], dtype=np.float64)
        elif camera_model == 'RADIAL':
            params = np.array([focal, cx, cy, 0.0, 0.0], dtype=np.float64)
        elif camera_model == 'OPENCV':
            params = np.array([focal, focal, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            params = np.array([focal, focal, cx, cy], dtype=np.float64)  # fallback PINHOLE
        
        db.execute("UPDATE cameras SET model=?, params=? WHERE camera_id=?",
                   (model_id, params.tobytes(), camera_id))


def _run_matching_pipeline(
    root_path: str,
    filelist_relpath: List[str],
    dp_output: Path,
    model: AsymmetricMASt3R,
    matching_strategy: MatchingStrategy,
    image_size: int = 512,
    camera_model: str = 'PINHOLE',
) -> dict:
    """Core matching pipeline for DB creation."""
    result = {'success': False, 'database_path': None, 'num_images': len(filelist_relpath), 'num_pairs': 0, 'error': None}
    
    if len(filelist_relpath) < 2:
        result['error'] = f"Need at least 2 images, found {len(filelist_relpath)}"
        return result
    
    # Load and process
    filelist_abs = [str(Path(root_path) / rp) for rp in filelist_relpath]
    imgs = load_images(filelist_abs, size=image_size, square_ok=True, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True, sim_mat=None)
    
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), True)
    image_pairs = [(filelist_relpath[p1['idx']], filelist_relpath[p2['idx']]) for p1, p2 in pairs]
    
    # Setup DB
    cache_dir = dp_output / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    colmap_db_path = cache_dir / 'colmap.db'
    if colmap_db_path.exists():
        os.remove(colmap_db_path)
    
    colmap_db = COLMAPDatabase.connect(str(colmap_db_path))
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        
        # Update camera model to desired type (default PINHOLE)
        _update_camera_model(colmap_db, camera_model)
        
        colmap_image_pairs = run_mast3r_matching(
            model, image_size, 16, "cuda", kdata, root_path, image_pairs, colmap_db,
            dense_matching=False, pixel_tol=5, matching_strategy=matching_strategy,
            skip_geometric_verification=False, min_len_track=3
        )
    finally:
        colmap_db.close()
    
    if not colmap_image_pairs:
        result['error'] = "No matches were kept"
        return result
    
    result['num_pairs'] = len(colmap_image_pairs)
    
    # Write pairs and verify
    pairs_txt = dp_output / 'pairs.txt'
    with open(pairs_txt, 'w') as f:
        f.writelines(f"{p1} {p2}\n" for p1, p2 in colmap_image_pairs)
    
    pycolmap.verify_matches(str(colmap_db_path), str(pairs_txt))
    
    # Copy DB out
    dst_db = dp_output / 'database.db'
    shutil.copy2(colmap_db_path, dst_db)
    (dp_output / 'sparse').mkdir(exist_ok=True)
    
    result['success'] = True
    result['database_path'] = dst_db
    return result


def create_db_from_folder(
    dp_images: Path,
    dp_output: Path,
    model: AsymmetricMASt3R,
    matching_strategy: MatchingStrategy = ('num_pts', 1000),
    image_size: int = 512,
    camera_model: str = 'PINHOLE',
) -> dict:
    """Create COLMAP database from an image folder."""
    start = time()
    dp_output.mkdir(parents=True, exist_ok=True)
    
    # Collect and copy images
    fps = sorted(dp_images.glob("*.jpg")) + sorted(dp_images.glob("*.png"))
    images_dir = dp_output / 'images'
    images_dir.mkdir(exist_ok=True)
    
    filelist_relpath = []
    for fp in fps:
        dst = images_dir / fp.name
        if not dst.exists():
            shutil.copy2(fp, dst)
        filelist_relpath.append(f"images/{fp.name}")
    
    result = _run_matching_pipeline(str(dp_output), filelist_relpath, dp_output, model, matching_strategy, image_size, camera_model)
    result['processing_time'] = time() - start
    result['output_dir'] = dp_output
    
    if result['success']:
        print(f"✓ DB created: {result['num_images']} images, {result['num_pairs']} pairs, {result['processing_time']:.1f}s")
    else:
        print(f"✗ Failed: {result['error']}")
    return result


def create_db_from_folder_with_structure(
    root_path: Path,
    filelist_relpath: List[str],
    dp_output: Path,
    model: AsymmetricMASt3R,
    matching_strategy: MatchingStrategy = ('num_pts', 1000),
    image_size: int = 512,
    camera_model: str = 'PINHOLE',
) -> dict:
    """Create COLMAP database from pre-structured images (already copied)."""
    start = time()
    dp_output.mkdir(parents=True, exist_ok=True)
    
    result = _run_matching_pipeline(str(root_path), filelist_relpath, dp_output, model, matching_strategy, image_size, camera_model=camera_model)
    result['processing_time'] = time() - start
    result['output_dir'] = dp_output
    
    if result['success']:
        print(f"✓ DB created: {result['num_images']} images, {result['num_pairs']} pairs, {result['processing_time']:.1f}s")
    else:
        print(f"✗ Failed: {result['error']}")
    return result


def run_pycolmap_mapping(database_path: Path, image_path: Path, output_path: Path,
                         mapper_options: dict = None) -> dict:
    """Run pycolmap incremental mapping and return detailed results.
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to images root directory
        output_path: Output directory for sparse reconstruction
        mapper_options: Dict of mapper options (ba_refine_focal_length, min_num_matches, etc.)
    """
    result = {'success': False, 'num_registered': 0, 'num_points3d': 0, 'time': 0, 'error': None}
    start = time()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        opts = pycolmap.IncrementalPipelineOptions()
        
        # Apply options - these are on opts (pipeline level), not opts.mapper
        if mapper_options:
            if 'min_num_matches' in mapper_options:
                opts.min_num_matches = mapper_options['min_num_matches']
            if 'multiple_models' in mapper_options:
                opts.multiple_models = mapper_options['multiple_models']
            if 'extract_colors' in mapper_options:
                opts.extract_colors = mapper_options['extract_colors']
            if 'ba_refine_focal_length' in mapper_options:
                opts.ba_refine_focal_length = mapper_options['ba_refine_focal_length']
            if 'ba_refine_principal_point' in mapper_options:
                opts.ba_refine_principal_point = mapper_options['ba_refine_principal_point']
            if 'ba_refine_extra_params' in mapper_options:
                opts.ba_refine_extra_params = mapper_options['ba_refine_extra_params']
        
        pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(output_path),
            options=opts
        )
        result['time'] = time() - start
        
        # Check reconstruction and get stats
        recon_path = output_path / '0'
        if recon_path.exists():
            try:
                recon = pycolmap.Reconstruction(str(recon_path))
                result['num_registered'] = recon.num_reg_images()
                result['num_points3d'] = recon.num_points3D()
                result['success'] = True
            except:
                result['success'] = True  # Reconstruction exists but couldn't read stats
        return result
    except Exception as e:
        result['error'] = str(e)
        result['time'] = time() - start
        return result
