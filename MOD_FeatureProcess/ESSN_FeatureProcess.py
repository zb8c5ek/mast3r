"""
ESSN Feature Processing Module
High-level gin-configurable function for MASt3R feature matching.

This module provides the gin-configurable entry point that passes
parameters down to the core functions in kern_feature_process.py.

Gin configurable - use gin.parse_config_file() to load configuration from _gconfs/
"""
import logging
from typing import List, Tuple, Union

import gin
import kapture
import numpy as np
from tqdm import tqdm

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import scene_prepare_images, remove_duplicates
from mast3r.colmap.database import export_matches

from kapture.converter.colmap.database_extra import get_colmap_camera_ids_from_db, get_colmap_image_ids_from_db

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference

# Import core functions from kern
from .kern_feature_process import kern_get_im_matches, kern_apply_matching_strategy, MatchingStrategy

logger = logging.getLogger(__name__)


@gin.configurable
def essn_run_mast3r_matching(
        model: AsymmetricMASt3R,
        maxdim: int,
        patch_size: int,
        device,
        kdata: kapture.Kapture,
        root_path: str,
        image_pairs_kapture: List[Tuple[str, str]],
        colmap_db,
        # Gin configurable parameters with defaults
        dense_matching: bool = False,
        pixel_tol: int = 5,
        matching_strategy: MatchingStrategy = ('num_pts', 10000),
        skip_geometric_verification: bool = False,
        min_len_track: int = 3,
        chunk_size: int = 4,
        subsample: int = 8,
        viz: bool = False
):
    """
    Run MASt3R matching on image pairs (ESSN version - gin configurable).
    
    This is the high-level entry point that is gin configurable.
    It passes parameters down to kern_get_im_matches for the actual work.
    
    Args:
        model: The MASt3R model
        maxdim: Maximum image dimension
        patch_size: Patch size for feature extraction
        device: Device to run on (e.g., 'cuda')
        kdata: Kapture data structure
        root_path: Root path to images
        image_pairs_kapture: List of image pairs to match
        colmap_db: COLMAP database connection
        dense_matching: Whether to use dense matching (default: False)
        pixel_tol: Pixel tolerance for matching (default: 5)
        matching_strategy: Tuple specifying the filtering strategy:
            - ('conf_thres', float): Fixed confidence threshold
            - ('ratio', float): Keep top X% of matches (e.g., 0.1 for top 10%)
            - ('num_pts', int): Keep top N matches by confidence (default: 10000)
        skip_geometric_verification: Whether to skip geometric verification (default: False)
        min_len_track: Minimum track length (default: 3)
        chunk_size: Batch size for inference chunks (default: 4)
        subsample: Subsampling factor for sparse matching (default: 8)
        viz: Whether to visualize matches (default: False)
        
    Returns:
        colmap_image_pairs: List of matched image pairs
    """
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    # compute 2D-2D matching from dust3r inference
    all_thresholds = []
    all_num_matches = []
    
    # Format strategy display string
    if isinstance(matching_strategy, list):
        strategy_str = ' AND '.join([f"{s[0]}={s[1]}" for s in matching_strategy])
    else:
        strategy_type, strategy_value = matching_strategy
        strategy_str = f"{strategy_type}={strategy_value}"
    
    pbar = tqdm(range(0, len(matching_pairs), chunk_size), desc="Matching")
    for chunk in pbar:
        pairs_chunk = matching_pairs[chunk:chunk + chunk_size]
        output = inference(pairs_chunk, model, device, batch_size=4, verbose=False)
        pred1, pred2 = output['pred1'], output['pred2']
        
        # Call kern function with all parameters passed down
        im_images_chunk, match_stats = kern_get_im_matches(
            pred1, pred2, pairs_chunk, image_to_colmap, im_keypoints,
            matching_strategy=matching_strategy,
            is_sparse=not dense_matching,
            subsample=subsample,
            pixel_tol=pixel_tol,
            viz=viz,
            device=device
        )
        im_matches.update(im_images_chunk.items())
        
        # Collect stats and update tqdm
        all_thresholds.extend(match_stats['thresholds'])
        all_num_matches.extend(match_stats['num_matches'])
        
        if all_num_matches:
            avg_matches = np.mean(all_num_matches)
            
            # Format threshold display
            if all_thresholds and isinstance(all_thresholds[0], list):
                # Combined strategies - show average of each
                num_strategies = len(all_thresholds[0])
                avg_thrs = [np.mean([t[i] for t in all_thresholds]) for i in range(num_strategies)]
                thr_str = ','.join([f'{t:.2f}' for t in avg_thrs])
            else:
                # Single strategy
                avg_thr = np.mean(all_thresholds)
                thr_str = f'{avg_thr:.3f}'
            
            pbar.set_postfix({
                'thr': thr_str,
                'matches': f'{avg_matches:.0f}',
                'mode': strategy_str
            })
        
        # Log debug info for each pair in chunk
        for i, (thr, n_matches, n_cand) in enumerate(zip(
                match_stats['thresholds'], 
                match_stats['num_matches'], 
                match_stats['total_candidates'])):
            if isinstance(thr, list):
                thr_str = f"[{', '.join([f'{t:.4f}' for t in thr])}]"
            else:
                thr_str = f"{thr:.4f}"
            logger.debug(f"Pair {chunk+i}: threshold={thr_str}, matches={n_matches}/{n_cand}")

    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs


# Backward compatible alias
run_mast3r_matching = essn_run_mast3r_matching


if __name__ == "__main__":
    import os
    import shutil
    from pathlib import Path
    import pycolmap
    from mast3r.colmap.mapping import kapture_import_image_folder_or_list
    from mast3r.image_pairs import make_pairs
    from kapture.converter.colmap.database import COLMAPDatabase
    from kapture.converter.colmap.database_extra import kapture_to_colmap
    from dust3r.utils.image import load_images
    
    # ============ CONFIGURATION ============
    dp_images = Path("/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam1/p-15_y-10_r+0")
    dp_output = dp_images.parent / f"{dp_images.name}_essn_output"
    gin_config = Path(__file__).parent / '_gconfs' / 'essn_default.gin'
    model_path = Path("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").resolve()
    # =======================================
    
    # Load gin config
    gin.parse_config_file(str(gin_config))
    print(f"Loaded config: {gin_config}")
    
    # Get images
    dp_output.mkdir(parents=True, exist_ok=True)
    cache_dir = dp_output / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")))
    print(f"Found {len(image_files)} images in {dp_images}")
    
    # Limit to last 30 images
    if len(image_files) > 30:
        image_files = image_files[-30:]
        print(f"Subsampled to last {len(image_files)} images")
    
    # Load model
    model = AsymmetricMASt3R.from_pretrained(model_path).to('cuda')
    
    # Prepare data
    filelist = [str(fp) for fp in image_files]
    root_path = os.path.commonpath(filelist)
    filelist_relpath = [os.path.relpath(f, root_path).replace('\\', '/') for f in filelist]
    
    imgs = load_images(filelist, size=512, square_ok=True, verbose=True)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True, sim_mat=None)
    
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), use_single_camera=True)
    image_pairs_kapture = [(filelist_relpath[p[0]['idx']], filelist_relpath[p[1]['idx']]) for p in pairs]
    
    # Setup COLMAP DB in cache folder
    colmap_db_path = cache_dir / 'colmap.db'
    if colmap_db_path.exists():
        os.remove(colmap_db_path)
    
    colmap_db = COLMAPDatabase.connect(str(colmap_db_path))
    kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                      keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
    
    # Run matching
    colmap_image_pairs = essn_run_mast3r_matching(
        model, 512, 16, 'cuda', kdata, root_path, image_pairs_kapture, colmap_db
    )
    colmap_db.close()
    
    if len(colmap_image_pairs) == 0:
        raise Exception("No matches were kept")
    
    # Write pairs.txt and verify matches
    pairs_txt_path = cache_dir / 'pairs.txt'
    with open(pairs_txt_path, 'w') as f:
        for image_path1, image_path2 in colmap_image_pairs:
            f.write(f"{image_path1} {image_path2}\n")
    
    print("verify_matches")
    pycolmap.verify_matches(str(colmap_db_path), str(pairs_txt_path))
    
    # Copy colmap.db to output root
    shutil.copy(colmap_db_path, dp_output / 'colmap.db')
    
    # Create images folder and copy processed images
    images_dir = dp_output / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    for fp in image_files:
        shutil.copy(fp, images_dir / fp.name)
    
    print(f"\nDone! Matched {len(colmap_image_pairs)} pairs.")
    print(f"Output: {dp_output}")
    print(f"  - colmap.db")
    print(f"  - images/ ({len(image_files)} files)")
    print(f"  - cache/ (intermediate files)")

