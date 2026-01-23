#!/usr/bin/env python
"""
Starter Script - Two Camera Poses Processing for cam0 (p+0_y+15_r+0 and p-20_y+15_r+0)
=======================================================================================
Generated: 20260114
Script: BatchedDB
Mode: Multiple poses with identifiable outputs - cam0

Processes two camera pose directories:
1. p+0_y+15_r+0
2. p-20_y+15_r+0

Batch size: 36
Overlap: 24 (stride = 12)

Each output folder will include the pose information in its name.
"""
import sys
from pathlib import Path
from time import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from SKRIPT_Batched_DB_Generator import process_images_in_batches, print_summary

def process_single_pose(model, base_path, pose_folder, batch_size, overlap, matching_strategy):
    """Process a single camera pose directory"""
    dp_images = base_path / pose_folder
    
    # Sanitize pose name for filename (e.g., "p+0_y+15_r+0" -> "pp0_yp15_rp0")
    pose_name = pose_folder.replace('+', 'p').replace('-', 'm')
    dp_output = dp_images.parent / f"mapping3r_batched_20260114_{pose_name}"
    
    print(f"\n{'=' * 80}")
    print(f"PROCESSING POSE: {pose_name}")
    print(f"{'=' * 80}")
    print(f"Images path: {dp_images}")
    print(f"Output path: {dp_output}")
    print(f"Batch size: {batch_size}")
    print(f"Overlap: {overlap}")
    print(f"Matching: {matching_strategy}")
    print(f"{'=' * 80}\n")
    
    # Process all batches for this pose
    batch_results = process_images_in_batches(
        dp_images=dp_images,
        dp_output=dp_output,
        model=model,
        batch_size=batch_size,
        overlap=overlap,
        mapper=None,  # Skip mapping
        start_frame=0,
        spacing=1,
        matching_strategy=matching_strategy
    )
    
    return batch_results, dp_output

if __name__ == "__main__":
    start_time = time()
    
    # Configuration - cam0
    BASE_PATH = Path("/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam0")
    POSE_FOLDERS = [
        "p+0_y+15_r+0",
        "p-20_y+15_r+0"
    ]
    
    # Batch parameters
    BATCH_SIZE = 30
    OVERLAP = 15  # stride = 24
    # Combined strategy: conf >= 2.5 AND top 1500 points
    MATCHING_STRATEGY = [('conf_thres', 2.5), ('num_pts', 1500)]
    
    print(f"{'=' * 80}")
    print(f"STARTER - TWO POSES PROCESSING (cam0)")
    print(f"{'=' * 80}")
    print(f"Base path: {BASE_PATH}")
    print(f"Poses to process: {POSE_FOLDERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Overlap: {OVERLAP} (stride: {BATCH_SIZE - OVERLAP})")
    print(f"Matching: {MATCHING_STRATEGY}")
    print(f"{'=' * 80}\n")
    
    # Load model once
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Process each pose
    all_results = {}
    for pose_folder in POSE_FOLDERS:
        pose_start = time()
        
        batch_results, output_path = process_single_pose(
            model=model,
            base_path=BASE_PATH,
            pose_folder=pose_folder,
            batch_size=BATCH_SIZE,
            overlap=OVERLAP,
            matching_strategy=MATCHING_STRATEGY
        )
        
        pose_time = time() - pose_start
        all_results[pose_folder] = {
            'batch_results': batch_results,
            'output_path': output_path,
            'time': pose_time
        }
        
        print(f"\n{'=' * 80}")
        print(f"POSE {pose_folder} SUMMARY")
        print(f"{'=' * 80}")
        print_summary(batch_results, pose_time)
        print(f"Output saved to: {output_path}")
    
    # Final summary
    total_time = time() - start_time
    print(f"\n\n{'=' * 80}")
    print(f"FINAL SUMMARY - ALL POSES")
    print(f"{'=' * 80}")
    for pose_folder, results in all_results.items():
        print(f"\nPose: {pose_folder}")
        print(f"  Output: {results['output_path']}")
        print(f"  Batches: {len(results['batch_results'])}")
        print(f"  Time: {results['time']:.1f}s")
    
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 80}\n")
