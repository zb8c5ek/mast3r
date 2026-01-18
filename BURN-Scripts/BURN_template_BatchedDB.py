#!/usr/bin/env python
"""
BURN Template - Generic BatchedDB Processor
============================================
This is a template script that accepts command line arguments.
Use with a corresponding .sh file that provides the parameters.

Usage:
    python BURN_template_BatchedDB.py \
        --base_path /d_disk/... \
        --poses "p+0_y+15_r+0,p-20_y+15_r+0" \
        --batch_size 36 \
        --overlap 24 \
        --conf_thres 2.0 \
        --num_pts 1500 \
        --output_prefix "mapping3r_batched_20260114"
"""
import sys
import argparse
from pathlib import Path
from time import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from SKRIPT_Batched_DB_Generator import process_images_in_batches, print_summary


def sanitize_pose_name(pose_folder):
    """Sanitize pose name for safe filenames: + -> p, - -> m"""
    return pose_folder.replace('+', 'p').replace('-', 'm')


def process_single_pose(model, base_path, pose_folder, batch_size, overlap, matching_strategy, output_prefix):
    """Process a single camera pose directory"""
    dp_images = base_path / pose_folder
    
    pose_name = sanitize_pose_name(pose_folder)
    dp_output = dp_images.parent / f"{output_prefix}_{pose_name}"
    
    print(f"\n{'=' * 80}")
    print(f"PROCESSING POSE: {pose_folder}")
    print(f"{'=' * 80}")
    print(f"Images path: {dp_images}")
    print(f"Output path: {dp_output}")
    print(f"Batch size: {batch_size}")
    print(f"Overlap: {overlap} (stride: {batch_size - overlap})")
    print(f"Matching: {matching_strategy}")
    print(f"{'=' * 80}\n")
    
    batch_results = process_images_in_batches(
        dp_images=dp_images,
        dp_output=dp_output,
        model=model,
        batch_size=batch_size,
        overlap=overlap,
        mapper=None,
        start_frame=0,
        spacing=1,
        matching_strategy=matching_strategy
    )
    
    return batch_results, dp_output


def main():
    parser = argparse.ArgumentParser(description='BURN Template - BatchedDB Processor')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path to camera folder (e.g., /d_disk/.../cam0)')
    parser.add_argument('--poses', type=str, required=True,
                        help='Comma-separated pose folders (e.g., "p+0_y+15_r+0,p-20_y+15_r+0")')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size (default: 30)')
    parser.add_argument('--overlap', type=int, default=15,
                        help='Overlap between batches (default: 15)')
    parser.add_argument('--conf_thres', type=float, default=2.5,
                        help='Confidence threshold (default: 2.5)')
    parser.add_argument('--num_pts', type=int, default=1500,
                        help='Number of points to keep (default: 1500)')
    parser.add_argument('--output_prefix', type=str, default='mapping3r_batched',
                        help='Output folder prefix (default: mapping3r_batched)')
    
    args = parser.parse_args()
    
    start_time = time()
    
    # Parse arguments
    BASE_PATH = Path(args.base_path)
    POSE_FOLDERS = [p.strip() for p in args.poses.split(',')]
    BATCH_SIZE = args.batch_size
    OVERLAP = args.overlap
    MATCHING_STRATEGY = [('conf_thres', args.conf_thres), ('num_pts', args.num_pts)]
    OUTPUT_PREFIX = args.output_prefix
    
    print(f"{'=' * 80}")
    print(f"BURN TEMPLATE - BatchedDB Processor")
    print(f"{'=' * 80}")
    print(f"Base path: {BASE_PATH}")
    print(f"Poses to process: {POSE_FOLDERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Overlap: {OVERLAP} (stride: {BATCH_SIZE - OVERLAP})")
    print(f"Matching: {MATCHING_STRATEGY}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
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
            matching_strategy=MATCHING_STRATEGY,
            output_prefix=OUTPUT_PREFIX
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


if __name__ == "__main__":
    main()
