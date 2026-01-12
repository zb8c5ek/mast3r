#!/usr/bin/env python
"""
Starter Script - FINAL FULL DATASET Run for BatchedDB Generator
================================================================
Generated: 20260112_1330
Script: BatchedDB
Mode: FULL DATASET (all batches)

Processes: /d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam1/p-15_y-10_r+0
Mode: Full batched processing (batch_size=30, overlap=15, all images)
"""
import sys
from pathlib import Path
from time import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from SKRIPT_Batched_DB_Generator import process_images_in_batches, print_summary

if __name__ == "__main__":
    start_time = time()
    
    # Configuration - Full dataset
    dp_images = Path("/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam1/p-15_y-10_r+0")
    dp_output = dp_images.parent / "mapping3r_batched_20260112_1330_final"
    
    # Batch parameters
    BATCH_SIZE = 30
    OVERLAP = 15  # stride = 15
    MAPPER = None  # Skip mapping
    START_FRAME = 0
    SPACING = 1
    # Combined strategy: conf >= 2.5 AND top 1500 points
    MATCHING_STRATEGY = [('conf_thres', 2.5), ('num_pts', 1500)]
    
    print(f"=" * 80)
    print(f"STARTER FINAL - FULL DATASET PROCESSING")
    print(f"=" * 80)
    print(f"Images path: {dp_images}")
    print(f"Output path: {dp_output}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Overlap: {OVERLAP}")
    print(f"Matching: {MATCHING_STRATEGY}")
    print(f"=" * 80)
    print()
    
    # Load model
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Process all batches
    batch_results = process_images_in_batches(
        dp_images=dp_images,
        dp_output=dp_output,
        model=model,
        batch_size=BATCH_SIZE,
        overlap=OVERLAP,
        mapper=MAPPER,
        start_frame=START_FRAME,
        spacing=SPACING,
        matching_strategy=MATCHING_STRATEGY
    )
    
    # Print summary
    total_time = time() - start_time
    print_summary(batch_results, total_time)
