#!/usr/bin/env python
"""
Starter Burn Script - Test run for BatchedDB Generator
=======================================================
Generated: 20260112_1330
Script: BatchedDB

Processes: /d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam1/p-15_y-10_r+0
Mode: Single batch test (batch_size=30, only 1 batch)
"""
import sys
from pathlib import Path
from time import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from SKRIPT_Batched_DB_Generator import get_reconstructed_scene

if __name__ == "__main__":
    start_time = time()
    
    # Configuration - Single batch test
    dp_images = Path("/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam1/p-15_y-10_r+0")
    dp_output = dp_images.parent / "burn_test_20260112_1330"
    dp_output.mkdir(parents=True, exist_ok=True)
    
    # Get images - only first 30 for single batch test
    fps_images = sorted(list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")))
    fps_images = fps_images[:30]  # Single batch of 30 images
    
    print(f"=" * 80)
    print(f"STARTER BURN TEST - 20260112_1330")
    print(f"=" * 80)
    print(f"Images path: {dp_images}")
    print(f"Output path: {dp_output}")
    print(f"Num images: {len(fps_images)} (single batch test)")
    print(f"=" * 80)
    print()
    
    # Load model
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Run single batch
    MATCHING_STRATEGY = ('num_pts', 1000)
    
    scene_state, outfile = get_reconstructed_scene(
        outdir=dp_output,
        model=model,
        filelist=[fp.resolve().as_posix() for fp in fps_images],
        mapper=None,  # Skip mapping for test
        colmap_output_dir=str(dp_output),
        matching_strategy=MATCHING_STRATEGY,
    )
    
    total_time = time() - start_time
    print(f"\n" + "=" * 80)
    print(f"BURN TEST COMPLETE")
    print(f"=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Output: {dp_output}")
    print(f"=" * 80)
