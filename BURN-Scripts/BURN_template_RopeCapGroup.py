#!/usr/bin/env python
"""
BURN Template - RopeCap Group Processor
========================================
Template script for processing grouped camera folders.

Reads configuration from a YAML file. The YAML file is the only thing that
changes between runs - this script remains unchanged.

Usage:
    python BURN-Scripts/BURN_template_RopeCapGroup.py --config configs/ropecap_YYYYMMDD_HHMMSS.yaml

Output structure:
    <output_base>/
    ├── group_001/
    │   ├── cam0/
    │   │   ├── images/
    │   │   ├── database.db
    │   │   ├── pairs.txt
    │   │   └── sparse/0/
    │   ├── cam1/
    │   └── ...
    ├── group_002/
    │   └── ...
    ├── processing_results.json
    └── config_used.yaml
"""
import sys
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from time import time
import json
import yaml
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from MOD_FeatureProcess.ESSN_CreateDBfromFolder import create_db_from_folder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_matching_strategy(match_cfg: dict):
    """Parse matching strategy from config."""
    if 'combined' in match_cfg:
        # Combined strategies: [('conf_thres', 2.5), ('num_pts', 1500)]
        return [(item['type'], item['value']) for item in match_cfg['combined']]
    else:
        # Single strategy
        return (match_cfg.get('type', 'num_pts'), match_cfg.get('value', 1000))


def run_mapping_subprocess(database_path: str, image_path: str, output_path: str) -> bool:
    """Run pycolmap mapping in a subprocess to isolate potential crashes."""
    try:
        import pycolmap
        Path(output_path).mkdir(parents=True, exist_ok=True)
        pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_path,
            output_path=output_path
        )
        return True
    except Exception as e:
        print(f"Mapping subprocess failed: {e}")
        return False


def process_single_folder(
        model,
        dp_images: Path,
        dp_output: Path,
        matching_strategy,
        image_size: int = 512,
        run_mapping: bool = True,
        mapping_timeout: int = 600
) -> dict:
    """
    Process a single image folder: create DB and optionally run mapping.
    """
    result = {
        'input_path': str(dp_images),
        'output_path': str(dp_output),
        'db_success': False,
        'mapping_success': False,
        'db_time': 0,
        'mapping_time': 0,
        'error': None
    }
    
    try:
        # Step 1: Create database
        db_result = create_db_from_folder(
            dp_images=dp_images,
            dp_output=dp_output,
            model=model,
            matching_strategy=matching_strategy,
            image_size=image_size
        )
        
        result['db_success'] = db_result['success']
        result['db_time'] = db_result['processing_time']
        result['num_images'] = db_result.get('num_images', 0)
        result['num_pairs'] = db_result.get('num_pairs', 0)
        
        if not db_result['success']:
            result['error'] = db_result.get('error', 'Unknown DB error')
            return result
        
        # Step 2: Run mapping (if requested)
        if run_mapping and db_result['database_path']:
            mapping_start = time()
            
            sparse_path = dp_output / 'sparse'
            
            # Run in subprocess to isolate crashes
            proc = multiprocessing.Process(
                target=run_mapping_subprocess,
                args=(
                    str(db_result['database_path']),
                    str(dp_output),
                    str(sparse_path)
                )
            )
            proc.start()
            proc.join(timeout=mapping_timeout)
            
            result['mapping_success'] = (proc.exitcode == 0)
            result['mapping_time'] = time() - mapping_start
            
            if not result['mapping_success']:
                result['error'] = f"Mapping failed (exit code: {proc.exitcode})"
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def process_all_groups(
        config: dict,
        model
) -> dict:
    """
    Process all groups based on configuration.
    """
    # Parse config
    paths_cfg = config.get('paths', {})
    processing_cfg = config.get('processing', {})
    match_cfg = config.get('matching', {})
    image_cfg = config.get('image', {})
    filters_cfg = config.get('filters', {})
    
    base_path = Path(paths_cfg['base_path'])
    
    # Auto-generate output path if not specified
    if 'output' in paths_cfg and paths_cfg['output']:
        output_base = Path(paths_cfg['output'])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = base_path.parent / f"mapping3r-{timestamp}"
    
    target_pose = processing_cfg.get('target_pose', 'p+0_y+0_r+0')
    run_mapping = processing_cfg.get('run_mapping', True)
    mapping_timeout = processing_cfg.get('mapping_timeout', 600)
    
    matching_strategy = parse_matching_strategy(match_cfg)
    image_size = image_cfg.get('size', 512)
    
    # Optional filters
    filter_groups = filters_cfg.get('groups', [])
    filter_cameras = filters_cfg.get('cameras', [])
    
    all_results = {
        'config_file': config.get('_config_file', 'unknown'),
        'base_path': str(base_path),
        'output_base': str(output_base),
        'target_pose': target_pose,
        'matching_strategy': str(matching_strategy),
        'groups': {},
        'total_processed': 0,
        'total_success': 0,
        'total_failed': 0,
        'total_time': 0
    }
    
    start_time = time()
    
    # Find all group folders
    group_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('group_')])
    
    # Apply group filter
    if filter_groups:
        group_folders = [g for g in group_folders if g.name in filter_groups]
    
    print(f"\n{'='*80}")
    print(f"BURN Template - RopeCap Group Processor")
    print(f"{'='*80}")
    print(f"Config: {config.get('_config_file', 'unknown')}")
    print(f"Base path: {base_path}")
    print(f"Output: {output_base}")
    print(f"Target pose: {target_pose}")
    print(f"Groups to process: {len(group_folders)}")
    print(f"Matching strategy: {matching_strategy}")
    print(f"Run mapping: {run_mapping}")
    print(f"Image size: {image_size}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Copy config to output directory
    config_backup_path = output_base / 'config_used.yaml'
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    for group_folder in group_folders:
        group_name = group_folder.name
        all_results['groups'][group_name] = {}
        
        print(f"\n{'='*60}")
        print(f"Processing {group_name}")
        print(f"{'='*60}")
        
        # Find all camera folders in this group
        cam_folders = sorted([d for d in group_folder.iterdir() if d.is_dir() and d.name.startswith('cam')])
        
        # Apply camera filter
        if filter_cameras:
            cam_folders = [c for c in cam_folders if c.name in filter_cameras]
        
        for cam_folder in cam_folders:
            cam_name = cam_folder.name
            
            # Check if target pose folder exists
            pose_folder = cam_folder / target_pose
            if not pose_folder.exists():
                print(f"  {cam_name}: Skipped (no {target_pose} folder)")
                continue
            
            # Check if there are images
            image_count = len(list(pose_folder.glob("*.jpg")) + list(pose_folder.glob("*.png")))
            if image_count < 2:
                print(f"  {cam_name}: Skipped (only {image_count} images)")
                continue
            
            print(f"\n  {cam_name}/{target_pose} ({image_count} images)")
            
            # Output path
            dp_output = output_base / group_name / cam_name
            
            # Process
            result = process_single_folder(
                model=model,
                dp_images=pose_folder,
                dp_output=dp_output,
                matching_strategy=matching_strategy,
                image_size=image_size,
                run_mapping=run_mapping,
                mapping_timeout=mapping_timeout
            )
            
            all_results['groups'][group_name][cam_name] = result
            all_results['total_processed'] += 1
            
            if result['db_success'] and (not run_mapping or result['mapping_success']):
                all_results['total_success'] += 1
                status = "✓ Success"
            else:
                all_results['total_failed'] += 1
                status = f"✗ Failed: {result.get('error', 'Unknown')}"
            
            print(f"    {status}")
            print(f"    DB: {result['db_time']:.1f}s, Mapping: {result['mapping_time']:.1f}s")
    
    all_results['total_time'] = time() - start_time
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total processed: {all_results['total_processed']}")
    print(f"Successful: {all_results['total_success']}")
    print(f"Failed: {all_results['total_failed']}")
    print(f"Total time: {all_results['total_time']:.1f}s ({all_results['total_time']/60:.1f} minutes)")
    print(f"{'='*80}\n")
    
    # Save results to JSON
    results_path = output_base / 'processing_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='BURN Template - RopeCap Group Processor')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to YAML config file (e.g., configs/ropecap_20260119_163000.yaml)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    config['_config_file'] = str(config_path)
    
    print(f"Loaded config from: {config_path}")
    
    # Load model
    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    checkpoint_dir = model_cfg.get('checkpoint_dir', 'checkpoints')
    
    weights_path = Path(checkpoint_dir) / (model_name + '.pth')
    weights_path = weights_path.resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Process all groups
    results = process_all_groups(config, model)
    
    return results


if __name__ == "__main__":
    main()
