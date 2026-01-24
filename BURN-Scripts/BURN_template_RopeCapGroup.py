#!/usr/bin/env python
"""
BURN Template - RopeCap Group Processor (Individual Cameras)
=============================================================
Template script for processing grouped camera folders individually.
Each camera folder gets its own PINHOLE camera with shared intrinsics.

Reads configuration from a YAML file. The YAML file is the only thing that
changes between runs - this script remains unchanged.

Usage:
    python BURN-Scripts/BURN_template_RopeCapGroup.py --config configs/ropecap_group_20260123.yaml

Output structure:
    <output_base>/  (mapping3r_individual_YYYYMMDD_HHMMSS/)
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
    ├── report.html
    └── config_used.yaml
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime
from time import time
import json
import yaml
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from MOD_FeatureProcess.ESSN_CreateDBfromFolder import create_db_from_folder, run_pycolmap_mapping
from Utils4BurnScript import MappingJobManager


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


def process_single_folder(
        model,
        dp_images: Path,
        dp_output: Path,
        matching_strategy,
        image_size: int = 512,
        camera_model: str = 'PINHOLE',
        batch_size: int = 16
) -> dict:
    """
    Process a single image folder: create DB only.
    Mapping is handled separately by MappingJobManager.
    
    Args:
        batch_size: Batch size for MASt3R inference (default: 16).
    """
    result = {
        'input_path': str(dp_images),
        'output_path': str(dp_output),
        'db_success': False,
        'db_path': None,
        'db_time': 0,
        'num_images': 0,
        'num_pairs': 0,
        'error': None
    }
    
    try:
        db_result = create_db_from_folder(
            dp_images=dp_images,
            dp_output=dp_output,
            model=model,
            matching_strategy=matching_strategy,
            image_size=image_size,
            camera_model=camera_model,
            batch_size=batch_size
        )
        
        result['db_success'] = db_result['success']
        result['db_time'] = db_result.get('processing_time', 0)
        result['num_images'] = db_result.get('num_images', 0)
        result['num_pairs'] = db_result.get('num_pairs', 0)
        result['db_path'] = db_result.get('database_path')
        
        if not db_result['success']:
            result['error'] = db_result.get('error', 'Unknown DB error')
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def generate_html_report(all_results: dict, output_path: Path):
    """Generate HTML summary report."""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Individual Camera Processing Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #2196F3; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.success {{ color: green; }} .failed {{ color: red; }}
.summary {{ background: #e7f3fe; padding: 15px; border-radius: 5px; margin: 20px 0; }}
</style></head><body>
<h1>Individual Camera Processing Report</h1>
<div class="summary">
<p><b>Config:</b> {all_results['config_file']}</p>
<p><b>Model:</b> {all_results.get('model_name', 'N/A')}</p>
<p><b>Camera Model:</b> {all_results.get('camera_model', 'PINHOLE')}</p>
<p><b>Base Path:</b> {all_results['base_path']}</p>
<p><b>Total Time:</b> {all_results['total_time']:.1f}s ({all_results['total_time']/60:.1f} min)</p>
<p><b>Processed:</b> {all_results['total_processed']} | 
   <span class="success">Success: {all_results['total_success']}</span> | 
   <span class="failed">Failed: {all_results['total_failed']}</span></p>
</div>
<table>
<tr><th>Group</th><th>Camera</th><th>Images</th><th>Pairs</th>
<th>Registered</th><th>3D Points</th><th>DB Time</th><th>Map Time</th><th>Status</th></tr>
"""
    for group_name, cameras in all_results['groups'].items():
        for cam_name, r in cameras.items():
            status = '<span class="success">OK</span>' if r.get('mapping_success') else '<span class="failed">FAIL</span>'
            html += f"""<tr><td>{group_name}</td><td>{cam_name}</td>
<td>{r.get('num_images', 0)}</td>
<td>{r.get('num_pairs', 0)}</td><td>{r.get('num_registered', 0)}</td>
<td>{r.get('num_points3d', 0)}</td><td>{r.get('db_time', 0):.1f}s</td>
<td>{r.get('mapping_time', 0):.1f}s</td><td>{status}</td></tr>
"""
    html += "</table></body></html>"
    with open(output_path, 'w') as f:
        f.write(html)


def process_all_groups(config: dict, model) -> dict:
    """
    Process all groups based on configuration.
    Each camera folder is processed independently with shared PINHOLE intrinsics.
    """
    # Parse config
    paths_cfg = config.get('paths', {})
    processing_cfg = config.get('processing', {})
    match_cfg = config.get('matching', {})
    image_cfg = config.get('image', {})
    filters_cfg = config.get('filters', {})
    model_cfg = config.get('model', {})
    mapper_cfg = config.get('mapper', {}).copy() if config.get('mapper') else {}
    inference_cfg = config.get('inference', {})
    
    # Extract camera_model from mapper config (used during DB creation)
    camera_model = mapper_cfg.pop('camera_model', 'PINHOLE')
    
    # Inference settings (batch_size for MASt3R)
    batch_size = inference_cfg.get('batch_size', 16)
    
    base_path = Path(paths_cfg['base_path'])
    
    # Auto-generate output path with mapping3r_individual_ prefix
    if 'output' in paths_cfg and paths_cfg['output']:
        output_base = Path(paths_cfg['output'])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = base_path.parent / f"mapping3r_individual_{timestamp}"
    
    target_pose = processing_cfg.get('target_pose', 'p+0_y+0_r+0')
    run_mapping = processing_cfg.get('run_mapping', True)
    async_mapping = processing_cfg.get('async_mapping', True)  # Default: run mapping in background
    mapping_timeout = processing_cfg.get('mapping_timeout', 1800)  # Default: 30 minutes
    
    matching_strategy = parse_matching_strategy(match_cfg)
    image_size = image_cfg.get('size', 512)
    
    # Optional filters
    filter_groups = filters_cfg.get('groups', [])
    filter_cameras = filters_cfg.get('cameras', [])
    
    all_results = {
        'config_file': config.get('_config_file', 'unknown'),
        'model_name': model_cfg.get('name', 'unknown'),
        'camera_model': camera_model,
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
    print(f"BURN Template - RopeCap Individual Camera Processor")
    print(f"{'='*80}")
    print(f"Config: {config.get('_config_file', 'unknown')}")
    print(f"Base path: {base_path}")
    print(f"Output: {output_base}")
    print(f"Target pose: {target_pose}")
    print(f"Camera model: {camera_model} (shared intrinsics per folder)")
    print(f"Groups to process: {len(group_folders)}")
    print(f"Matching strategy: {matching_strategy}")
    print(f"Inference batch_size: {batch_size}")
    print(f"Run mapping: {run_mapping}")
    print(f"Async mapping: {async_mapping} (run mapping in background)")
    print(f"Mapping timeout: {mapping_timeout}s ({mapping_timeout/60:.0f} min)")
    print(f"Image size: {image_size}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create mapping job manager (for async mapping with live dashboard)
    job_mgr = MappingJobManager(dashboard_dir=output_base) if async_mapping else None
    
    # Backup config to output directory
    config_backup_path = output_base / 'config_used.yaml'
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config backed up to: {config_backup_path}\n")
    
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
            
            # Step 1: Create database (GPU-intensive)
            result = process_single_folder(
                model=model,
                dp_images=pose_folder,
                dp_output=dp_output,
                matching_strategy=matching_strategy,
                image_size=image_size,
                camera_model=camera_model,
                batch_size=batch_size
            )
            
            # Initialize result tracking
            result['mapping_success'] = False
            result['mapping_job_id'] = None
            result['mapping_time'] = 0
            result['num_registered'] = 0
            result['num_points3d'] = 0
            
            all_results['groups'][group_name][cam_name] = result
            all_results['total_processed'] += 1
            
            if not result['db_success']:
                all_results['total_failed'] += 1
                print(f"    DB FAIL: {result.get('error', 'Unknown')}")
                continue
            
            print(f"    DB OK: {result['num_images']} imgs, {result['num_pairs']} pairs ({result['db_time']:.1f}s)")
            
            # Step 2: Start mapping (async or sync)
            if run_mapping and result['db_path']:
                sparse_path = dp_output / 'sparse'
                job_name = f"{group_name}/{cam_name}"
                
                if async_mapping:
                    # Start mapping in background - GPU is free for next DB
                    job_id = job_mgr.start_job(
                        name=job_name,
                        database_path=result['db_path'],
                        image_path=dp_output,
                        output_path=sparse_path,
                        mapper_options=mapper_cfg if mapper_cfg else None
                    )
                    result['mapping_job_id'] = job_id
                else:
                    # Sync mapping (blocking)
                    map_result = run_pycolmap_mapping(
                        result['db_path'], dp_output, sparse_path,
                        mapper_options=mapper_cfg if mapper_cfg else None
                    )
                    result['mapping_success'] = map_result['success']
                    result['mapping_time'] = map_result['time']
                    result['num_registered'] = map_result['num_registered']
                    result['num_points3d'] = map_result['num_points3d']
                    
                    if map_result['success']:
                        all_results['total_success'] += 1
                        print(f"    Map OK: Reg {result['num_registered']}, Pts {result['num_points3d']} ({result['mapping_time']:.1f}s)")
                    else:
                        all_results['total_failed'] += 1
                        result['error'] = map_result.get('error', 'Mapping failed')
                        print(f"    Map FAIL: {result['error']}")
    
    # Wait for all async mapping jobs
    if async_mapping and run_mapping and job_mgr.jobs:
        mapping_results = job_mgr.wait_all(timeout=mapping_timeout)
        
        # Collect results back into all_results
        for group_name, group_data in all_results['groups'].items():
            for cam_name, result in group_data.items():
                job_id = result.get('mapping_job_id')
                if job_id and job_id in mapping_results:
                    map_result = mapping_results[job_id]
                    result['mapping_success'] = map_result.get('success', False)
                    result['mapping_time'] = map_result.get('elapsed', 0)
                    result['num_registered'] = map_result.get('num_registered', 0)
                    result['num_points3d'] = map_result.get('num_points3d', 0)
                    if not map_result.get('success'):
                        result['error'] = map_result.get('error', 'Mapping failed')
                    
                    if result['db_success'] and result['mapping_success']:
                        all_results['total_success'] += 1
                    else:
                        all_results['total_failed'] += 1
        
        job_mgr.print_summary()
    
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
    
    # Save results to JSON and HTML report
    results_path = output_base / 'processing_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    generate_html_report(all_results, output_base / 'report.html')
    print(f"Results: {results_path}")
    print(f"Report: {output_base / 'report.html'}")
    
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
