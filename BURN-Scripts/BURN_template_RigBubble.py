#!/usr/bin/env python
"""
BURN Template - RopeCap Rig-Bubble Processor
=============================================
Processes grouped camera folders with rig-bubbles (multiple cameras per bubble).

Usage:
    python BURN-Scripts/BURN_template_RigBubble.py --config configs/ropecap_rigbubble_20260121.yaml

Output structure:
    <output_base>/
    ├── group_000/
    │   ├── bubble_00/ (images/, cache/, database.db, sparse/)
    │   ├── bubble_01/
    │   ├── bubble_02/
    │   └── group_000_report.json
    ├── group_001/
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
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from mast3r.model import AsymmetricMASt3R
from MOD_FeatureProcess.ESSN_CreateDBfromFolder import create_db_from_folder_with_structure, run_pycolmap_mapping
from Utils4BurnScript import MappingJobManager


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_matching_strategy(match_cfg: dict):
    if 'combined' in match_cfg:
        return [(item['type'], item['value']) for item in match_cfg['combined']]
    return (match_cfg.get('type', 'num_pts'), match_cfg.get('value', 1000))


def collect_bubble_images(group_folder: Path, cameras: List[str], target_pose: str, stride_frame: int = 1) -> Dict[str, List[Path]]:
    """Collect images from multiple cameras for a bubble with stride."""
    images_by_cam = {}
    for cam_name in cameras:
        pose_folder = group_folder / cam_name / target_pose
        if not pose_folder.exists():
            continue
        images = sorted(list(pose_folder.glob("*.jpg")) + list(pose_folder.glob("*.png")))
        if stride_frame > 1:
            images = images[::stride_frame]
        if images:
            images_by_cam[cam_name] = images
    return images_by_cam


def prepare_bubble_images(images_by_cam: Dict[str, List[Path]], output_dir: Path) -> tuple:
    """Copy images to output directory with structure: images/camX/"""
    images_root = output_dir / 'images'
    images_root.mkdir(parents=True, exist_ok=True)
    all_rel_paths = []
    for cam_name, image_paths in images_by_cam.items():
        cam_dir = images_root / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for img_path in image_paths:
            dst_path = cam_dir / img_path.name
            if not dst_path.exists():
                shutil.copy2(img_path, dst_path)
            all_rel_paths.append(f"images/{cam_name}/{img_path.name}")
    return output_dir, all_rel_paths


def process_single_bubble(model, images_by_cam: Dict[str, List[Path]], dp_output: Path,
                          matching_strategy, image_size: int = 512,
                          camera_model: str = 'PINHOLE',
                          share_intrinsics_by_subfolder: bool = True,
                          batch_size: int = 16) -> dict:
    """Process a single bubble: create DB only. Mapping handled by MappingJobManager.
    
    Args:
        share_intrinsics_by_subfolder: If True, images in each camera subfolder (images/cam0/, images/cam1/)
                                       share the same camera intrinsics. Default True for multi-camera rigs.
        batch_size: Batch size for MASt3R inference (default: 16).
    """
    result = {
        'output_path': str(dp_output), 'cameras': list(images_by_cam.keys()),
        'db_success': False, 'db_path': None,
        'db_time': 0, 'num_images': 0, 'num_pairs': 0, 'error': None
    }
    
    total_images = sum(len(imgs) for imgs in images_by_cam.values())
    result['num_images'] = total_images
    
    if total_images < 2:
        result['error'] = f"Need at least 2 images, found {total_images}"
        return result
    
    try:
        # Prepare images (organized as images/cam0/, images/cam1/, etc.)
        root_path, filelist_relpath = prepare_bubble_images(images_by_cam, dp_output)
        
        # Create database with specified camera model
        db_result = create_db_from_folder_with_structure(
            root_path=root_path, filelist_relpath=filelist_relpath,
            dp_output=dp_output, model=model,
            matching_strategy=matching_strategy, image_size=image_size,
            camera_model=camera_model,
            share_intrinsics_by_subfolder=share_intrinsics_by_subfolder,
            batch_size=batch_size
        )
        result['db_success'] = db_result['success']
        result['db_time'] = db_result.get('processing_time', 0)
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
<html><head><meta charset="utf-8"><title>RigBubble Processing Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #4CAF50; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.success {{ color: green; }} .failed {{ color: red; }}
.summary {{ background: #e7f3fe; padding: 15px; border-radius: 5px; margin: 20px 0; }}
</style></head><body>
<h1>RigBubble Processing Report</h1>
<div class="summary">
<p><b>Config:</b> {all_results['config_file']}</p>
<p><b>Model:</b> {all_results.get('model_name', 'N/A')}</p>
<p><b>Base Path:</b> {all_results['base_path']}</p>
<p><b>Total Time:</b> {all_results['total_time']:.1f}s ({all_results['total_time']/60:.1f} min)</p>
<p><b>Processed:</b> {all_results['total_processed']} | 
   <span class="success">Success: {all_results['total_success']}</span> | 
   <span class="failed">Failed: {all_results['total_failed']}</span></p>
</div>
<table>
<tr><th>Group</th><th>Bubble</th><th>Cameras</th><th>Images</th><th>Pairs</th>
<th>Registered</th><th>3D Points</th><th>DB Time</th><th>Map Time</th><th>Status</th></tr>
"""
    for group_name, bubbles in all_results['groups'].items():
        for bubble_name, r in bubbles.items():
            status = '<span class="success">✓</span>' if r.get('mapping_success') else '<span class="failed">✗</span>'
            html += f"""<tr><td>{group_name}</td><td>{bubble_name}</td>
<td>{', '.join(r.get('cameras', []))}</td><td>{r.get('num_images', 0)}</td>
<td>{r.get('num_pairs', 0)}</td><td>{r.get('num_registered', 0)}</td>
<td>{r.get('num_points3d', 0)}</td><td>{r.get('db_time', 0):.1f}s</td>
<td>{r.get('mapping_time', 0):.1f}s</td><td>{status}</td></tr>
"""
    html += "</table></body></html>"
    with open(output_path, 'w') as f:
        f.write(html)


def process_all_groups(config: dict, model) -> dict:
    """Process all groups based on configuration."""
    paths_cfg = config.get('paths', {})
    processing_cfg = config.get('processing', {})
    match_cfg = config.get('matching', {})
    image_cfg = config.get('image', {})
    model_cfg = config.get('model', {})
    rig_bubbles_cfg = config.get('rig_bubbles', [])
    mapper_cfg = config.get('mapper', {}).copy() if config.get('mapper') else {}
    inference_cfg = config.get('inference', {})
    
    # Extract camera_model from mapper config (used during DB creation)
    camera_model = mapper_cfg.pop('camera_model', 'PINHOLE')
    # Share intrinsics by subfolder - each camera folder (cam0, cam1, etc.) shares one camera
    share_intrinsics = mapper_cfg.pop('share_intrinsics_by_subfolder', True)
    
    # Inference settings (batch_size for MASt3R)
    batch_size = inference_cfg.get('batch_size', 16)
    
    base_path = Path(paths_cfg['base_path'])
    output_base = Path(paths_cfg['output']) if paths_cfg.get('output') else \
                  base_path.parent / f"mapping3r_rigbubble-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    target_pose = processing_cfg.get('target_pose', 'p+0_y+0_r+0')
    stride_frame = processing_cfg.get('stride_frame', 1)
    run_mapping = processing_cfg.get('run_mapping', True)
    async_mapping = processing_cfg.get('async_mapping', True)  # Default: run mapping in background
    mapping_timeout = processing_cfg.get('mapping_timeout', 1800)  # Default: 30 minutes
    matching_strategy = parse_matching_strategy(match_cfg)
    image_size = image_cfg.get('size', 512)
    
    filter_groups = config.get('filters', {}).get('groups', [])
    
    all_results = {
        'config_file': config.get('_config_file', 'unknown'),
        'model_name': model_cfg.get('name', 'unknown'),
        'base_path': str(base_path), 'output_base': str(output_base),
        'target_pose': target_pose, 'stride_frame': stride_frame,
        'matching_strategy': str(matching_strategy),
        'groups': {}, 'total_processed': 0, 'total_success': 0, 'total_failed': 0, 'total_time': 0
    }
    
    start_time = time()
    group_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('group_')])
    if filter_groups:
        group_folders = [g for g in group_folders if g.name in filter_groups]
    
    print(f"\n{'='*80}\nBURN Template - RopeCap Rig-Bubble Processor\n{'='*80}")
    print(f"Base: {base_path}\nOutput: {output_base}\nGroups: {len(group_folders)}\nStride: {stride_frame}")
    print(f"Camera model: {camera_model}, Share intrinsics by subfolder: {share_intrinsics}")
    print(f"Inference batch_size: {batch_size}")
    print(f"Async mapping: {async_mapping}, Timeout: {mapping_timeout}s ({mapping_timeout/60:.0f} min)\n{'='*80}\n")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create mapping job manager (for async mapping with live dashboard)
    job_mgr = MappingJobManager(dashboard_dir=output_base) if async_mapping else None
    
    with open(output_base / 'config_used.yaml', 'w') as f:
        yaml.dump(config, f)
    
    for group_folder in group_folders:
        group_name = group_folder.name
        all_results['groups'][group_name] = {}
        group_results = {'group': group_name, 'bubbles': {}}
        
        print(f"\n{'='*60}\nProcessing {group_name}\n{'='*60}")
        
        for bubble_cfg in rig_bubbles_cfg:
            bubble_name, cameras = bubble_cfg['name'], bubble_cfg['cameras']
            print(f"\n  {bubble_name}: {cameras}")
            
            images_by_cam = collect_bubble_images(group_folder, cameras, target_pose, stride_frame)
            if not images_by_cam:
                print(f"    Skipped (no images)")
                continue
            
            total_imgs = sum(len(v) for v in images_by_cam.values())
            print(f"    Images: {total_imgs} ({', '.join(f'{k}:{len(v)}' for k,v in images_by_cam.items())})")
            
            dp_output = output_base / group_name / bubble_name
            
            # Step 1: Create database (GPU-intensive)
            result = process_single_bubble(
                model, images_by_cam, dp_output, matching_strategy, image_size,
                camera_model=camera_model,
                share_intrinsics_by_subfolder=share_intrinsics,
                batch_size=batch_size
            )
            
            # Initialize mapping result fields
            result['mapping_success'] = False
            result['mapping_job_id'] = None
            result['mapping_time'] = 0
            result['num_registered'] = 0
            result['num_points3d'] = 0
            
            all_results['groups'][group_name][bubble_name] = result
            group_results['bubbles'][bubble_name] = result
            all_results['total_processed'] += 1
            
            if not result['db_success']:
                all_results['total_failed'] += 1
                print(f"    DB FAIL: {result.get('error', 'Unknown')}")
                continue
            
            print(f"    DB OK: {result['num_images']} imgs, {result['num_pairs']} pairs ({result['db_time']:.1f}s)")
            
            # Step 2: Start mapping (async or sync)
            if run_mapping and result['db_path']:
                sparse_path = dp_output / 'sparse'
                job_name = f"{group_name}/{bubble_name}"
                
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
        
        # Save per-group JSON report
        group_report_path = output_base / group_name / f"{group_name}_report.json"
        group_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(group_report_path, 'w') as f:
            json.dump(group_results, f, indent=2)
    
    # Wait for all async mapping jobs
    if async_mapping and run_mapping and job_mgr.jobs:
        mapping_results = job_mgr.wait_all(timeout=mapping_timeout)
        
        # Collect results back into all_results
        for group_name, bubbles in all_results['groups'].items():
            for bubble_name, result in bubbles.items():
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
    
    print(f"\n\n{'='*80}\nSUMMARY: {all_results['total_processed']} processed, "
          f"{all_results['total_success']} success, {all_results['total_failed']} failed, "
          f"{all_results['total_time']:.1f}s total\n{'='*80}\n")
    
    # Save final JSON and HTML reports
    with open(output_base / 'processing_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    generate_html_report(all_results, output_base / 'report.html')
    print(f"Reports: {output_base / 'processing_results.json'}, {output_base / 'report.html'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='BURN Template - RopeCap Rig-Bubble Processor')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    config['_config_file'] = str(config_path)
    
    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    weights_path = (Path(model_cfg.get('checkpoint_dir', 'checkpoints')) / (model_name + '.pth')).resolve()
    
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded!\n")
    
    return process_all_groups(config, model)


if __name__ == "__main__":
    main()
