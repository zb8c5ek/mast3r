#!/usr/bin/env python
"""
BURN Template - RopeCap Multi-Pose Processor
=============================================
Single camera + multiple poses → single COLMAP DB.
Collects images from multiple pose folders (e.g., p+0_y+0_r+0, p+0_y-10_r+0).

Usage:
    python BURN-Scripts/BURN_template_RopeCapMultiPose.py --config configs/ropecap_multipose.yaml
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime
from time import time
import json
import yaml
import argparse

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


def collect_multipose_images(cam_folder: Path, target_poses: list, stride_frame: int = 1) -> dict:
    """Collect images from multiple pose folders for a single camera.
    
    Returns: {pose_name: [image_paths]} or flattened for single-intrinsics mode
    """
    images_by_pose = {}
    for pose in target_poses:
        pose_folder = cam_folder / pose
        if not pose_folder.exists():
            continue
        images = sorted(list(pose_folder.glob("*.jpg")) + list(pose_folder.glob("*.png")))
        if stride_frame > 1:
            images = images[::stride_frame]
        if images:
            images_by_pose[pose] = images
    return images_by_pose


def prepare_multipose_images(images_by_pose: dict, output_dir: Path, share_by_pose: bool = False) -> tuple:
    """Copy images to output directory.
    
    If share_by_pose=True: images/pose_name/img.jpg (each pose shares intrinsics)
    If share_by_pose=False: images/img.jpg (all share one camera)
    """
    images_root = output_dir / 'images'
    images_root.mkdir(parents=True, exist_ok=True)
    all_rel_paths = []
    
    for pose_name, image_paths in images_by_pose.items():
        if share_by_pose:
            pose_dir = images_root / pose_name
            pose_dir.mkdir(parents=True, exist_ok=True)
            for img_path in image_paths:
                dst = pose_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                all_rel_paths.append(f"images/{pose_name}/{img_path.name}")
        else:
            for img_path in image_paths:
                # Prefix with pose to avoid name collision
                dst_name = f"{pose_name}_{img_path.name}"
                dst = images_root / dst_name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                all_rel_paths.append(f"images/{dst_name}")
    
    return output_dir, all_rel_paths


def process_single_multipose(model, images_by_pose: dict, dp_output: Path,
                             matching_strategy, image_size: int = 512,
                             camera_model: str = 'PINHOLE',
                             share_intrinsics_by_pose: bool = False,
                             batch_size: int = 16) -> dict:
    """Process multiple poses from single camera: create DB only."""
    result = {
        'output_path': str(dp_output), 'poses': list(images_by_pose.keys()),
        'db_success': False, 'db_path': None,
        'db_time': 0, 'num_images': 0, 'num_pairs': 0, 'error': None
    }
    
    total_images = sum(len(imgs) for imgs in images_by_pose.values())
    result['num_images'] = total_images
    
    if total_images < 2:
        result['error'] = f"Need at least 2 images, found {total_images}"
        return result
    
    try:
        root_path, filelist_relpath = prepare_multipose_images(
            images_by_pose, dp_output, share_by_pose=share_intrinsics_by_pose
        )
        
        db_result = create_db_from_folder_with_structure(
            root_path=root_path, filelist_relpath=filelist_relpath,
            dp_output=dp_output, model=model,
            matching_strategy=matching_strategy, image_size=image_size,
            camera_model=camera_model,
            share_intrinsics_by_subfolder=share_intrinsics_by_pose,
            batch_size=batch_size
        )
        result['db_success'] = db_result['success']
        result['db_time'] = db_result.get('processing_time', 0)
        result['num_pairs'] = db_result.get('num_pairs', 0)
        # Convert Path to string for JSON serialization
        db_path = db_result.get('database_path')
        result['db_path'] = str(db_path) if db_path else None
        
        if not db_result['success']:
            result['error'] = db_result.get('error', 'Unknown DB error')
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def generate_html_report(all_results: dict, output_path: Path):
    """Generate HTML summary report."""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MultiPose Processing Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #00d4ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th {{ background: #0f3460; color: #00d4ff; padding: 10px; text-align: left; }}
td {{ padding: 8px; border-bottom: 1px solid #333; }}
.success {{ color: #00ff88; }} .failed {{ color: #ff4757; }}
.summary {{ background: #16213e; padding: 15px; border-radius: 8px; margin: 15px 0; }}
</style></head><body>
<h1>MultiPose Processing Report</h1>
<div class="summary">
<p><b>Config:</b> {all_results['config_file']}</p>
<p><b>Poses:</b> {', '.join(all_results.get('target_poses', []))}</p>
<p><b>Total Time:</b> {all_results['total_time']:.1f}s</p>
<p><b>Processed:</b> {all_results['total_processed']} | 
   <span class="success">Success: {all_results['total_success']}</span> | 
   <span class="failed">Failed: {all_results['total_failed']}</span></p>
</div>
<table>
<tr><th>Group</th><th>Camera</th><th>Poses</th><th>Images</th><th>Pairs</th>
<th>Registered</th><th>3D Points</th><th>DB Time</th><th>Map Time</th><th>Status</th></tr>
"""
    for group_name, cameras in all_results['groups'].items():
        for cam_name, r in cameras.items():
            status = '<span class="success">OK</span>' if r.get('mapping_success') else '<span class="failed">FAIL</span>'
            poses = ', '.join(r.get('poses', []))
            html += f"""<tr><td>{group_name}</td><td>{cam_name}</td><td>{poses}</td>
<td>{r.get('num_images', 0)}</td><td>{r.get('num_pairs', 0)}</td>
<td>{r.get('num_registered', 0)}</td><td>{r.get('num_points3d', 0)}</td>
<td>{r.get('db_time', 0):.1f}s</td><td>{r.get('mapping_time', 0):.1f}s</td><td>{status}</td></tr>
"""
    html += "</table></body></html>"
    with open(output_path, 'w') as f:
        f.write(html)


def process_all_groups(config: dict, model) -> dict:
    """Process all groups: single camera + multiple poses → single DB."""
    paths_cfg = config.get('paths', {})
    processing_cfg = config.get('processing', {})
    match_cfg = config.get('matching', {})
    image_cfg = config.get('image', {})
    model_cfg = config.get('model', {})
    mapper_cfg = config.get('mapper', {}).copy() if config.get('mapper') else {}
    inference_cfg = config.get('inference', {})
    
    camera_model = mapper_cfg.pop('camera_model', 'PINHOLE')
    share_intrinsics_by_pose = mapper_cfg.pop('share_intrinsics_by_pose', False)
    batch_size = inference_cfg.get('batch_size', 16)
    
    base_path = Path(paths_cfg['base_path'])
    output_base = Path(paths_cfg['output']) if paths_cfg.get('output') else \
                  base_path.parent / f"mapping3r_multipose-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Multiple target poses
    target_poses = processing_cfg.get('target_poses', ['p+0_y+0_r+0'])
    stride_frame = processing_cfg.get('stride_frame', 1)
    run_mapping = processing_cfg.get('run_mapping', True)
    async_mapping = processing_cfg.get('async_mapping', True)
    mapping_timeout = processing_cfg.get('mapping_timeout', 1800)
    matching_strategy = parse_matching_strategy(match_cfg)
    image_size = image_cfg.get('size', 512)
    
    filter_groups = config.get('filters', {}).get('groups', [])
    filter_cameras = config.get('filters', {}).get('cameras', [])
    
    all_results = {
        'config_file': config.get('_config_file', 'unknown'),
        'model_name': model_cfg.get('name', 'unknown'),
        'base_path': str(base_path), 'output_base': str(output_base),
        'target_poses': target_poses, 'stride_frame': stride_frame,
        'matching_strategy': str(matching_strategy),
        'groups': {}, 'total_processed': 0, 'total_success': 0, 'total_failed': 0, 'total_time': 0
    }
    
    start_time = time()
    group_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('group_')])
    if filter_groups:
        group_folders = [g for g in group_folders if g.name in filter_groups]
    
    print(f"\n{'='*80}\nBURN Template - RopeCap Multi-Pose Processor\n{'='*80}")
    print(f"Base: {base_path}\nOutput: {output_base}")
    print(f"Target poses: {target_poses}")
    print(f"Groups: {len(group_folders)}, Stride: {stride_frame}")
    print(f"Share intrinsics by pose: {share_intrinsics_by_pose}")
    print(f"Async mapping: {async_mapping}, Timeout: {mapping_timeout}s\n{'='*80}\n")
    
    output_base.mkdir(parents=True, exist_ok=True)
    job_mgr = MappingJobManager(dashboard_dir=output_base) if async_mapping else None
    
    with open(output_base / 'config_used.yaml', 'w') as f:
        yaml.dump(config, f)
    
    for group_folder in group_folders:
        group_name = group_folder.name
        all_results['groups'][group_name] = {}
        print(f"\n{'='*60}\nProcessing {group_name}\n{'='*60}")
        
        cam_folders = sorted([d for d in group_folder.iterdir() if d.is_dir() and d.name.startswith('cam')])
        if filter_cameras:
            cam_folders = [c for c in cam_folders if c.name in filter_cameras]
        
        for cam_folder in cam_folders:
            cam_name = cam_folder.name
            images_by_pose = collect_multipose_images(cam_folder, target_poses, stride_frame)
            
            if not images_by_pose:
                print(f"  {cam_name}: Skipped (no images in target poses)")
                continue
            
            total_imgs = sum(len(v) for v in images_by_pose.values())
            print(f"\n  {cam_name}: {total_imgs} images from {list(images_by_pose.keys())}")
            
            dp_output = output_base / group_name / cam_name
            
            result = process_single_multipose(
                model, images_by_pose, dp_output, matching_strategy, image_size,
                camera_model=camera_model, share_intrinsics_by_pose=share_intrinsics_by_pose,
                batch_size=batch_size
            )
            
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
            
            if run_mapping and result['db_path']:
                sparse_path = dp_output / 'sparse'
                job_name = f"{group_name}/{cam_name}"
                
                if async_mapping:
                    job_id = job_mgr.start_job(
                        name=job_name, database_path=result['db_path'],
                        image_path=dp_output, output_path=sparse_path,
                        mapper_options=mapper_cfg if mapper_cfg else None
                    )
                    result['mapping_job_id'] = job_id
                else:
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
                        print(f"    Map OK: Reg {result['num_registered']}, Pts {result['num_points3d']}")
                    else:
                        all_results['total_failed'] += 1
                        print(f"    Map FAIL: {map_result.get('error')}")
    
    # Wait for async jobs
    if async_mapping and run_mapping and job_mgr.jobs:
        mapping_results = job_mgr.wait_all(timeout=mapping_timeout)
        for group_name, group_data in all_results['groups'].items():
            for cam_name, result in group_data.items():
                job_id = result.get('mapping_job_id')
                if job_id and job_id in mapping_results:
                    mr = mapping_results[job_id]
                    result['mapping_success'] = mr.get('success', False)
                    result['mapping_time'] = mr.get('elapsed', 0)
                    result['num_registered'] = mr.get('num_registered', 0)
                    result['num_points3d'] = mr.get('num_points3d', 0)
                    if not mr.get('success'):
                        result['error'] = mr.get('error', 'Mapping failed')
                    
                    if result['db_success'] and result['mapping_success']:
                        all_results['total_success'] += 1
                    else:
                        all_results['total_failed'] += 1
        job_mgr.print_summary()
    
    all_results['total_time'] = time() - start_time
    
    print(f"\n\n{'='*80}\nSUMMARY: {all_results['total_processed']} processed, "
          f"{all_results['total_success']} success, {all_results['total_failed']} failed, "
          f"{all_results['total_time']:.1f}s total\n{'='*80}\n")
    
    with open(output_base / 'processing_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    generate_html_report(all_results, output_base / 'report.html')
    print(f"Reports: {output_base / 'processing_results.json'}, {output_base / 'report.html'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='BURN Template - RopeCap Multi-Pose Processor')
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
