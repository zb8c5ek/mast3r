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


def _share_intrinsics_by_subfolder(db, camera_model: str = 'PINHOLE'):
    """
    Share camera intrinsics by subfolder.
    Images in the same subfolder (e.g., images/cam0/, images/cam1/) share the same camera.
    This is useful for multi-camera rigs where each physical camera has consistent intrinsics.
    """
    import numpy as np
    model_id = CAMERA_MODEL_IDS.get(camera_model, 1)
    
    # Get all images with their paths
    rows = db.execute("SELECT image_id, camera_id, name FROM images").fetchall()
    if not rows:
        return
    
    # First, collect camera info before any modifications
    camera_info = {}
    for row in db.execute("SELECT camera_id, width, height FROM cameras").fetchall():
        camera_info[row[0]] = (row[1], row[2])
    
    if not camera_info:
        return
    
    # Get default dimensions from first camera
    first_cam_id = list(camera_info.keys())[0]
    default_width, default_height = camera_info[first_cam_id]
    
    # Group images by subfolder (e.g., "images/cam0" from "images/cam0/img001.jpg")
    subfolder_to_images = {}
    subfolder_dimensions = {}  # Store dimensions per subfolder
    
    for image_id, camera_id, name in rows:
        # Extract subfolder from path (e.g., "images/cam0" from "images/cam0/img001.jpg")
        parts = name.split('/')
        if len(parts) >= 2:
            subfolder = '/'.join(parts[:-1])  # e.g., "images/cam0"
        else:
            subfolder = ''  # root level images
        
        if subfolder not in subfolder_to_images:
            subfolder_to_images[subfolder] = []
            # Store dimensions from first image's camera
            if camera_id in camera_info:
                subfolder_dimensions[subfolder] = camera_info[camera_id]
            else:
                subfolder_dimensions[subfolder] = (default_width, default_height)
        
        subfolder_to_images[subfolder].append((image_id, camera_id, name))
    
    # Assign new camera IDs to each subfolder
    subfolder_to_new_camera = {}
    new_camera_id = 1
    for subfolder in sorted(subfolder_to_images.keys()):
        subfolder_to_new_camera[subfolder] = new_camera_id
        new_camera_id += 1
    
    # Delete existing cameras
    db.execute("DELETE FROM cameras")
    
    # Insert new cameras (one per subfolder)
    for subfolder, camera_id in subfolder_to_new_camera.items():
        width, height = subfolder_dimensions.get(subfolder, (default_width, default_height))
        
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
            params = np.array([focal, focal, cx, cy], dtype=np.float64)
        
        db.execute("INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
                   (camera_id, model_id, width, height, params.tobytes(), 0))
    
    # Update images to point to their subfolder's camera
    for subfolder, images in subfolder_to_images.items():
        camera_id = subfolder_to_new_camera[subfolder]
        for image_id, _, _ in images:
            db.execute("UPDATE images SET camera_id=? WHERE image_id=?", (camera_id, image_id))
    
    db.commit()
    print(f"  Shared intrinsics: {len(subfolder_to_new_camera)} cameras for {len(rows)} images")


def _run_matching_pipeline(
    root_path: str,
    filelist_relpath: List[str],
    dp_output: Path,
    model: AsymmetricMASt3R,
    matching_strategy: MatchingStrategy,
    image_size: int = 512,
    camera_model: str = 'PINHOLE',
    share_intrinsics_by_subfolder: bool = False,
    batch_size: int = 16,
) -> dict:
    """Core matching pipeline for DB creation.
    
    Args:
        share_intrinsics_by_subfolder: If True, images in the same subfolder share camera intrinsics.
                                       Useful for multi-camera rigs (e.g., images/cam0/, images/cam1/).
        batch_size: Batch size for MASt3R inference (default: 16).
    """
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
        
        # Update camera model - either share by subfolder or use single camera model
        if share_intrinsics_by_subfolder:
            _share_intrinsics_by_subfolder(colmap_db, camera_model)
        else:
            _update_camera_model(colmap_db, camera_model)
        
        colmap_image_pairs = run_mast3r_matching(
            model, image_size, 16, "cuda", kdata, root_path, image_pairs, colmap_db,
            dense_matching=False, pixel_tol=5, matching_strategy=matching_strategy,
            skip_geometric_verification=False, min_len_track=3,
            batch_size=batch_size, chunk_size=batch_size
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
    batch_size: int = 16,
) -> dict:
    """Create COLMAP database from an image folder.
    
    Args:
        batch_size: Batch size for MASt3R inference (default: 16).
    """
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
    
    result = _run_matching_pipeline(str(dp_output), filelist_relpath, dp_output, model, matching_strategy, image_size, camera_model, batch_size=batch_size)
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
    share_intrinsics_by_subfolder: bool = False,
    batch_size: int = 16,
) -> dict:
    """Create COLMAP database from pre-structured images (already copied).
    
    Args:
        share_intrinsics_by_subfolder: If True, images in the same subfolder (e.g., images/cam0/, images/cam1/)
                                       share the same camera intrinsics. Default False.
        batch_size: Batch size for MASt3R inference (default: 16).
    """
    start = time()
    dp_output.mkdir(parents=True, exist_ok=True)
    
    result = _run_matching_pipeline(
        str(root_path), filelist_relpath, dp_output, model, matching_strategy, image_size,
        camera_model=camera_model, share_intrinsics_by_subfolder=share_intrinsics_by_subfolder,
        batch_size=batch_size
    )
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
        mapper_options: Dict of mapper options (min_num_matches, multiple_models, etc.)
    """
    result = {'success': False, 'num_registered': 0, 'num_points3d': 0, 'time': 0, 'error': None}
    start = time()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        opts = pycolmap.IncrementalPipelineOptions()
        
        # Apply options if provided
        if mapper_options:
            # Pipeline-level options
            if 'min_num_matches' in mapper_options:
                opts.min_num_matches = mapper_options['min_num_matches']
            if 'multiple_models' in mapper_options:
                opts.multiple_models = mapper_options['multiple_models']
            if 'extract_colors' in mapper_options:
                opts.extract_colors = mapper_options['extract_colors']
            
            # Note: BA refinement options (ba_refine_focal_length, ba_refine_principal_point, 
            # ba_refine_extra_params) are not exposed in pycolmap Python bindings.
            # COLMAP uses defaults which typically refine focal length and principal point.
        
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


# =============================================================================
# ASYNC SUBPROCESS MAPPING
# =============================================================================
import subprocess
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from queue import Queue
import sys


@dataclass
class MappingJob:
    """Tracks a background mapping subprocess."""
    job_id: str
    database_path: Path
    image_path: Path
    output_path: Path
    process: subprocess.Popen = None
    status_file: Path = None
    result: dict = field(default_factory=dict)
    started_at: float = 0
    finished: bool = False
    

def _mapping_worker_script():
    """Returns the inline script to run mapping in subprocess."""
    return '''
import sys
import json
import pycolmap
from pathlib import Path
from time import time

def run_mapping(db_path, img_path, out_path, opts_json, status_file):
    result = {"success": False, "num_registered": 0, "num_points3d": 0, "time": 0, "error": None, "status": "running"}
    start = time()
    
    def write_status(status_msg):
        result["status"] = status_msg
        result["time"] = time() - start
        with open(status_file, "w") as f:
            json.dump(result, f)
    
    try:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        
        write_status("initializing mapper")
        opts = pycolmap.IncrementalPipelineOptions()
        
        if opts_json:
            mapper_options = json.loads(opts_json)
            if "min_num_matches" in mapper_options:
                opts.min_num_matches = mapper_options["min_num_matches"]
            if "multiple_models" in mapper_options:
                opts.multiple_models = mapper_options["multiple_models"]
            if "extract_colors" in mapper_options:
                opts.extract_colors = mapper_options["extract_colors"]
        
        write_status("running incremental mapping...")
        pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(img_path),
            output_path=str(out_path),
            options=opts
        )
        
        write_status("reading reconstruction stats")
        recon_path = out_path / "0"
        if recon_path.exists():
            try:
                recon = pycolmap.Reconstruction(str(recon_path))
                result["num_registered"] = recon.num_reg_images()
                result["num_points3d"] = recon.num_points3D()
                result["success"] = True
            except:
                result["success"] = True
        
        result["status"] = "completed"
        result["time"] = time() - start
        
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "failed"
        result["time"] = time() - start
    
    with open(status_file, "w") as f:
        json.dump(result, f)
    return result

if __name__ == "__main__":
    db_path, img_path, out_path, opts_json, status_file = sys.argv[1:6]
    run_mapping(db_path, img_path, out_path, opts_json, status_file)
'''


class MappingJobManager:
    """Manages background mapping jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, MappingJob] = {}
        self._job_counter = 0
    
    def start_job(self, database_path: Path, image_path: Path, output_path: Path,
                  mapper_options: dict = None, job_id: str = None) -> MappingJob:
        """Start a mapping job in background subprocess."""
        if job_id is None:
            self._job_counter += 1
            job_id = f"map_{self._job_counter:03d}"
        
        # Create status file in output directory
        output_path.mkdir(parents=True, exist_ok=True)
        status_file = output_path / '.mapping_status.json'
        
        # Write initial status
        with open(status_file, 'w') as f:
            json.dump({"status": "starting", "success": False}, f)
        
        # Prepare options as JSON string
        opts_json = json.dumps(mapper_options) if mapper_options else "{}"
        
        # Create job
        job = MappingJob(
            job_id=job_id,
            database_path=database_path,
            image_path=image_path,
            output_path=output_path,
            status_file=status_file,
            started_at=time()
        )
        
        # Write worker script to temp file and run
        import tempfile
        script_file = output_path / '.mapping_worker.py'
        with open(script_file, 'w') as f:
            f.write(_mapping_worker_script())
        
        # Start subprocess
        job.process = subprocess.Popen(
            [sys.executable, str(script_file), 
             str(database_path), str(image_path), str(output_path), 
             opts_json, str(status_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.jobs[job_id] = job
        print(f"    [BG] Started mapping job {job_id} (PID: {job.process.pid})")
        return job
    
    def check_job(self, job_id: str) -> Optional[dict]:
        """Check job status. Returns result dict if finished, None if still running."""
        if job_id not in self.jobs:
            return {"error": f"Unknown job: {job_id}"}
        
        job = self.jobs[job_id]
        
        if job.finished:
            return job.result
        
        # Check if process is still running
        poll = job.process.poll()
        
        # Read current status
        status = {"status": "unknown"}
        if job.status_file.exists():
            try:
                with open(job.status_file, 'r') as f:
                    status = json.load(f)
            except:
                pass
        
        if poll is not None:
            # Process finished
            job.finished = True
            job.result = status
            job.result['time'] = time() - job.started_at
            
            # Cleanup temp files
            if job.status_file.exists():
                job.status_file.unlink()
            worker_script = job.output_path / '.mapping_worker.py'
            if worker_script.exists():
                worker_script.unlink()
            
            return job.result
        
        return None  # Still running
    
    def get_status(self, job_id: str) -> str:
        """Get current status message for a job."""
        if job_id not in self.jobs:
            return "unknown"
        
        job = self.jobs[job_id]
        if job.finished:
            return job.result.get('status', 'completed')
        
        if job.status_file and job.status_file.exists():
            try:
                with open(job.status_file, 'r') as f:
                    data = json.load(f)
                    elapsed = time() - job.started_at
                    return f"{data.get('status', 'running')} ({elapsed:.0f}s)"
            except:
                pass
        
        return f"running ({time() - job.started_at:.0f}s)"
    
    def wait_all(self, timeout: float = None) -> Dict[str, dict]:
        """Wait for all jobs to finish and return results."""
        results = {}
        start = time()
        
        while True:
            all_done = True
            for job_id in list(self.jobs.keys()):
                result = self.check_job(job_id)
                if result is not None:
                    results[job_id] = result
                else:
                    all_done = False
            
            if all_done:
                break
            
            if timeout and (time() - start) > timeout:
                break
            
            import time as time_module
            time_module.sleep(0.5)
        
        return results
    
    def print_status(self):
        """Print status of all jobs."""
        for job_id, job in self.jobs.items():
            status = self.get_status(job_id)
            if job.finished:
                r = job.result
                if r.get('success'):
                    print(f"    [{job_id}] Done: {r.get('num_registered', '?')} reg, {r.get('num_points3d', '?')} pts, {r.get('time', 0):.1f}s")
                else:
                    print(f"    [{job_id}] Failed: {r.get('error', 'unknown')}")
            else:
                print(f"    [{job_id}] {status}")


# Global job manager instance
_mapping_job_manager = None

def get_mapping_job_manager() -> MappingJobManager:
    """Get or create the global mapping job manager."""
    global _mapping_job_manager
    if _mapping_job_manager is None:
        _mapping_job_manager = MappingJobManager()
    return _mapping_job_manager


def start_mapping_async(database_path: Path, image_path: Path, output_path: Path,
                        mapper_options: dict = None, job_id: str = None) -> str:
    """Start mapping in background subprocess. Returns job_id."""
    mgr = get_mapping_job_manager()
    job = mgr.start_job(database_path, image_path, output_path, mapper_options, job_id)
    return job.job_id


def check_mapping_async(job_id: str) -> Optional[dict]:
    """Check if mapping job is done. Returns result if done, None if still running."""
    return get_mapping_job_manager().check_job(job_id)


def wait_all_mapping_jobs(timeout: float = None) -> Dict[str, dict]:
    """Wait for all background mapping jobs to complete."""
    return get_mapping_job_manager().wait_all(timeout)
