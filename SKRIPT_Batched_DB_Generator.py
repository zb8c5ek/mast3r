__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen

Batched DB Generator - create COLMAP-style matching databases and sparse reconstructions.

Historically this script generated one COLMAP `database.db` per temporal batch/window
(`batch_001`, `batch_002`, ...), hence the name "batched DB". In the current version,
the script processes one provided image set per run, but the "DB" still refers to the
COLMAP matching database that stores images, keypoints, and verified matches.

Output Data Format (COLMAP-style structure):
============================================
<output_directory>/
├── batch_001_f0000_to_f0049_sp1_bs50/
│   ├── images/              # Copied source images for this batch
│   │   ├── image_0000.jpg
│   │   ├── image_0001.jpg
│   │   └── ...
│   ├── sparse/              # COLMAP sparse reconstruction
│   │   └── 0/               # Reconstruction model (cameras.bin, images.bin, points3D.bin)
│   ├── database.db          # COLMAP database with keypoints and matches
│   ├── pairs.txt            # Image pairs used for matching
│   ├── batch_info.json      # Batch parameters and file locations
│   ├── *_scene.glb          # 3D scene model (GLB format)
│   └── *_scene.ply          # 3D point cloud (PLY format)
├── batch_002_f0040_to_f0089_sp1_bs50/
│   └── ... (same structure)
└── ...

Folder naming: batch_{num}_f{start}_to_f{end}_sp{spacing}_bs{batch_size}
This structure is compatible with COLMAP and can be used for further mapping/reconstruction.
"""
import copy
import json
import os
import shutil
import tempfile
import subprocess
import multiprocessing
from pathlib import Path
from time import time
import argparse

import yaml
import PIL.Image
import numpy as np
import pycolmap
from mast3r.model import AsymmetricMASt3R
import trimesh
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import kapture_to_colmap
from scipy.spatial.transform import Rotation

import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r.utils.image import load_images
from dust3rDir.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL
from mast3r.colmap.mapping import kapture_import_image_folder_or_list, glomap_run_mapper
from mast3r.image_pairs import make_pairs
from MOD_FeatureProcess import probe_mast3r_matching, run_mast3r_matching
from mast3r.retrieval.processor import Retriever


class GlomapRecon:
    def __init__(self, world_to_cam, intrinsics, points3d, imgs):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d
        self.imgs = imgs


class GlomapReconState:
    def __init__(self, glomap_recon, should_delete=False, cache_dir=None, outfile_name=None):
        self.glomap_recon = glomap_recon
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def write_ply(filename, points, colors):
    """
    Write points and colors to a PLY file.

    :param filename: The name of the PLY file to write.
    :param points: A numpy array of shape (N, 3) containing the 3D points.
    :param colors: A numpy array of shape (N, 3) containing the RGB colors.
    """
    # Ensure colors are in the range [0, 255]
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    # Create the PLY header
    header = f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    # Write the header and data to the file
    with open(filename, 'w') as ply_file:
        ply_file.write(header)
        for point, color in zip(points, colors):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
    print(f"Saved point cloud to {filename}")


def get_3D_model_from_scene(silent, scene_state, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    recon = scene_state.glomap_recon

    scene = trimesh.Scene()
    pts = np.stack([p[0] for p in recon.points3d], axis=0)
    col = np.stack([p[1] for p in recon.points3d], axis=0)
    pct = trimesh.PointCloud(pts, colors=col)
    # Write PLY Out
    fn_ply_output = outfile.replace('.glb', '.ply')
    write_ply(fn_ply_output, pts, col)

    scene.add_geometry(pct)

    # add each camera
    cams2world = []
    for i, (id, pose_w2c_3x4) in enumerate(recon.world_to_cam.items()):
        intrinsics = recon.intrinsics[id]
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        pose_w2c = np.eye(4)
        pose_w2c[:3, :] = pose_w2c_3x4
        pose_c2w = np.linalg.inv(pose_w2c)
        cams2world.append(pose_c2w)
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else recon.imgs[id], focal,
                      imsize=recon.imgs[id].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)

    return outfile


def _format_strategy_for_foldername(matching_strategy):
    """
    Format matching strategy into a COLMAP-safe folder name string.
    
    Examples:
        ('conf_thres', 2.5) -> 'conf_thres_2_50'
        ('num_pts', 1500) -> 'num_pts_1500'
        [('conf_thres', 2.5), ('num_pts', 1500)] -> 'conf_thres_2_50_num_pts_1500'
    """
    def format_single(strategy_type, strategy_value):
        # Convert value to string, replace . with _ for decimals
        if isinstance(strategy_value, float):
            # Format with 2 decimal places, replace . with _
            val_str = f"{strategy_value:.2f}".replace('.', '_')
        else:
            val_str = str(strategy_value)
        return f"{strategy_type}_{val_str}"
    
    # Check if combined strategy (list of tuples)
    if isinstance(matching_strategy, list):
        parts = [format_single(st, sv) for st, sv in matching_strategy]
        return "_".join(parts)
    else:
        strategy_type, strategy_value = matching_strategy
        return format_single(strategy_type, strategy_value)


def _run_pycolmap_mapper_subprocess(database_path: str, image_path: str, output_path: str, colmap_executable: str = None):
    """
    Run pycolmap incremental mapping in a subprocess.
    This isolates the mapping and prevents crashes from affecting the main process.
    """
    try:
        import pycolmap
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.num_threads = multiprocessing.cpu_count() - 2

        if colmap_executable:
            pycolmap.set_colmap_executable(colmap_executable)

        pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_path,
            output_path=output_path,
            options=mapper_options
        )
        return True
    except Exception as e:
        print(f"Mapping failed: {e}")
        return False


def run_mapper_async(
    database_path: str,
    image_path: str,
    output_path: str,
    mapper_type: str = 'COLMAP',
    wait: bool = False,
    timeout: int = None,
    colmap_executable: str = None
):
    """
    Run mapper in a separate process (async by default).
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to images directory
        output_path: Output directory for reconstruction
        mapper_type: 'COLMAP' or 'GLOMAP'
        wait: If True, wait for completion. If False, return immediately.
        timeout: Timeout in seconds (only if wait=True)
        colmap_executable: Path to the COLMAP executable (e.g., 'D:/COLMAP/colmap.bat')

    Returns:
        Process object if wait=False, success boolean if wait=True
    """
    os.makedirs(output_path, exist_ok=True)
    
    if mapper_type == 'COLMAP':
        # Use multiprocessing to run pycolmap in isolation
        proc = multiprocessing.Process(
            target=_run_pycolmap_mapper_subprocess,
            args=(database_path, image_path, output_path, colmap_executable)
        )
        proc.start()
        
        if wait:
            proc.join(timeout=timeout)
            return proc.exitcode == 0
        return proc
        
    elif mapper_type == 'GLOMAP':
        # Run glomap as subprocess
        cmd = [
            'glomap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(image_path),
            '--output_path', str(output_path),
        ]
        
        if wait:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                print(f"GLOMAP failed: {result.stderr}")
            return result.returncode == 0
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc
    else:
        raise ValueError(f"Unknown mapper type: {mapper_type}")


def _prepare_images_folder(outdir, filelist, original_image_roots):
    """
    Copy images to an 'images' subfolder, preserving the relative path
    from their common ancestor directory to avoid name collisions.

    For example, if image roots are '.../KeyFrames/cam0' and '.../KeyFrames/cam1',
    the common root is '.../KeyFrames'. An image from '.../KeyFrames/cam0/pic.jpg'
    will be copied to 'outdir/images/cam0/pic.jpg'.

    Returns:
        Tuple of (new_root_path, new_filelist) with images in the 'images' subfolder.
    """
    images_dir = Path(outdir) / 'images'
    images_dir.mkdir(exist_ok=True)

    # Find the common ancestor path for all image roots
    # os.path.commonpath works on strings
    str_image_roots = [str(p) for p in original_image_roots]
    common_root = Path(os.path.commonpath(str_image_roots))
    print(f"  Common image root detected: {common_root}")

    new_filelist = []
    for src_path_str in filelist:
        src_path = Path(src_path_str)

        # The relative path from the common ancestor
        try:
            relative_path = src_path.relative_to(common_root)
        except ValueError:
            # This might happen if a file is not under the common root, which is unlikely
            # but we can fall back to a safe default (using the last parts of the path)
            relative_path = Path(*src_path.parts[-3:])

        dst_path = images_dir / relative_path

        # Create the necessary subdirectories
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        new_filelist.append(str(dst_path))

    # The root path for COLMAP is the output directory, so relative paths will be like
    # 'images/cam0/p+0_y+30_r+0/image.jpg'
    return outdir, new_filelist


def _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path):
    """
    Copy database to COLMAP-style folder structure.
    Images should already be in images/ subfolder.

    Output structure:
        colmap_output_dir/
        ├── images/              # Already populated
        │   ├── image1.jpg
        │   └── ...
        ├── sparse/              (empty, ready for reconstruction)
        └── database.db
    """
    os.makedirs(colmap_output_dir, exist_ok=True)

    # Create sparse directory (empty, for later reconstruction)
    sparse_dir = os.path.join(colmap_output_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)

    # Copy database
    dst_db_path = os.path.join(colmap_output_dir, 'database.db')
    shutil.copy2(colmap_db_path, dst_db_path)
    
    print(f"  COLMAP structure ready at: {colmap_output_dir}")
    print(f"    - images/: {len(filelist)} images")
    print(f"    - database.db: copied")
    print(f"    - sparse/: created (empty)")


def _parse_matching_strategy(strategy_cfg, default=('num_pts', 1000)):
    """Parse a matching strategy config block into tuple/list form."""
    strategy_cfg = strategy_cfg or {}
    if 'combined' in strategy_cfg and strategy_cfg['combined']:
        return [(item['type'], item['value']) for item in strategy_cfg['combined']]
    return (
        strategy_cfg.get('type', default[0]),
        strategy_cfg.get('value', default[1])
    )


def _parse_probe_config(config: dict) -> dict:
    """Parse optional low-resolution probe-stage configuration."""
    probe_cfg = config.get('probe') or {}
    probe_batch_size = int(probe_cfg.get('inference_batch_size', 32))
    return {
        'enabled': bool(probe_cfg.get('enabled', False)),
        'image_size': int(probe_cfg.get('image_size', 256)),
        'inference_batch_size': max(1, probe_batch_size),
        'chunk_size': max(1, int(probe_cfg.get('chunk_size', probe_batch_size))),
        'min_matches': max(0, int(probe_cfg.get('min_matches', 100))),
        'matching_strategy': _parse_matching_strategy(
            probe_cfg.get('matching', {}),
            default=('conf_thres', 1.001)
        ),
    }


def get_reconstructed_scene(
        outdir,
        model,
        filelist,
        original_image_roots,
        inference_batch_size=16,
        shared_intrinsics=True,
        mapper='COLMAP',
        colmap_output_dir=None,
        matching_strategy=('conf_thres', 1.001),
        colmap_executable=None,
        probe_config=None
):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene

    Args:
        outdir: Output directory
        model: MASt3R model
        filelist: List of image file paths
        original_image_roots: List of original root directories for the images
        inference_batch_size: Batch size for the model inference step.
        shared_intrinsics: Whether to use shared intrinsics
        mapper: Mapper to use - 'GLOMAP', 'COLMAP', or None to skip mapping
        colmap_output_dir: Path to copy COLMAP-style output (images/, sparse/, database.db)
        matching_strategy: Tuple specifying the filtering strategy:
            - ('conf_thres', float): Fixed confidence threshold (default: 1.001)
            - ('ratio', float): Keep top X% of matches (e.g., 0.1 for top 10%)
            - ('num_pts', int): Keep top N matches by confidence
        colmap_executable: Path to the COLMAP executable (e.g., 'D:/COLMAP/colmap.bat')
        probe_config: Optional low-resolution probe-stage configuration.
    """
    silent = False
    image_size = 512
    patch_size = 16
    probe_config = copy.deepcopy(probe_config or {'enabled': False})
    
    # First copy images to images/ subfolder so COLMAP paths are preserved
    root_path, filelist = _prepare_images_folder(outdir, filelist, original_image_roots)

    imgs = load_images(filelist, size=image_size, square_ok=True, verbose=not silent)
    assert len(imgs) > 1, "Need at least 2 images to run reconstruction"

    scene_graph_params = ["complete"] # k
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    cache_dir = os.path.join(outdir, 'cache')

    # Relative paths will now be 'images/filename.png'
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]
    device = "cuda"
    dense_matching = False
    matching_info = {
        'generated_pairs': len(image_pairs),
        'full_inference': {
            'image_size': image_size,
            'patch_size': patch_size,
            'inference_batch_size': int(inference_batch_size),
        },
    }

    if probe_config.get('enabled'):
        probe_image_size = int(probe_config.get('image_size', 256))
        probe_batch_size = max(1, int(probe_config.get('inference_batch_size', 32)))
        probe_chunk_size = max(probe_batch_size, int(probe_config.get('chunk_size', probe_batch_size)))
        probe_min_matches = max(0, int(probe_config.get('min_matches', 100)))
        probe_matching_strategy = probe_config.get('matching_strategy', ('conf_thres', 1.001))

        print(
            f"Running probe sweep at {probe_image_size}px "
            f"(batch={probe_batch_size}, min_matches={probe_min_matches})..."
        )
        probed_image_pairs, probe_stats = probe_mast3r_matching(
            model=model,
            maxdim=probe_image_size,
            patch_size=patch_size,
            device=device,
            kdata=kdata,
            root_path=root_path,
            image_pairs_kapture=image_pairs,
            min_matches=probe_min_matches,
            dense_matching=dense_matching,
            pixel_tol=5,
            matching_strategy=probe_matching_strategy,
            chunk_size=probe_chunk_size,
            batch_size=probe_batch_size,
        )
        matching_info['probe'] = {
            'enabled': True,
            'image_size': probe_image_size,
            'patch_size': patch_size,
            'inference_batch_size': probe_batch_size,
            'matching_strategy': probe_matching_strategy,
            **probe_stats,
        }
        if len(probed_image_pairs) == 0:
            raise Exception("probe rejected all image pairs")

        image_pairs = probed_image_pairs
        print(
            f"Probe kept {len(image_pairs)} / {probe_stats['pairs_before_probe']} unique pairs "
            f"for full {image_size}px inference."
        )
    else:
        matching_info['probe'] = {
            'enabled': False,
            'pairs_before_probe': len(image_pairs),
            'pairs_after_probe': len(image_pairs),
        }

    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        # TODO: how about set dense matching to True ? -> not very helpful, results: D:\RunningData\ZhiNengDao\75to94-720P_32
        colmap_image_pairs = run_mast3r_matching(
            model=model,
            maxdim=image_size,
            patch_size=patch_size,
            device=device,
            kdata=kdata,
            root_path=root_path,
            image_pairs_kapture=image_pairs,
            colmap_db=colmap_db,
            dense_matching=dense_matching,
            pixel_tol=5,
            matching_strategy=matching_strategy,
            skip_geometric_verification=False,
            min_len_track=3,
            chunk_size=max(16, int(inference_batch_size)),
            batch_size=max(1, int(inference_batch_size)),
        )
    except Exception as e:
        raise RuntimeError(f"matching failed: {e}") from e
    finally:
        colmap_db.close()

    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    matching_info['full_matching'] = {
        'pairs_sent_to_full_matching': len(image_pairs),
        'pairs_with_verified_matches': len(colmap_image_pairs),
    }

    # colmap db is now full, run colmap

    print("verify_matches")
    f = open(cache_dir + '/pairs.txt', "w")
    for image_path1, image_path2 in colmap_image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_db_path, cache_dir + '/pairs.txt')

    # Use sparse_dir if provided, otherwise create 'reconstruction' in cache_dir
    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    
    if mapper == 'GLOMAP':
        print("Using GLOMAP mapper...")
        glomap_run_mapper('glomap', colmap_db_path, reconstruction_path, root_path)
    elif mapper == 'COLMAP':
        print("Using COLMAP incremental mapper (sync)...")
        # Use pycolmap incremental mapping
        if colmap_executable:
            pycolmap.set_colmap_executable(colmap_executable)

        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.num_threads = multiprocessing.cpu_count() - 2

        pycolmap.incremental_mapping(
            database_path=colmap_db_path,
            image_path=root_path,
            output_path=reconstruction_path,
            options=mapper_options
        )
    elif mapper == 'COLMAP_ASYNC':
        print("Launching COLMAP incremental mapper (async subprocess)...")
        # Copy files to COLMAP structure first
        if colmap_output_dir:
            _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path)
        # Start async mapping
        proc = run_mapper_async(
            database_path=colmap_db_path,
            image_path=root_path,
            output_path=reconstruction_path,
            mapper_type='COLMAP',
            wait=False,
            colmap_executable=colmap_executable
        )
        print(f"  Mapper process started (PID: {proc.pid})")
        return None, None, matching_info  # Return immediately, mapping runs in background
    elif mapper == 'GLOMAP_ASYNC':
        print("Launching GLOMAP mapper (async subprocess)...")
        # Copy files to COLMAP structure first
        if colmap_output_dir:
            _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path)
        # Start async mapping
        proc = run_mapper_async(
            database_path=colmap_db_path,
            image_path=root_path,
            output_path=reconstruction_path,
            mapper_type='GLOMAP',
            wait=False
        )
        print(f"  Mapper process started (PID: {proc.pid})")
        return None, None, matching_info  # Return immediately, mapping runs in background
    else:
        print(f"Skipping mapping (mapper='{mapper}')")
        # Copy files to COLMAP structure if colmap_output_dir is provided
        if colmap_output_dir:
            _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path)
        return None, None, matching_info

    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
    print(ouput_recon.summary())

    colmap_world_to_cam = {}
    colmap_intrinsics = {}
    colmap_image_id_to_name = {}
    images = {}
    num_reg_images = ouput_recon.num_reg_images()
    for idx, (colmap_imgid, colmap_image) in enumerate(ouput_recon.images.items()):
        colmap_image_id_to_name[colmap_imgid] = colmap_image.name
        if callable(colmap_image.cam_from_world.matrix):
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix(
            )
        else:
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix
        camera = ouput_recon.cameras[colmap_image.camera_id]
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        colmap_intrinsics[colmap_imgid] = K

        with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
            images[colmap_imgid] = np.asarray(im)

        if idx + 1 == num_reg_images:
            break  # bug with the iterable ?
    points3D = []
    num_points3D = ouput_recon.num_points3D()
    for idx, (pt3d_id, pts3d) in enumerate(ouput_recon.points3D.items()):
        points3D.append((pts3d.xyz, pts3d.color))
        if idx + 1 == num_points3D:
            break  # bug with the iterable ?
    scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
    scene_state = GlomapReconState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state)
    return scene_state, outfile, matching_info


def process_images(
        dp_images,
        dp_output,
        model,
        mapper='GLOMAP',
        matching_strategy=('conf_thres', 1.001),
        colmap_executable=None,
        inference_batch_size=16,
        probe_config=None
):
    """
    Process a set of images all at once.

    Args:
        dp_images: Path or list of paths to directories containing images.
        dp_output: Path to the output directory.
        model: The MASt3R model.
        mapper: Mapper to use - 'GLOMAP', 'COLMAP', or None to skip mapping.
        matching_strategy: Tuple specifying the filtering strategy.
        colmap_executable: Path to the COLMAP executable.
        inference_batch_size: Batch size for the model inference step.
        probe_config: Optional low-resolution pre-filter stage config.
    """
    # Get all image files from single or multiple directories
    fps_images = []
    image_roots = []
    if isinstance(dp_images, list):
        image_roots = [Path(p) for p in dp_images]
        for dp in image_roots:
            fps_images.extend(list(dp.glob("*.jpg")))
            fps_images.extend(list(dp.glob("*.png")))
            fps_images.extend(list(dp.glob("*.jpeg")))
    else:
        dp = Path(dp_images)
        image_roots = [dp]
        fps_images.extend(list(dp.glob("*.jpg")))
        fps_images.extend(list(dp.glob("*.png")))
        fps_images.extend(list(dp.glob("*.jpeg")))

    # Sort images by name for consistent ordering
    fps_images = sorted(fps_images)
    
    total_images = len(fps_images)
    print(f"\n{'='*80}")
    print(f"Total images found: {total_images}")
    print(f"Mapper: {mapper if mapper else 'None (skip mapping)'}")
    print(f"{'='*80}\n")
    
    assert total_images > 1, "Need at least 2 images to run reconstruction"

    dp_output.mkdir(parents=True, exist_ok=True)

    # Process all images in a single batch
    try:
        batch_start_time = time()
        scene_state, outfile, matching_info = get_reconstructed_scene(
            outdir=dp_output,
            model=model,
            filelist=[fp.resolve().as_posix() for fp in fps_images],
            original_image_roots=image_roots,
            mapper=mapper,
            colmap_output_dir=str(dp_output),
            matching_strategy=matching_strategy,
            colmap_executable=colmap_executable,
            inference_batch_size=inference_batch_size,
            probe_config=probe_config,
        )
        batch_time = time() - batch_start_time

        # Write batch_info.json summary file
        batch_info = {
            'num_images': total_images,
            'processing_time_seconds': batch_time,
            'paths': {
                'images': 'images/',
                'sparse': 'sparse/',
                'database': 'database.db',
                'colmap_db': 'cache/colmap.db',
            },
            'matching': {
                'strategy': matching_strategy,
                **matching_info,
            },
            'source_images': [fp.name for fp in fps_images],
            'success': True
        }
        batch_info_path = dp_output / 'run_info.json'
        with open(batch_info_path, 'w') as f:
            json.dump(batch_info, f, indent=2)

        print(f"\n✓ Processing completed in {batch_time:.2f}s")
        print(f"  Output saved to: {dp_output}")
        print(f"  Run info: {batch_info_path}")

        return {
            'output_dir': dp_output,
            'outfile': outfile,
            'time': batch_time,
            'success': True,
            'matching': matching_info,
        }

    except Exception as e:
        print(f"\n✗ Processing FAILED: {str(e)}")
        return {
            'output_dir': dp_output,
            'outfile': None,
            'time': 0,
            'success': False,
            'error': str(e)
        }


def print_summary(batch_results, total_time):
    """Print a summary of all batch processing results."""
    print(f"\n\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}\n")
    
    successful_batches = [b for b in batch_results if b['success']]
    failed_batches = [b for b in batch_results if not b['success']]
    
    print(f"Total batches: {len(batch_results)}")
    print(f"Successful: {len(successful_batches)}")
    print(f"Failed: {len(failed_batches)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
    
    if successful_batches:
        print("Successful batches:")
        for b in successful_batches:
            print(f"  Batch {b['batch_num']}: images {b['start_idx']}-{b['end_idx']} "
                  f"({b['num_images']} images) - {b['time']:.2f}s")
            print(f"    Output: {b['output_dir']}")
    
    if failed_batches:
        print("\nFailed batches:")
        for b in failed_batches:
            print(f"  Batch {b['batch_num']}: images {b['start_idx']}-{b['end_idx']} "
                  f"({b['num_images']} images)")
            print(f"    Error: {b.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}\n")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_from_config(config_path: str):
    """Run batch processing from a YAML config file."""
    start_time = time()
    
    # Load config
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Parse paths
    dp_images_raw = config['paths']['images']
    if isinstance(dp_images_raw, list):
        dp_images = [Path(p) for p in dp_images_raw]
        # Use the parent of the first image folder for the output name
        # Heuristic: find a common parent for a cleaner output path
        try:
            common_parent = Path(os.path.commonpath([str(p) for p in dp_images]))
            output_base_parent = common_parent
            output_base_name = "fused_output"
        except ValueError:
            output_base_parent = dp_images[0].parent
            output_base_name = dp_images[0].name
    else:
        dp_images = Path(dp_images_raw)
        output_base_name = dp_images.name
        output_base_parent = dp_images.parent

    if 'output' in config['paths'] and config['paths']['output']:
        dp_output = Path(config['paths']['output'])
    else:
        dp_output = output_base_parent / ("mapping3r_%s" % output_base_name)

    # Parse mapper
    MAPPER = config.get('mapper', None)
    
    # Parse COLMAP executable path
    COLMAP_EXECUTABLE = config.get('colmap_executable', None)

    # Parse matching strategy
    match_cfg = config.get('matching', {})
    MATCHING_STRATEGY = _parse_matching_strategy(match_cfg, default=('num_pts', 1000))
    PROBE_CONFIG = _parse_probe_config(config)
    
    # Parse model config
    model_cfg = config.get('model', {})
    model_name = model_cfg.get('name', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    checkpoint_dir = model_cfg.get('checkpoint_dir', 'checkpoints')
    inference_batch_size = model_cfg.get('inference_batch_size', 16)

    # Load model
    weights_path = Path(checkpoint_dir) / (model_name + '.pth')
    weights_path = weights_path.resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Process images
    result = process_images(
        dp_images=dp_images,
        dp_output=dp_output,
        model=model,
        mapper=MAPPER,
        matching_strategy=MATCHING_STRATEGY,
        colmap_executable=COLMAP_EXECUTABLE,
        inference_batch_size=inference_batch_size,
        probe_config=PROBE_CONFIG,
    )
    
    # Print summary
    total_time = time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds.")
    if result['success']:
        print("✅ Run finished successfully.")
    else:
        print(f"❌ Run failed with error: {result.get('error')}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batched DB Generator for MASt3R')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML config file (e.g., configs/sample.yml)')
    args = parser.parse_args()

    if args.config:
        # Run from config file
        run_from_config(args.config)
    else:
        # This legacy mode is no longer supported with the new structure.
        # Please use a config file.
        print("Please use a YAML configuration file with the --config argument.")
        print("Example: python SKRIPT_Batched_DB_Generator.py --config configs/sample.yml")
