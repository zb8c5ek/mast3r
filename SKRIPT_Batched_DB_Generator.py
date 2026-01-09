__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen

Batched DB Generator - Batched processing script for large image sets.
Processes images in overlapping batches (bubbles) of 60 images each.
Each batch overlaps by 10 images with the previous batch for continuity.

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
from pathlib import Path
from time import time

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
from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.image_pairs import make_pairs
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
    # ==============

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


def _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path):
    """
    Copy database and images to COLMAP-style folder structure.
    
    Output structure:
        colmap_output_dir/
        ├── images/
        │   ├── image1.jpg
        │   └── ...
        ├── sparse/          (empty, ready for reconstruction)
        └── database.db
    """
    os.makedirs(colmap_output_dir, exist_ok=True)
    
    # Create images directory and copy images
    images_dir = os.path.join(colmap_output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    for src_path in filelist:
        dst_path = os.path.join(images_dir, os.path.basename(src_path))
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    # Create sparse directory (empty, for later reconstruction)
    sparse_dir = os.path.join(colmap_output_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Copy database
    dst_db_path = os.path.join(colmap_output_dir, 'database.db')
    shutil.copy2(colmap_db_path, dst_db_path)
    
    print(f"  Copied COLMAP structure to: {colmap_output_dir}")
    print(f"    - images/: {len(filelist)} images")
    print(f"    - database.db: copied")
    print(f"    - sparse/: created (empty)")


def get_reconstructed_scene(
        outdir,
        model,
        filelist,
        shared_intrinsics=True,
        mapper='GLOMAP',
        colmap_output_dir=None
):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    
    Args:
        outdir: Output directory
        model: MASt3R model
        filelist: List of image file paths
        shared_intrinsics: Whether to use shared intrinsics
        mapper: Mapper to use - 'GLOMAP', 'COLMAP', or None to skip mapping
        colmap_output_dir: Path to copy COLMAP-style output (images/, sparse/, database.db)
    """
    silent = False
    image_size = 512
    imgs = load_images(filelist, size=image_size, square_ok=True, verbose=not silent)
    assert len(imgs) > 1, "Need at least 2 images to run reconstruction"

    scene_graph_params = ["complete"] # k
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    cache_dir = os.path.join(outdir, 'cache')

    # Use original image path (like SCRIPT_Glomap_on_Images.py)
    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        device = "cuda"
        # TODO: how about set dense matching to True ? -> not very helpful, results: D:\RunningData\ZhiNengDao\75to94-720P_32
        dense_matching = False   # False
        conf_thr = 1.001  # 1.001 previously
        colmap_image_pairs = run_mast3r_matching(model, image_size, 16, device,
                                                 kdata, root_path, image_pairs, colmap_db,
                                                 dense_matching, 5, conf_thr,
                                                 False, 3)
        colmap_db.close()
    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        exit(1)

    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

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
        print("Using COLMAP incremental mapper...")
        # Use pycolmap incremental mapping
        pycolmap.incremental_mapping(
            database_path=colmap_db_path,
            image_path=root_path,
            output_path=reconstruction_path
        )
    else:
        print(f"Skipping mapping (mapper='{mapper}')")
        # Copy files to COLMAP structure if colmap_output_dir is provided
        if colmap_output_dir:
            _copy_to_colmap_structure(colmap_output_dir, colmap_db_path, filelist, root_path)
        return None, None

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
    return scene_state, outfile


def process_images_in_batches(
        dp_images,
        dp_output,
        model,
        batch_size=60,
        overlap=10,
        mapper='GLOMAP',
        start_frame=0,
        spacing=1
):
    """
    Process images in overlapping batches.
    
    Args:
        dp_images: Path to the directory containing images
        dp_output: Path to the output directory
        model: The MASt3R model
        batch_size: Number of images per batch (default: 60)
        overlap: Number of overlapping images between batches (default: 10)
        mapper: Mapper to use - 'GLOMAP', 'COLMAP', or None to skip mapping
        start_frame: Frame index to start from (default: 0). Use this to resume processing.
        spacing: Frame spacing/stride (default: 1). E.g., spacing=2 selects every 2nd frame.
                 Batch still contains batch_size images but spans batch_size*spacing frames.
    """
    # Get all image files
    fps_images = (list(dp_images.glob("*.jpg")) + 
                  list(dp_images.glob("*.png")) + 
                  list(dp_images.glob("*.jpeg")))
    
    # Sort images by name for consistent ordering
    fps_images = sorted(fps_images)
    
    total_images = len(fps_images)
    print(f"\n{'='*80}")
    print(f"Total images found: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Overlap: {overlap}")
    print(f"Spacing: {spacing} (each batch spans {batch_size * spacing} frames)")
    print(f"Start frame: {start_frame}")
    print(f"Mapper: {mapper if mapper else 'None (skip mapping)'}")
    print(f"{'='*80}\n")
    
    assert total_images > 1, "Need at least 2 images to run reconstruction"
    assert start_frame < total_images, f"Start frame {start_frame} >= total images {total_images}"
    assert spacing >= 1, f"Spacing must be >= 1, got {spacing}"
    
    # Calculate batches (step in frame indices, accounting for spacing)
    # Each batch spans batch_size * spacing frames
    # Overlap of N images means overlap of N * spacing frame indices
    step_size = (batch_size - overlap) * spacing
    batch_results = []
    
    # Calculate batch number based on start_frame
    batch_num = start_frame // step_size if start_frame > 0 else 0
    start_idx = start_frame
    
    while start_idx < total_images:
        batch_num += 1
        # End index in frame space (not accounting for spacing yet)
        end_idx_raw = start_idx + batch_size * spacing
        
        # Get images for this batch with spacing
        batch_indices = list(range(start_idx, min(end_idx_raw, total_images), spacing))
        batch_images = [fps_images[i] for i in batch_indices]
        num_images_in_batch = len(batch_images)
        
        # Actual end frame index (last frame in batch)
        end_idx = batch_indices[-1] if batch_indices else start_idx
        
        print(f"\n{'='*80}")
        print(f"Processing Batch {batch_num}")
        print(f"Frame range: {start_idx} to {end_idx} (spacing={spacing}, {num_images_in_batch} images)")
        print(f"{'='*80}\n")
        
        # Create batch-specific output directory with full parameters in name
        # Format: batch_{num}_f{start}_to_f{end}_sp{spacing}_bs{batch_size}
        batch_output = dp_output / f"batch_{batch_num:03d}_f{start_idx:04d}_to_f{end_idx:04d}_sp{spacing}_bs{batch_size}"
        batch_output.mkdir(parents=True, exist_ok=True)
        
        # Process this batch
        try:
            batch_start_time = time()
            scene_state, outfile = get_reconstructed_scene(
                outdir=batch_output,
                model=model,
                filelist=[fp.resolve().as_posix() for fp in batch_images],
                mapper=mapper,
                colmap_output_dir=str(batch_output),
            )
            batch_time = time() - batch_start_time
            
            # Write batch_info.json summary file
            batch_info = {
                'batch_num': batch_num,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'spacing': spacing,
                'batch_size': batch_size,
                'overlap': overlap,
                'num_images': num_images_in_batch,
                'processing_time_seconds': batch_time,
                'paths': {
                    'images': 'images/',
                    'sparse': 'sparse/',
                    'database': 'database.db',
                    'colmap_db': 'cache/colmap.db',
                },
                'source_images': [fp.name for fp in batch_images],
                'success': True
            }
            batch_info_path = batch_output / 'batch_info.json'
            with open(batch_info_path, 'w') as f:
                json.dump(batch_info, f, indent=2)
            
            batch_results.append({
                'batch_num': batch_num,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'spacing': spacing,
                'batch_size': batch_size,
                'num_images': num_images_in_batch,
                'output_dir': batch_output,
                'outfile': outfile,
                'time': batch_time,
                'success': True
            })
            
            print(f"\n✓ Batch {batch_num} completed in {batch_time:.2f}s")
            print(f"  Output saved to: {batch_output}")
            print(f"  Batch info: {batch_info_path}")
            
        except Exception as e:
            print(f"\n✗ Batch {batch_num} FAILED: {str(e)}")
            batch_results.append({
                'batch_num': batch_num,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'spacing': spacing,
                'batch_size': batch_size,
                'num_images': num_images_in_batch,
                'output_dir': batch_output,
                'outfile': None,
                'time': 0,
                'success': False,
                'error': str(e)
            })
        
        # Move to next batch
        if end_idx >= total_images:
            break
        start_idx += step_size
    
    return batch_results


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


if __name__ == "__main__":
    start_time = time()
    
    # Configuration
    dp_images = Path("/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam0/p-20_y+15_r+0")
    dp_output = dp_images.parent / ("mapping3r_batched_%s" % dp_images.name)
    
    # Batch parameters
    BATCH_SIZE = 40  # Number of images per batch
    OVERLAP = 10     # Number of overlapping images between batches
    MAPPER = None  # Options: 'GLOMAP', 'COLMAP', or None to skip mapping
    START_FRAME = 70  # Frame index to start from (0 = beginning, use to resume processing)
    SPACING = 2      # Frame spacing (1 = consecutive, 2 = every 2nd frame, etc.)
    
    # Load model
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    print(f"Loading model from {weights_path}...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    print("Model loaded successfully!\n")
    
    # Process images in batches
    batch_results = process_images_in_batches(
        dp_images=dp_images,
        dp_output=dp_output,
        model=model,
        batch_size=BATCH_SIZE,
        overlap=OVERLAP,
        mapper=MAPPER,
        start_frame=START_FRAME,
        spacing=SPACING
    )
    
    # Print summary
    total_time = time() - start_time
    print_summary(batch_results, total_time)
