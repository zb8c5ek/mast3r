__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen

Batched processing script for large image sets.
Processes images in overlapping batches (bubbles) of 60 images each.
Each batch overlaps by 10 images with the previous batch for continuity.
"""
import copy
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


def get_reconstructed_scene(
        outdir,
        model,
        filelist,
        shared_intrinsics=False,
        mapper='GLOMAP'
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
    """
    silent = False
    image_size = 512
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    assert len(imgs) > 1, "Need at least 2 images to run reconstruction"

    scene_graph_params = ["complete"] # k
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    cache_dir = os.path.join(outdir, 'cache')

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
        mapper='GLOMAP'
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
    print(f"Mapper: {mapper if mapper else 'None (skip mapping)'}")
    print(f"{'='*80}\n")
    
    assert total_images > 1, "Need at least 2 images to run reconstruction"
    
    # Calculate batches
    step_size = batch_size - overlap
    batch_results = []
    
    batch_num = 0
    start_idx = 0
    
    while start_idx < total_images:
        batch_num += 1
        end_idx = min(start_idx + batch_size, total_images)
        
        # Get images for this batch
        batch_images = fps_images[start_idx:end_idx]
        num_images_in_batch = len(batch_images)
        
        print(f"\n{'='*80}")
        print(f"Processing Batch {batch_num}")
        print(f"Images: {start_idx} to {end_idx-1} ({num_images_in_batch} images)")
        print(f"{'='*80}\n")
        
        # Create batch-specific output directory
        batch_output = dp_output / f"batch_{batch_num:03d}_images_{start_idx:04d}_to_{end_idx-1:04d}"
        batch_output.mkdir(parents=True, exist_ok=True)
        
        # Process this batch
        try:
            batch_start_time = time()
            scene_state, outfile = get_reconstructed_scene(
                outdir=batch_output,
                model=model,
                filelist=[fp.resolve().as_posix() for fp in batch_images],
                mapper=mapper,
            )
            batch_time = time() - batch_start_time
            
            batch_results.append({
                'batch_num': batch_num,
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'num_images': num_images_in_batch,
                'output_dir': batch_output,
                'outfile': outfile,
                'time': batch_time,
                'success': True
            })
            
            print(f"\n✓ Batch {batch_num} completed in {batch_time:.2f}s")
            print(f"  Output saved to: {batch_output}")
            
        except Exception as e:
            print(f"\n✗ Batch {batch_num} FAILED: {str(e)}")
            batch_results.append({
                'batch_num': batch_num,
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
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
    dp_images = Path("/d_disk/_DataTemp/Apart/fuse-rig/p+15_y+15_r+0")
    dp_output = dp_images.parent / ("mapping3r_batched_%s" % dp_images.name)
    
    # Batch parameters
    BATCH_SIZE = 60  # Number of images per batch
    OVERLAP = 10     # Number of overlapping images between batches
    MAPPER = None  # Options: 'GLOMAP', 'COLMAP', or None to skip mapping
    
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
        mapper=MAPPER
    )
    
    # Print summary
    total_time = time() - start_time
    print_summary(batch_results, total_time)
