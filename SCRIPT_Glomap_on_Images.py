"""
SCRIPT_Glomap_on_Images.py -- Kapture-free 3D reconstruction from images
=========================================================================

Uses MASt3R matching + GLOMAP/pycolmap mapper.  No kapture dependency.

Usage:
    python SCRIPT_Glomap_on_Images.py

Replaces the old kapture-based pipeline:
  OLD: kapture_import_image_folder_or_list → kapture_to_colmap → run_mast3r_matching
  NEW: ColmapDatabase.connect → run_mast3r_matching  (all-in-one)
"""

__author__ = 'Xuanli CHEN'

import copy
import os
import shutil
import tempfile

import numpy as np
import PIL.Image
import pycolmap
import trimesh
from scipy.spatial.transform import Rotation

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL

# Kapture-free imports from our new mapping.py
from mast3r.colmap.mapping import (
    ColmapDatabase,
    run_mast3r_matching,
    glomap_run_mapper,
)


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
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(filename, 'w') as f:
        f.write(header)
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")
    print(f"Saved point cloud to {filename}")


def get_3D_model_from_scene(silent, scene_state, transparent_cams=False, cam_size=0.05):
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

    fn_ply_output = outfile.replace('.glb', '.ply')
    write_ply(fn_ply_output, pts, col)

    scene.add_geometry(pct)

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
    shared_intrinsics=True,
    conf_thr=1.001,
):
    """From a list of images, run MASt3R matching and GLOMAP reconstruction.

    Kapture-free: uses ColmapDatabase from mast3r.colmap.mapping directly.
    """
    silent = False
    image_size = 512

    assert len(filelist) > 1, "Need at least 2 images to run reconstruction"

    # Compute image pairs using dust3r's load_images + make_pairs
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    scene_graph_params = ["complete"]
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None,
                       symmetrize=True, sim_mat=None)

    # Prepare paths
    cache_dir = os.path.join(outdir, 'cache')
    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]

    # Build image pair list from make_pairs output
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    # Create COLMAP database (kapture-free)
    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)
    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)

    colmap_db = ColmapDatabase.connect(colmap_db_path)

    try:
        device = "cuda"
        dense_matching = False

        # Run MASt3R matching and populate the COLMAP database
        # This replaces: kapture_to_colmap + run_mast3r_matching(kdata, ...)
        colmap_image_pairs = run_mast3r_matching(
            model, image_size, 16, device,
            filelist_relpath, root_path, image_pairs, colmap_db,
            dense_matching, 5, conf_thr,
            False, 3,
            shared_intrinsics=shared_intrinsics,
            camera_model="PINHOLE",
        )
        colmap_db.close()
    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        raise

    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    # Geometric verification
    print("verify_matches")
    pairs_txt = os.path.join(cache_dir, 'pairs.txt')
    with open(pairs_txt, "w") as f:
        for path1, path2 in colmap_image_pairs:
            f.write(f"{path1} {path2}\n")
    pycolmap.verify_matches(colmap_db_path, pairs_txt)

    # Run GLOMAP mapper
    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)

    glomap_run_mapper('glomap', colmap_db_path, reconstruction_path, root_path)

    # Read reconstruction output
    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)
    output_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
    print(output_recon.summary())

    colmap_world_to_cam = {}
    colmap_intrinsics = {}
    images = {}
    num_reg_images = output_recon.num_reg_images()

    for idx, (colmap_imgid, colmap_image) in enumerate(output_recon.images.items()):
        if callable(colmap_image.cam_from_world.matrix):
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix()
        else:
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix

        camera = output_recon.cameras[colmap_image.camera_id]
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        colmap_intrinsics[colmap_imgid] = K

        with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
            images[colmap_imgid] = np.asarray(im)

        if idx + 1 == num_reg_images:
            break

    points3D = []
    num_points3D = output_recon.num_points3D()
    for idx, (pt3d_id, pts3d) in enumerate(output_recon.points3D.items()):
        points3D.append((pts3d.xyz, pts3d.color))
        if idx + 1 == num_points3D:
            break

    scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
    scene_state = GlomapReconState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state)

    return scene_state, outfile


if __name__ == "__main__":
    from pathlib import Path
    from time import time

    start_time = time()

    dp_images = Path("/d_disk/_DataBuffer/WristHeadOsmo/0108/glue-mix-42")
    conf_thr = 4.501
    dp_output = (
        dp_images.parent
        / f"mapping3r_{dp_images.stem}_undist_cam1_conf_{conf_thr:02f}".replace('.', '_')
    )

    fps_images = (
        list(dp_images.glob("*.jpg"))
        + list(dp_images.glob("*.png"))
        + list(dp_images.glob("*.jpeg"))
    )
    assert len(fps_images) > 1, "Need at least 2 images to run reconstruction"

    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')

    get_reconstructed_scene(
        outdir=dp_output,
        model=model,
        filelist=[fp.resolve().as_posix() for fp in fps_images],
        conf_thr=conf_thr,
    )
    print(f"Time taken: {time() - start_time:.2f}s")
