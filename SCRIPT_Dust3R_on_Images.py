__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import copy
import os

import matplotlib.pyplot as pl
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3rDir.dust3r.image_pairs import make_pairs
from dust3rDir.dust3r.inference import inference
from dust3rDir.dust3r.utils.device import to_numpy
from dust3rDir.dust3r.utils.image import load_images, rgb
from dust3rDir.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from mast3r.model import AsymmetricMASt3R


def get_reconstructed_scene(outdir, model, filelist, niter=300, min_conf_thr=3,
                           ):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    silent = False
    image_size = 512
    device = "cuda"
    scenegraph_type = "complete"  # k
    transparent_cams = False
    cam_size = 0.05
    clean_depth = True
    mask_sky = False
    as_pointcloud = True
    schedule = 'linear' # 'linear' or 'cosine'

    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


if __name__ == '__main__':
    from pathlib import Path

    # Locate the Model
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    assert weights_path.exists(), f"Model file {weights_path} not found."
    weights_path = weights_path.as_posix()

    model = AsymmetricMASt3R.from_pretrained(weights_path).to("cuda")
    dp_images = Path("/d_disk/RunningData/ZhiNengDao/75to94-720P_32/images")
    dp_output = Path("/d_disk/RunningData/ZhiNengDao/75to94-720P_32/dust3rGA")
    fps_images = list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")) + list(dp_images.glob("*.jpeg"))
    assert len(fps_images) > 1, "Need at least 2 images to run reconstruction"

    # For each Image, make a blob, that only its neighbors forward 2 and back ward 2 timestamps are selected
    if len(fps_images) > 20:
        fps_images = fps_images[:20]    # 32 Images cause Docker shut down.
    dp_output.mkdir(parents=True, exist_ok=True)
    get_reconstructed_scene(
        outdir=dp_output,
        model=model,
        filelist=[fp.resolve().as_posix() for fp in fps_images]
    )
