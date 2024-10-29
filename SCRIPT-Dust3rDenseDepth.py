__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import copy
import os
import struct

import imageio
import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3rDir.dust3r.image_pairs import make_pairs
from dust3rDir.dust3r.inference import inference
from dust3rDir.dust3r.utils.device import to_numpy
from dust3rDir.dust3r.utils.image import load_images, rgb
from dust3rDir.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from mast3r.model import AsymmetricMASt3R


def depth_map_to_3D_points(depth_map, img_rgb, focal_length, principal_point=None):
    """
    Convert a depth map to 3D points.

    Args:
    - depth_map (numpy.ndarray): Depth map.
    - focal_length (float): Focal length of the camera.
    - principal_point (tuple): Principal point of the camera.

    Returns:
    - numpy.ndarray: 3D points.
    """
    height, width = depth_map.shape
    if principal_point is None:
        u0 = width / 2
        v0 = height / 2
    else:
        u0, v0 = principal_point
    points = np.zeros((height, width, 3))
    colors = np.zeros((height, width, 3))
    for v in range(height):
        for u in range(width):
            z = depth_map[v, u]
            x = (u - u0) * z / focal_length
            y = (v - v0) * z / focal_length
            points[v, u] = [x, y, z]
            colors[v, u] = img_rgb[v, u]

    return points, colors


def interpolate_array(array, size, mode='nearest'):
    """
    Interpolates a NumPy array to the given size.

    Args:
    - array (numpy.ndarray): Input array of shape (H, W, C) or (H, W).
    - size (tuple): Target size (height, width).
    - mode (str): Interpolation mode. Default is 'nearest'.

    Returns:
    - numpy.ndarray: Interpolated array.
    """
    if array.ndim == 2:  # 1-channel depth map
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
    elif array.ndim == 3:  # 3-channel image
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    else:
        raise ValueError("Input array must have 2 or 3 dimensions.")

    interpolated_tensor = F.interpolate(tensor, size=size, mode=mode)
    interpolated_array = interpolated_tensor.squeeze(0).permute(1, 2,
                                                                0).numpy() if array.ndim == 3 else interpolated_tensor.squeeze(
        0).squeeze(0).numpy()

    return interpolated_array


def get_reconstructed_scene(outdir, model, filelist, niter=300, min_conf_thr=3
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
    schedule = 'linear'  # 'linear' or 'cosine'

    LIST_ori_img_size = [Image.open(fp).size for fp in filelist]

    # Check whether there is only one size of image
    if len(set(LIST_ori_img_size)) == 1:
        ori_img_size = LIST_ori_img_size[0][::-1]
    else:
        print("Multiple Image Size Found: ")
        for idx, img_size in enumerate(set(LIST_ori_img_size)):
            print(f"Image {idx} Size: {img_size}")
        raise ValueError("Image Sizes Inconsistent !")

    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    # PointCloudOptimizer by Default
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile, clean_depth_maps_pack = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud,
                                                             mask_sky,
                                                             clean_depth, transparent_cams, cam_size)

    dp_depthmaps = outdir / "depthmaps3r"
    dp_depthmaps.mkdir(parents=True, exist_ok=True)
    # =========== Double Check the DepthMap ===============
    # TODO: make it a debug function, and maybe used in the future.
    # for idx, clean_depth_map in enumerate(clean_depth_maps_pack):
    #     current_img = to_numpy(scene.imgs[idx])
    #     current_depth = clean_depth_map
    #     pts, cols = depth_map_to_3D_points(current_depth, current_img, to_numpy(scene.get_focals()[idx][0]).item())
    #     write_ply(outdir / f"scene_{idx}-clean-reproj.ply", pts.reshape(-1, 3), cols.reshape(-1, 3))
    #     # Interpolate image
    #     current_img_ori = interpolate_array(current_img, size=ori_img_size, mode='nearest')
    #
    #     # Interpolate depth map
    #     current_depth_ori = interpolate_array(current_depth, size=ori_img_size, mode='nearest')
    #
    #     focal_ori = to_numpy(scene.get_focals()[idx][0]).item() * (max(ori_img_size) / image_size)
    #     pts_ori, cols_ori = depth_map_to_3D_points(current_depth_ori, current_img_ori, focal_ori)
    #     write_ply(outdir / f"scene_{idx}-clean-reproj-nn_ori.ply", pts_ori.reshape(-1, 3), cols_ori.reshape(-1, 3))

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])

    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths_viz = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs_viz = [cmap(d / confs_max) for d in confs]

    imgs = []
    viz_imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths_viz[i]))
        imgs.append(rgb(confs[i]))
        viz_imgs.append(np.hstack([rgbimg[i], apply_jet_colormap(depths[i]), confs_viz[i][:, :, :3]]))
    images_to_video(viz_imgs, outdir / "depths.mp4", fps=2)
    # Upscale Depth Map to Ori Size, with Applying Confs

    # Get the Masks
    # scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())
    for idx, (dep, msk) in enumerate(zip(depths, masks)):
        dep_conf = dep.copy()
        dep_conf[~msk] = 0
        current_img = scene.imgs[idx]

        # Interpolate image
        current_img_ori = interpolate_array(current_img, size=ori_img_size, mode='nearest')
        # Interpolate depth map
        current_depth_ori = interpolate_array(dep_conf, size=ori_img_size, mode='nearest')
        current_depth_ori_raw = interpolate_array(dep, size=ori_img_size, mode='nearest')
        focal = to_numpy(scene.get_focals()[idx][0]).item() * (max(ori_img_size) / image_size)
        pts, cols = depth_map_to_3D_points(current_depth_ori, current_img_ori, focal)
        # Format the number in scientific notation with '_' replacing the decimal point
        formatted_min_conf_thr = f"{scene.min_conf_thr:.2e}".replace('.', '_')
        write_ply(outdir / f"scene_{idx}-mct_{formatted_min_conf_thr}.ply", pts.reshape(-1, 3), cols.reshape(-1, 3))
        pts, cols = depth_map_to_3D_points(current_depth_ori_raw, current_img_ori, focal)
        write_ply(outdir / f"scene_{idx}-raw.ply", pts.reshape(-1, 3), cols.reshape(-1, 3))

        # Write Ori Raw Depths Out
        write_array(current_depth_ori_raw, dp_depthmaps / f"{Path(filelist[idx]).stem}.bin")

    return scene, outfile, imgs


def apply_jet_colormap(gray_image):
    """
    Apply jet colormap to a grayscale image.

    Args:
    - gray_image (numpy.ndarray): Grayscale image.

    Returns:
    - numpy.ndarray: Image with jet colormap applied.
    """
    if gray_image.ndim != 2:
        raise ValueError("Input image must be a grayscale image (2D array).")

    # Normalize the grayscale image to the range [0, 1]
    norm_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())

    # Apply the jet colormap
    jet_colormap = pl.cm.jet(norm_image)

    # Discard the alpha channel
    jet_image = jet_colormap[:, :, :3]

    return jet_image


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


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
    outfile, clean_depth_maps_reproj = _convert_scene_output_to_glb(
        outdir, rgbimg, pts3d, msk, focals, cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent
    )
    return outfile, clean_depth_maps_reproj


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


def project_pts3d_to_depthmap(pts, focals, imgs, cams2world, col):
    """
    Project 3D points back to depth maps.

    Args:
    - pts (numpy.ndarray): 3D points of shape (N, 3).
    - focals (numpy.ndarray): Focal lengths of shape (n_imgs, 2).
    - imgs (numpy.ndarray): Images of shape (n_imgs, H, W, 3).
    - cams2world (numpy.ndarray): Camera extrinsics of shape (n_imgs, 4, 4).

    Returns:
    - depth_maps (list of numpy.ndarray): List of depth maps for each image.
    """
    n_imgs = len(imgs)
    H, W, _ = imgs[0].shape
    depth_maps = [np.zeros((H, W), dtype=np.float32) for _ in range(n_imgs)]
    LIST_pts_cams_valid = []
    for i in range(n_imgs):
        # Get camera intrinsics
        fx, fy = focals[i][0], focals[i][0]
        cx, cy = W / 2, H / 2

        # Get camera extrinsics
        cam2world = cams2world[i]
        world2cam = np.linalg.inv(cam2world)

        # Transform points to camera frame
        pts_cam = (world2cam[:3, :3] @ pts.T + world2cam[:3, 3:4]).T

        # Project points to image plane
        x = (pts_cam[:, 0] * fx / pts_cam[:, 2]) + cx
        y = (pts_cam[:, 1] * fy / pts_cam[:, 2]) + cy
        z = pts_cam[:, 2]

        # Filter points that are within the image bounds
        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z > 0)
        # TODO: using CloughTocher2DInterpolator to interpolate the depth

        x = x[valid_mask].astype(int)
        y = y[valid_mask].astype(int)
        z = z[valid_mask]
        pts_cam_valid = pts_cam[valid_mask]
        col_cam_valid = col[valid_mask]
        LIST_pts_cams_valid.append((pts_cam_valid, col_cam_valid))
        # Fill depth map
        depth_maps[i][y, x] = z

    return depth_maps, LIST_pts_cams_valid


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
        # outfile_ply = os.path.join(outdir, 'scene.ply')
        # write_ply(outfile_ply, pts, col)
        clean_depth_maps, LIST_pts_valid = project_pts3d_to_depthmap(pts, focals, imgs, cams2world, col)
        # for i, (pt_cam, col_cam) in tqdm(enumerate(LIST_pts_valid)):
        #     outfile_ply_cam = os.path.join(outdir, f'scene-cam_{i}.ply')
        #     write_ply(outfile_ply_cam, pt_cam, col_cam)
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)
        clean_depth_maps, LIST_pts_valid = None, None

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
    return outfile, clean_depth_maps


def images_to_video(input_source, output_video_path, fps):
    """
    Convert images in a folder or a list of numpy arrays to a video.

    Args:
    - input_source (str or list): The path to the folder containing the images or a list of numpy arrays.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    if isinstance(input_source, str):
        # Load images from folder
        for filename in sorted(os.listdir(input_source)):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
                image_path = os.path.join(input_source, filename)
                image = imageio.imread(image_path)
                images.append(image)
    elif isinstance(input_source, list):
        # Load images from list of numpy arrays
        for image in input_source:
            if isinstance(image, np.ndarray):
                images.append(image)
            else:
                raise ValueError("All elements in the list must be numpy arrays.")
    else:
        raise ValueError("input_source must be either a string (folder path) or a list of numpy arrays.")

    codec = 'libx264'
    # Normalize the images to the range [0, 255] and convert to uint8
    images_uint8 = [(image * 255).astype(np.uint8) for image in images]

    # Save the video using the converted images
    imageio.mimwrite(output_video_path, images_uint8, fps=fps, codec=codec, quality=10)


if __name__ == '__main__':
    from pathlib import Path

    # Locate the Model
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    assert weights_path.exists(), f"Model file {weights_path} not found."
    weights_path = weights_path.as_posix()
    niter = 100
    for min_conf_thr in [3]:
        model = AsymmetricMASt3R.from_pretrained(weights_path).to("cuda")
        dp_images = Path("/d_disk/RunningData/ZhiNengDao/blob2/images")
        dp_output = Path("/d_disk/RunningData/ZhiNengDao/blob2/dust3rGA-ni%04d-conf%02d" % (niter, min_conf_thr))
        fps_images = list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")) + list(dp_images.glob("*.jpeg"))
        assert len(fps_images) > 1, "Need at least 2 images to run reconstruction"

        # For each Image, make a blob, that only its neighbors forward 2 and back ward 2 timestamps are selected

        dp_output.mkdir(parents=True, exist_ok=True)
        scene, outfile, imgs = get_reconstructed_scene(
            outdir=dp_output,
            model=model,
            niter=niter,
            filelist=[fp.resolve().as_posix() for fp in fps_images],
            min_conf_thr=min_conf_thr
        )
