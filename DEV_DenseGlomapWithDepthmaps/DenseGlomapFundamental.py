__author__ = 'Xuanli CHEN'

import trimesh
from scipy.spatial.transform import Rotation
from dust3rDir.dust3r.viz import CAM_COLORS, add_scene_cam, OPENGL

"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""

"""
a forked version of D:\mast3r\mast3r\colmap\mapping.py
"""
import pycolmap
import os
import os.path as path
import kapture.io
import kapture.io.csv
import subprocess
import PIL
from tqdm import tqdm
import PIL.Image
from pathlib import Path
import numpy as np
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F


from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import export_matches, get_im_matches
from dust3rDir.dust3r.utils.device import to_numpy
import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r_visloc.datasets.utils import get_resize_function
from dust3rDir.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import kapture
from kapture.converter.colmap.database_extra import get_colmap_camera_ids_from_db, get_colmap_image_ids_from_db
from kapture.utils.paths import path_secure
from dust3rDir.dust3r.inference import inference
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def scene_prepare_images(root: str, maxdim: int, patch_size: int, image_paths: List[str]):
    images = []
    # image loading
    for idx in tqdm(range(len(image_paths)), desc="Preparing images"):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        # image dictionary
        images.append({'img': rgb_tensor.unsqueeze(0),
                       'true_shape': np.int32([rgb_tensor.shape[1:]]),
                       'to_orig': to_orig,
                       'idx': idx,
                       'instance': image_paths[idx],
                       'orig_shape': np.int32([H, W])})
    return images


def remove_duplicates(images, image_pairs):
    pairs_added = set()
    pairs = []
    for (i, _), (j, _) in image_pairs:
        smallidx, bigidx = min(i, j), max(i, j)
        if (smallidx, bigidx) in pairs_added:
            continue
        pairs_added.add((smallidx, bigidx))
        pairs.append((images[i], images[j]))
    return pairs


def run_mast3r_matching(dp_output, model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
                        kdata: kapture.Kapture, root_path: str, image_pairs_kapture: List[Tuple[str, str]],
                        colmap_db,
                        dense_matching: bool, pixel_tol: int, conf_thr: float, skip_geometric_verification: bool,
                        min_len_track: int):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}
    # TODO: ts stands for timestamp, kapture might be a good framework, worth learning a bit. yet perhaps, COLMAP cam would be more direct ?
    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)

    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    fps_images = []
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        fps_images.append(Path(os.path.join(root_path, image_path)))
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    # compute 2D-2D matching from dust3r inference

    silent = False
    niter = 100 # 100 or 300 the loss are just about 0.20 something.

    output = inference(matching_pairs, model, device, batch_size=12, verbose=not silent)
    pred1, pred2 = output['pred1'], output['pred2']
    # TODO handle caching
    im_images_chunk = get_im_matches(pred1, pred2, matching_pairs, image_to_colmap,
                                     im_keypoints, conf_thr, not dense_matching, pixel_tol)
    im_matches.update(im_images_chunk.items())

    # Generate Dense Depth Using Dust3r
    mode = GlobalAlignerMode.PointCloudOptimizer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    # PointCloudOptimizer by Default
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule='linear', lr=lr)
    # TODO: Follow up the depthmaps pipeline
    # TODO: check about using mesh to project the points
    outfile, clean_depth_maps_pack = get_3D_model_from_scene_d3r_dense(
        dp_output, silent, scene,
        min_conf_thr=3, as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05)

    # Output Depth Maps
    dp_depthmaps = dp_output / "depthmapsd3r"
    dp_cache_pc = dp_output / "cache" / "pointclouds"
    dp_depthmaps.mkdir(parents=True, exist_ok=True)
    dp_cache_pc.mkdir(parents=True, exist_ok=True)
    masks = to_numpy(scene.get_masks())
    LIST_ori_img_size = [PIL.Image.open(fp).size for fp in fps_images]

    # Check whether there is only one size of image
    if len(set(LIST_ori_img_size)) == 1:
        ori_img_size = LIST_ori_img_size[0][::-1]
    else:
        print("Multiple Image Size Found: ")
        for idx, img_size in enumerate(set(LIST_ori_img_size)):
            print(f"Image {idx} Size: {img_size}")
        raise ValueError("Image Sizes Inconsistent !")
    for idx, clean_depth_map in enumerate(clean_depth_maps_pack):
        current_img = to_numpy(scene.imgs[idx])
        current_depth = clean_depth_map
        pts, cols = depth_map_to_3D_points(current_depth, current_img, to_numpy(scene.get_focals()[idx][0]).item())
        write_ply(dp_cache_pc / f"scene_{idx}-clean-reproj.ply", pts.reshape(-1, 3), cols.reshape(-1, 3))
        # Interpolate image
        current_img_ori = interpolate_array(current_img, size=ori_img_size, mode='nearest')

        # Interpolate depth map
        current_depth_ori = interpolate_array(current_depth, size=ori_img_size, mode='nearest')

        focal_ori = to_numpy(scene.get_focals()[idx][0]).item() * (max(ori_img_size) / maxdim)
        pts_ori, cols_ori = depth_map_to_3D_points(current_depth_ori, current_img_ori, focal_ori)
        write_ply(dp_cache_pc / f"scene_{idx}-clean-reproj-nn_ori.ply", pts_ori.reshape(-1, 3), cols_ori.reshape(-1, 3))
    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs



def pycolmap_run_triangulator(colmap_db_path, prior_recon_path, recon_path, image_root_path):
    print("running mapping")
    reconstruction = pycolmap.Reconstruction(prior_recon_path)
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        refine_intrinsics=False,
    )


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({'multiple_models': False,
                                                     'extract_colors': True,
                                                     })
    )


def glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, glomap_bin)
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')


def kapture_import_image_folder_or_list(images_path: Union[str, Tuple[str, List[str]]],
                                        use_single_camera=False) -> kapture.Kapture:
    images = kapture.RecordsCamera()

    if isinstance(images_path, str):
        images_root = images_path
        file_list = [path.relpath(path.join(dirpath, filename), images_root)
                     for dirpath, dirs, filenames in os.walk(images_root)
                     for filename in filenames]
        file_list = sorted(file_list)
    else:
        images_root, file_list = images_path

    sensors = kapture.Sensors()
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with PIL.Image.open(path.join(images_root, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            print(f'Skipping invalid image file {filename}')
            continue

        camera_id = f'sensor'
        if use_single_camera and camera_id not in sensors:
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)
        elif use_single_camera:
            assert sensors[camera_id].camera_params[0] == width and sensors[camera_id].camera_params[1] == height
        else:
            camera_id = camera_id + f'{n}'
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

        images[(n, camera_id)] = path_secure(filename)  # don't forget windows

    return kapture.Kapture(sensors=sensors, records_camera=images)


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


def get_3D_model_from_scene_d3r_dense(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
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
        clean_depth_maps, LIST_pts_valid = project_pts3d_to_depthmap(pts, focals, imgs, cams2world, col)
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


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0]+[len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)

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