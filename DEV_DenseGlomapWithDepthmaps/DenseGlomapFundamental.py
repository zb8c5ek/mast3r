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
import numpy as np
from typing import List, Tuple, Union

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


def put_images_into_segment_pairs():
    pass


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


def run_mast3r_matching(model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
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
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    # compute 2D-2D matching from dust3r inference
    # TODO: here the pts3D is already generated, perhaps dense depth map can be further generated as well
    # TODO: the confidence lower by observing or hindering, inler points or outer points, might work.
    chunk_size = 12
    silent = False
    niter = 100
    for chunk in tqdm(range(0, len(matching_pairs), chunk_size)):
        pairs_chunk = matching_pairs[chunk:chunk + chunk_size]
        output = inference(pairs_chunk, model, device, batch_size=12, verbose=not silent)
        pred1, pred2 = output['pred1'], output['pred2']
        # TODO handle caching
        im_images_chunk = get_im_matches(pred1, pred2, pairs_chunk, image_to_colmap,
                                         im_keypoints, conf_thr, not dense_matching, pixel_tol)
        im_matches.update(im_images_chunk.items())

        # Generate Dense Depth Using Dust3r
        mode = GlobalAlignerMode.PointCloudOptimizer if chunk_size > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
        # PointCloudOptimizer by Default
        lr = 0.01

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule='linear', lr=lr)
        # TODO: set a chunk outdir, can be upto 12 imgs, or say, blobs
        outfile, clean_depth_maps_pack = get_3D_model_from_scene(outdir, silent, scene,
                                                                 min_conf_thr=3, as_pointcloud=False,
                                                                 mask_sky=False,
                                                                 clean_depth=True,
                                                                 transparent_cams=False,
                                                                 cam_size=0.05)
        # min_conf_thr=3, as_pointcloud=False, mask_sky=False,
        # clean_depth=False, transparent_cams=False, cam_size=0.05
    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs


# def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
#                             clean_depth=False, transparent_cams=False, cam_size=0.05):
#     """
#     extract 3D_model (glb file) from a reconstructed scene
#     """
#     if scene is None:
#         return None
#     # post processes
#     if clean_depth:
#         scene = scene.clean_pointcloud()
#     if mask_sky:
#         scene = scene.mask_sky()
#     # get optimized values from scene
#     rgbimg = scene.imgs
#     focals = scene.get_focals().cpu()
#     cams2world = scene.get_im_poses().cpu()
#     # 3D pointcloud from depthmap, poses and intrinsics
#     pts3d = to_numpy(scene.get_pts3d())
#     scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
#     msk = to_numpy(scene.get_masks())
#     outfile, clean_depth_maps_reproj = _convert_scene_output_to_glb(
#         outdir, rgbimg, pts3d, msk, focals, cams2world,
#         as_pointcloud=as_pointcloud,
#         transparent_cams=transparent_cams, cam_size=cam_size, silent=silent
#     )
#     return outfile, clean_depth_maps_reproj


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


def kapture_import_image_folder_or_list(images_path: Union[str, Tuple[str, List[str]]], use_single_camera=False) -> kapture.Kapture:
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
