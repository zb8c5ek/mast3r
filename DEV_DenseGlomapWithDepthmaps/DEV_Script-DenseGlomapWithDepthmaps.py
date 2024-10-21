__author__ = 'Xuanli CHEN'

from DEV_DenseGlomapWithDepthmaps.DenseGlomapFundamental import get_3D_model_from_scene

"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import os
import shutil
import tempfile

import PIL.Image
import numpy as np
import pycolmap
from mast3r.model import AsymmetricMASt3R
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import kapture_to_colmap
from dust3rDir.dust3r.utils.device import to_numpy

import mast3r.utils.path_to_dust3r  # noqa
from dust3rDir.dust3r.utils.image import load_images
from DenseGlomapFundamental import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.image_pairs import make_pairs

class BlobDivider(object):
    def __init__(self, dp_input):
        self.fps_all_images = list(dp_input.glob("*.jpg")) + list(dp_input.glob("*.png")) + list(dp_input.glob("*.jpeg"))

        self.fps2ids = {fp: idx for idx, fp in enumerate(self.fps_all_images)}
        self.ids2fps = {idx: fp for idx, fp in enumerate(self.fps_all_images)}

        self.ids_and_its_ts = self.ts_extract_()
        self.ts_and_its_imgids = self.assign_each_img_to_its_ts_()
        self.LIST_all_ts = list(self.ts_and_its_imgids.keys())
        self.LIST_all_ts.sort()

    def ts_extract_(self):
        """
        Extract the timestamps from the image names and associate them with IDs.
        """
        timestamps = {}
        for idx, image in enumerate(self.fps_all_images):
            ts = int(image.name.split("_")[-1][5:-4])
            timestamps[idx] = ts
        return timestamps

    def assign_each_img_to_its_ts_(self):
        """
        Assign each image to its timestamp.
        """
        ts_images = {}
        for idx, ts in self.ids_and_its_ts.items():
            if ts not in ts_images:
                ts_images[ts] = []
            ts_images[ts].append(idx)

        return ts_images

    def get_blob_division(self, num_neighbor_ts=1):
        """
        Divide the images into blobs based on the timestamps.
        """
        blobs = {}
        for idx in range(num_neighbor_ts, len(self.LIST_all_ts) - 1):
            start_ts = self.LIST_all_ts[idx - num_neighbor_ts]
            end_ts = self.LIST_all_ts[idx + num_neighbor_ts]
            LIST_blob_ts = [ts for ts in self.LIST_all_ts if start_ts <= ts <= end_ts]
            LIST_blob_img_fps = []
            for ts in LIST_blob_ts:
                LIST_blob_img_fps += [self.ids2fps[img_id] for img_id in self.ts_and_its_imgids[ts]]
            blobs[(start_ts, end_ts)] = LIST_blob_img_fps
        return blobs


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


def get_reconstructed_scene(
        dp_output,
        model,
        filelist,
        shared_intrinsics=False
):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    silent = False
    image_size = 512
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    assert len(imgs) > 1, "Need at least 2 images to run reconstruction"

    scene_graph_params = ["complete"] # k
    scene_graph = '-'.join(scene_graph_params)

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    cache_dir = dp_output / 'cache'

    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    # TODO: define and associate sensor and data properly
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
        # Comment: how about set dense matching to True ? -> not very helpful, results: D:\RunningData\ZhiNengDao\75to94-720P_32
        dense_matching = True   # False
        conf_thr = 1.001  # 1.001 previously
        colmap_image_pairs = run_mast3r_matching(dp_output, model, image_size, 16, device,
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
    f = open(cache_dir / 'pairs.txt', "w")
    for image_path1, image_path2 in colmap_image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_db_path, cache_dir.as_posix() + '/pairs.txt')

    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    glomap_run_mapper('glomap', colmap_db_path, reconstruction_path, root_path)

    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=dp_output)

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


if __name__ == "__main__":
    from pathlib import Path
    from time import time

    dp_images = Path("/d_disk/RunningData/ZhiNengDao/20-from-2075-to-94-720P_160/images")
    dp_output = Path("/d_disk/RunningData/ZhiNengDao/20-from-2075-to-94-720P_160/blob3recon")
    fps_images_all = list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")) + list(dp_images.glob("*.jpeg"))
    assert len(fps_images_all) > 1, "Need at least 2 images to run reconstruction"
    model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')

    # Analyze the time-stamps, each time feed 3 time stamps to the model
    bd_ins = BlobDivider(dp_images)
    blobs = bd_ins.get_blob_division(num_neighbor_ts=1)

    for blob_idx, (start_ts, end_ts) in enumerate(blobs.keys()):
        start_time = time()
        print(f"Processing Blob: {blob_idx}")
        fps_images = blobs[(start_ts, end_ts)]
        dp_output_blob = dp_output / f"blob_{blob_idx}-start{start_ts}_end{end_ts}"
        dp_output_blob.mkdir(parents=True, exist_ok=True)
        scene_state, outfile = get_reconstructed_scene(
            dp_output=dp_output_blob,
            model=model,
            filelist=[fp.resolve().as_posix() for fp in fps_images],
        )
        print(f"Time taken for blob {blob_idx}: {time() - start_time} seconds.")