__author__ = 'Xuanli CHEN'

from tensorboard.compat.tensorflow_stub.error_codes import UNIMPLEMENTED

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
from tqdm import tqdm
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

from EssentialCOLMapper import COLMapper3r

class BlobDivider(object):
    def __init__(self, dp_input):
        self.fps_all_images = list(dp_input.glob("*.jpg")) + list(dp_input.glob("*.png")) + list(dp_input.glob("*.jpeg"))
        self.fps_all_images.sort()
        # in new version of time-format, by sort itself, images would be in time sequence.
        #   Details check: https://www.notion.so/Run-though-the-Vehicle-dataset-12e8ff7304658073ab76c5a7bf48f86f?pvs=4
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
            ts = int(image.name.split("_")[0]) + int(image.name.split("_")[1]) / 1e9    # 1e9 is the nano-second
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

    def get_blob_division(self, num_neighbor_ts=(0, 1)):
        """
        Divide the images into blobs based on the timestamps.
        """
        blobs = {}
        for idx in range(num_neighbor_ts[0], len(self.LIST_all_ts) - num_neighbor_ts[1]):
            start_ts = self.LIST_all_ts[idx - num_neighbor_ts[0]]
            end_ts = self.LIST_all_ts[idx + num_neighbor_ts[1]]
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
        shared_intrinsics=False,
        silent=True,
        skip_GLOMAP=False
):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
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
        # TODO: add the COLMAP-DSP mapper, which seems to be with good performance for the moment.
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        device = "cuda"
        # Comment: how about set dense matching to True ? -> not very helpful, results: D:\RunningData\ZhiNengDao\75to94-720P_32
        dense_matching = True   # False
        conf_thr = 4.001 # 1.001 previously
        colmap_image_pairs = run_mast3r_matching(dp_output, model, image_size, 16, device,
                                                 kdata, root_path, image_pairs, colmap_db,
                                                 dense_matching, 5, conf_thr,
                                                 False, 3, silent=silent)
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


    if not skip_GLOMAP:

        reconstruction_path = os.path.join(cache_dir, "reconstruction")
        if os.path.isdir(reconstruction_path):
            shutil.rmtree(reconstruction_path)
        os.makedirs(reconstruction_path)
        glomap_run_mapper('glomap', colmap_db_path, reconstruction_path, root_path)

        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=dp_output)

        ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
        print(ouput_recon.summary())
        # Export GLOMAP Reconstruction to 3D Model
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
        try:
            scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
            scene_state = GlomapReconState(scene, False, cache_dir, outfile_name)
            outfile = get_3D_model_from_scene(silent, scene_state)
        except Exception as e:
            print(f'Error {e}')
            scene_state = None
            outfile = None
    # return scene_state, outfile


def tell_whether_it_belongs_to_the_sling(fp_image):

    ts, ns, stem, stem_1, rot_angle, ts_raw = fp_image.name.split("_")

    slings = [stem]

    if stem == "front":
        if rot_angle.startswith('-'):
            slings.append("right")
        else:
            angle = int(rot_angle)
            if angle > 0:
                slings.append("left")
    elif stem == "left":
        if rot_angle.startswith('-'):
            slings.append("front")
        else:
            angle = int(rot_angle)
            if angle > 0:
                slings.append("rear")
    elif stem == "right":
        if rot_angle.startswith('-'):
            slings.append("rear")
        else:
            angle = int(rot_angle)
            if angle > 0:
                slings.append("front")
    elif stem == "rear":
        if rot_angle.startswith('-'):
            slings.append("left")
        else:
            angle = int(rot_angle)
            if angle > 0:
                slings.append("right")

    return slings


if __name__ == "__main__":
    from pathlib import Path
    from time import time
    from datetime import datetime
    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_params = (1, 1)
    dp_images = Path(r"/d_disk/RunningData/Cone2/undistorted_2024-11-05_16-06-32/DEVcache_sfm-frames_ts-590_te-594_int-4_num-144/images")
    CHOICE_blob_mode = ['sling']   # ['sling', '360'] # ONLY One is Supperted for Now.
    if len(CHOICE_blob_mode) > 1:
        raise UNIMPLEMENTED("Only One Mode is Supported for Now.")
    # model_name = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    dp_output = dp_images.parent / f"{model_name.split('_')[0]}_blobs-{blob_params[0] + blob_params[1] + 1}_recon_{current_time}_{CHOICE_blob_mode[0]}"
    # TODO: make it a config file, and run from there.
    # TODO: Limit the sequence to have like in total <= 150 images, so that the processing time is about 10 min for Mapping.
    # TODO: add a config file for im_conf enabble.
    # TODO: Add Output Mesh Option, in fact, as default.
    # TODO: shall load the Recon Model first, then using the know poses.
    # TODO: the start poses shall be available from the InstantSPlat -> Firstly, apply the Mesh Part to Instant Splat. and Move that Dust3r as default version.
    FLAG_ohne_rear = True
    FLAG_all_mappings = True
    FLAG_skip_GLOMAP = False
    FLAG_silent = False
    # ================================================================
    fps_images_all = list(dp_images.glob("*.jpg")) + list(dp_images.glob("*.png")) + list(dp_images.glob("*.jpeg"))
    assert len(fps_images_all) > 1, "Need at least 2 images to run reconstruction"

    weights_path = Path("checkpoints/" + model_name + '.pth').resolve()
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')

    # Analyze the time-stamps, each time feed 3 time stamps to the model
    bd_ins = BlobDivider(dp_images)
    blobs = bd_ins.get_blob_division(num_neighbor_ts=blob_params)
    if FLAG_all_mappings:
        # TODO: mappinf all in the sling use some CPUs in the background, when blobs finish processing, join them.
        # COLMAPPer can first load the models to see performance first.
        DICT_blob_sparse_mapper = COLMapper3r(dp_images, dp_images.parent / "cache-sparse-all-multi-cam-default")

    for blob_idx, (start_ts, end_ts) in tqdm(enumerate(blobs.keys()), total=len(blobs)):
        start_time = time()
        print(f"Processing Blob: {blob_idx}")
        fps_images = blobs[(start_ts, end_ts)]
        blob_mode_str = '-'.join(CHOICE_blob_mode)
        dp_output_blob = dp_output / f"blob_{blob_idx:04d}-{blob_mode_str}-start{start_ts}_end{end_ts}"
        dp_output_blob.mkdir(parents=True, exist_ok=True)
        dp_blob_images = dp_output_blob / "images"
        dp_blob_images.mkdir(parents=True, exist_ok=True)
        fps_blob_images = []
        for fp_image in fps_images:
            if FLAG_ohne_rear and "rear" in fp_image.name:
                continue
            shutil.copy(fp_image, dp_blob_images / fp_image.name)
            fps_blob_images.append(dp_blob_images / fp_image.name)

        # Dense Recon for the 360
        if "360" in CHOICE_blob_mode:
            get_reconstructed_scene(
                dp_output=dp_output_blob,
                model=model,
                filelist=[fp.resolve().as_posix() for fp in fps_blob_images],
                silent=FLAG_silent,
                skip_GLOMAP=FLAG_skip_GLOMAP
            )
            print(f"Time taken for blob {blob_idx}: {time() - start_time} seconds.")

        # Dense Recon for the Sling
        if "sling" in CHOICE_blob_mode:
            for blob_sline_stem in ['front', 'left', 'right']:  # 'rear' is ignored for now. by FLAG_ohne_rear
                dp_blob_sling = dp_output_blob / blob_sline_stem
                dp_blob_sling.mkdir(parents=True, exist_ok=True)

                dp_blob_sling_imgs = dp_blob_sling / "images"
                dp_blob_sling_imgs.mkdir(parents=True, exist_ok=True)
                fps_blob_sling_images = []

                for fp_image in fps_blob_images:
                    if blob_sline_stem in tell_whether_it_belongs_to_the_sling(fp_image):
                        shutil.copy(fp_image, dp_blob_sling_imgs / fp_image.name)
                        fps_blob_sling_images.append(dp_blob_sling_imgs / fp_image.name)

                get_reconstructed_scene(
                    dp_output=dp_blob_sling,
                    model=model,
                    filelist=[fp.resolve().as_posix() for fp in fps_blob_sling_images],
                    silent=FLAG_silent,
                    skip_GLOMAP=FLAG_skip_GLOMAP
                )
                print(f"Time taken for blob {blob_idx}: {time() - start_time} seconds.")
