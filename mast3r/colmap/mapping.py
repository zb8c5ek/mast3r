# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Kapture-free COLMAP/GLOMAP mapping from MASt3R matches
# --------------------------------------------------------
#
# Drop-in replacement for the upstream mapping.py that removes the
# kapture dependency.  All functions that previously required kapture
# now operate on plain lists/dicts.
#
# Functions provided:
#   scene_prepare_images       - load & resize images into MASt3R dict format
#   remove_duplicates          - deduplicate symmetric image pairs
#   run_mast3r_matching        - full MASt3R → COLMAP DB pipeline (no kapture)
#   glomap_run_mapper          - run GLOMAP mapper subprocess
#   pycolmap_run_mapper        - run pycolmap incremental mapper
#   pycolmap_run_triangulator  - run pycolmap triangulator
#   ColmapDatabase             - lightweight sqlite3 COLMAP DB (no kapture)
# --------------------------------------------------------

import os
import os.path as path
import sqlite3
import struct
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import pycolmap
import torch
from tqdm import tqdm

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import (
    export_matches,
    get_im_matches,
)

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from dust3r.inference import inference


# ---------------------------------------------------------------------------
#  Lightweight COLMAP database (replaces kapture.converter.colmap.database)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
    prior_tx REAL, prior_ty REAL, prior_tz REAL
);
CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
);
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
);
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
);
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER DEFAULT 0,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB
);
"""


def _pair_id(id1, id2):
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * 2147483647 + id2


class ColmapDatabase:
    """Lightweight COLMAP sqlite3 database.

    API-compatible with kapture's COLMAPDatabase for the methods used by
    ``mast3r.colmap.database.export_images`` and ``export_matches``.
    """

    @staticmethod
    def connect(db_path):
        return ColmapDatabase(db_path)

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def commit(self):
        self.conn.commit()

    # -- cameras --
    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params_blob = np.asarray(params, dtype=np.float64).tobytes()
        pfl = 1 if prior_focal_length else 0
        if camera_id is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO cameras "
                "(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, model, width, height, params_blob, pfl),
            )
            return camera_id
        cursor = self.conn.execute(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, width, height, params_blob, pfl),
        )
        return cursor.lastrowid

    # -- images --
    def add_image(self, name, camera_id, prior_q=None, prior_t=None,
                  image_id=None):
        if prior_q is None:
            prior_q = np.zeros(4)
        if prior_t is None:
            prior_t = np.zeros(3)
        qw, qx, qy, qz = prior_q
        tx, ty, tz = prior_t
        if image_id is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO images "
                "(image_id, name, camera_id, "
                "prior_qw, prior_qx, prior_qy, prior_qz, "
                "prior_tx, prior_ty, prior_tz) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, name, camera_id,
                 float(qw), float(qx), float(qy), float(qz),
                 float(tx), float(ty), float(tz)),
            )
            return image_id
        cursor = self.conn.execute(
            "INSERT INTO images "
            "(name, camera_id, "
            "prior_qw, prior_qx, prior_qy, prior_qz, "
            "prior_tx, prior_ty, prior_tz) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, camera_id,
             float(qw), float(qx), float(qy), float(qz),
             float(tx), float(ty), float(tz)),
        )
        return cursor.lastrowid

    # -- keypoints --
    def add_keypoints(self, image_id, keypoints):
        kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        self.conn.execute(
            "INSERT OR REPLACE INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, kp.shape[0], kp.shape[1], kp.tobytes()),
        )

    # -- matches --
    def add_matches(self, image_id1, image_id2, matches):
        m = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
        pid = _pair_id(image_id1, image_id2)
        self.conn.execute(
            "INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?)",
            (pid, m.shape[0], m.shape[1], m.tobytes()),
        )

    # -- two-view geometry --
    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=None, E=None, H=None, config=2):
        m = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
        pid = _pair_id(image_id1, image_id2)
        F_blob = np.zeros((3, 3), dtype=np.float64).tobytes() if F is None else np.asarray(F).tobytes()
        E_blob = np.zeros((3, 3), dtype=np.float64).tobytes() if E is None else np.asarray(E).tobytes()
        H_blob = np.zeros((3, 3), dtype=np.float64).tobytes() if H is None else np.asarray(H).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, m.shape[0], m.shape[1], m.tobytes(),
             config, F_blob, E_blob, H_blob, None, None),
        )


# ---------------------------------------------------------------------------
#  Image loading (replaces dust3r_visloc.datasets.utils.get_resize_function)
# ---------------------------------------------------------------------------

def _get_resize_function(maxdim: int, patch_size: int, H: int, W: int):
    """Compute resize transform so the longest side is <= maxdim and
    both sides are divisible by patch_size.

    Returns:
        resize_fn:  callable(tensor) -> torch.Tensor (C,H',W')
        (H', W'):   output spatial dims
        to_orig:    3x3 np.float32 matrix mapping resized coords -> original
    """
    scale = maxdim / max(H, W)
    new_H = int(round(H * scale / patch_size)) * patch_size
    new_W = int(round(W * scale / patch_size)) * patch_size

    def resize_fn(img_tensor):
        # img_tensor is (C,H,W) from ImgNorm
        return torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

    to_orig = np.eye(3, dtype=np.float32)
    to_orig[0, 0] = W / new_W
    to_orig[1, 1] = H / new_H

    return resize_fn, (new_H, new_W), to_orig


def scene_prepare_images(
    root: str,
    maxdim: int,
    patch_size: int,
    image_paths: List[str],
) -> List[Dict]:
    """Load images and prepare them in the dict format expected by
    MASt3R inference and ``mast3r.colmap.database`` helpers.

    Each returned dict has keys:
        img, true_shape, to_orig, idx, instance, orig_shape
    """
    images = []
    for idx in tqdm(range(len(image_paths)), desc="Loading images"):
        rgb_image = PIL.Image.open(
            os.path.join(root, image_paths[idx])
        ).convert('RGB')

        W, H = rgb_image.size
        resize_fn, _, to_orig = _get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_fn(ImgNorm(rgb_image))

        images.append({
            'img': rgb_tensor.unsqueeze(0),
            'true_shape': np.int32([rgb_tensor.shape[1:]]),
            'to_orig': to_orig,
            'idx': idx,
            'instance': image_paths[idx],
            'orig_shape': np.int32([H, W]),
        })
    return images


# ---------------------------------------------------------------------------
#  Pair deduplication
# ---------------------------------------------------------------------------

def remove_duplicates(images, image_pairs):
    """Deduplicate symmetric image pairs.

    Args:
        images:      list of image dicts (indexed by int idx)
        image_pairs: list of ((idx_i, path_i), (idx_j, path_j))

    Returns:
        list of (images[i], images[j]) with each unordered pair once.
    """
    pairs_added = set()
    pairs = []
    for (i, _), (j, _) in image_pairs:
        lo, hi = min(i, j), max(i, j)
        if (lo, hi) in pairs_added:
            continue
        pairs_added.add((lo, hi))
        pairs.append((images[i], images[j]))
    return pairs


# ---------------------------------------------------------------------------
#  MASt3R matching -> COLMAP DB  (kapture-free)
# ---------------------------------------------------------------------------

def _export_images_to_db(db, images, image_paths, camera_model="PINHOLE"):
    """Populate a COLMAP database with cameras and images.

    Reimplements the logic of ``mast3r.colmap.database.export_images``
    using our lightweight ColmapDatabase API.

    Returns:
        (image_to_colmap, im_keypoints)
    """
    from scipy.spatial.transform import Rotation as R

    image_to_colmap = {}
    im_keypoints = {}

    for idx in range(len(image_paths)):
        im_keypoints[idx] = {}
        H, W = images[idx]["orig_shape"]

        # Default focal length (no prior)
        focal_x = focal_y = 1.2 * max(W, H)
        cx = W / 2.0
        cy = H / 2.0

        # Scale to resized coordinates
        focal_x = focal_x * images[idx]["to_orig"][0, 0]
        focal_y = focal_y * images[idx]["to_orig"][1, 1]

        if camera_model == "SIMPLE_PINHOLE":
            model_id = 0
            focal = (focal_x + focal_y) / 2.0
            params = np.asarray([focal, cx, cy], np.float64)
        elif camera_model == "PINHOLE":
            model_id = 1
            params = np.asarray([focal_x, focal_y, cx, cy], np.float64)
        elif camera_model == "SIMPLE_RADIAL":
            model_id = 2
            focal = (focal_x + focal_y) / 2.0
            params = np.asarray([focal, cx, cy, 0.0], np.float64)
        elif camera_model == "OPENCV":
            model_id = 4
            params = np.asarray(
                [focal_x, focal_y, cx, cy, 0.0, 0.0, 0.0, 0.0], np.float64)
        else:
            raise ValueError(f"invalid camera model {camera_model}")

        H, W = int(H), int(W)
        camid = db.add_camera(
            model_id, W, H, params, prior_focal_length=False)

        prior_t = np.zeros(3)
        prior_q = np.zeros(4)
        imid = db.add_image(
            image_paths[idx], camid, prior_q=prior_q, prior_t=prior_t)

        image_to_colmap[idx] = {
            'colmap_imid': imid,
            'colmap_camid': camid,
        }
    return image_to_colmap, im_keypoints


def run_mast3r_matching(
    model: AsymmetricMASt3R,
    maxdim: int,
    patch_size: int,
    device,
    image_paths: List[str],
    root_path: str,
    image_pairs: List[Tuple[str, str]],
    colmap_db,
    dense_matching: bool,
    pixel_tol: int,
    conf_thr: float,
    skip_geometric_verification: bool,
    min_len_track: int,
    shared_intrinsics: bool = True,
    camera_model: str = "PINHOLE",
):
    """Run MASt3R pairwise matching and export to a COLMAP database.

    This is the kapture-free replacement.  Instead of a kapture.Kapture
    object, it takes plain ``image_paths`` and ``image_pairs``.

    Args:
        model:           Loaded MASt3R model.
        maxdim:          Maximum image dimension for inference.
        patch_size:      ViT patch size (typically 16).
        device:          Torch device string.
        image_paths:     List of relative image paths.
        root_path:       Root directory containing the images.
        image_pairs:     List of (rel_path_a, rel_path_b) pairs.
        colmap_db:       Open COLMAP database (ColmapDatabase or compatible).
        dense_matching:  If True use dense matching, else sparse.
        pixel_tol:       Pixel tolerance for correspondence extraction.
        conf_thr:        Confidence threshold for match filtering.
        skip_geometric_verification: Skip two-view geometry export.
        min_len_track:   Minimum track length for keypoint export.
        shared_intrinsics: Share camera intrinsics across images.
        camera_model:    COLMAP camera model name.

    Returns:
        colmap_image_pairs: list of (path1, path2) pairs kept after export.
    """
    # Build path -> idx mapping
    image_path_to_idx = {ip: idx for idx, ip in enumerate(image_paths)}

    # Load and prepare images
    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)

    # Build image pairs as ((idx, path), (idx, path)) then deduplicate
    raw_pairs = [
        ((image_path_to_idx[p1], p1), (image_path_to_idx[p2], p2))
        for p1, p2 in image_pairs
        if p1 in image_path_to_idx and p2 in image_path_to_idx
    ]
    matching_pairs = remove_duplicates(images, raw_pairs)

    # Populate COLMAP DB with cameras & images
    image_to_colmap, im_keypoints = _export_images_to_db(
        colmap_db, images, image_paths,
        camera_model=camera_model,
    )

    # Run MASt3R inference in chunks and extract 2D-2D matches
    im_matches = {}
    batch_size = 4
    for chunk_start in tqdm(range(0, len(matching_pairs), batch_size),
                            desc="MASt3R matching"):
        pairs_chunk = matching_pairs[chunk_start:chunk_start + batch_size]

        with torch.no_grad():
            output = inference(
                pairs_chunk, model, device,
                batch_size=1, verbose=False,
            )
        pred1, pred2 = output['pred1'], output['pred2']

        chunk_matches = get_im_matches(
            pred1=pred1,
            pred2=pred2,
            pairs=pairs_chunk,
            image_to_colmap=image_to_colmap,
            im_keypoints=im_keypoints,
            conf_thr=conf_thr,
            is_sparse=not dense_matching,
            pixel_tol=pixel_tol,
            device=device,
        )
        im_matches.update(chunk_matches.items())

    # Filter matches, build tracks, export keypoints & matches to DB
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints,
        im_matches, min_len_track, skip_geometric_verification,
    )
    colmap_db.commit()

    return colmap_image_pairs


# ---------------------------------------------------------------------------
#  COLMAP / GLOMAP mapper wrappers
# ---------------------------------------------------------------------------

def glomap_run_mapper(
    glomap_bin: str,
    colmap_db_path: str,
    recon_path: str,
    image_root_path: str,
):
    """Run the GLOMAP mapper as a subprocess."""
    print("running mapping")
    args = [
        glomap_bin,
        'mapper',
        '--database_path', colmap_db_path,
        '--image_path', image_root_path,
        '--output_path', recon_path,
    ]
    proc = subprocess.Popen(args)
    proc.wait()

    if proc.returncode != 0:
        raise ValueError(
            f'\nSubprocess Error (Return code: {proc.returncode})')


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    """Run pycolmap incremental mapper."""
    print("running mapping")
    pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({
            'multiple_models': False,
            'extract_colors': True,
        }),
    )


def pycolmap_run_triangulator(
    colmap_db_path, prior_recon_path, recon_path, image_root_path,
):
    """Run pycolmap point triangulation."""
    print("running mapping")
    reconstruction = pycolmap.Reconstruction(prior_recon_path)
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        refine_intrinsics=False,
    )
