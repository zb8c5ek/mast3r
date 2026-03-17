"""
Kern Feature Processing Module
Core functions for MASt3R feature matching.
These are the low-level functions that do the actual work.
NOT gin configurable - parameters are passed from higher-level functions.
"""
import copy
from typing import List, Tuple, Union, Dict

import numpy as np

from mast3r.fast_nn import extract_correspondences_nonsym, bruteforce_reciprocal_nns
from mast3r.colmap.database import convert_im_matches_pairs

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


# Type alias for matching strategy
# Single strategy: ('conf_thres', float) or ('ratio', float) or ('num_pts', int)
# Combined strategies: [('conf_thres', 2.5), ('num_pts', 1500)] - uses intersection (AND)
MatchingStrategy = Union[Tuple[str, Union[float, int]], List[Tuple[str, Union[float, int]]]]


def kern_apply_matching_strategy(conf: np.ndarray, matching_strategy: MatchingStrategy) -> Tuple[np.ndarray, Union[float, List[float]]]:
    """
    Apply matching strategy to get a boolean mask for filtering matches.
    Supports single or combined strategies (combined = intersection/AND).
    
    Args:
        conf: Array of confidence values for all potential matches
        matching_strategy: Single strategy tuple OR list of strategy tuples:
            - ('conf_thres', float): Fixed confidence threshold
            - ('ratio', float): Keep top X% of matches (e.g., 0.1 for top 10%)
            - ('num_pts', int): Keep top N matches by confidence
            - [('conf_thres', 2.5), ('num_pts', 1500)]: Combined (AND logic)
            
    Returns:
        Tuple of (mask, thresholds):
            - mask: Boolean mask array where True indicates matches to keep
            - thresholds: Single threshold value or list of thresholds for combined strategies
    """
    # Handle combined strategies (list of strategies)
    if isinstance(matching_strategy, list):
        if len(matching_strategy) == 0:
            return np.ones(len(conf), dtype=bool), []
        
        # Apply each strategy and collect masks and thresholds
        masks = []
        thresholds = []
        for strategy in matching_strategy:
            mask, thr = kern_apply_matching_strategy(conf, strategy)
            masks.append(mask)
            thresholds.append(thr)
        
        # Combine masks with AND (intersection)
        combined_mask = np.logical_and.reduce(masks)
        return combined_mask, thresholds
    
    # Handle single strategy (original logic)
    strategy_type, strategy_value = matching_strategy
    
    if strategy_type == 'conf_thres':
        # Original behavior: fixed confidence threshold
        threshold = float(strategy_value)
        return conf >= threshold, threshold
    
    elif strategy_type == 'ratio':
        # Keep top X% of matches
        if len(conf) == 0:
            return np.array([], dtype=bool), 0.0
        ratio = float(strategy_value)
        assert 0.0 < ratio <= 1.0, f"Ratio must be in (0, 1], got {ratio}"
        # Find the threshold that keeps top ratio% of matches
        threshold = np.percentile(conf, (1.0 - ratio) * 100)
        return conf >= threshold, float(threshold)
    
    elif strategy_type == 'num_pts':
        # Keep top N matches
        if len(conf) == 0:
            return np.array([], dtype=bool), 0.0
        num_pts = int(strategy_value)
        if len(conf) <= num_pts:
            return np.ones(len(conf), dtype=bool), float(conf.min()) if len(conf) > 0 else 0.0
        # Find the threshold that keeps top N matches
        threshold = np.partition(conf, -num_pts)[-num_pts]
        return conf >= threshold, float(threshold)
    
    else:
        raise ValueError(f"Unknown matching strategy: {strategy_type}. "
                        f"Supported: 'conf_thres', 'ratio', 'num_pts'")


def kern_get_im_matches(pred1, pred2, pairs, image_to_colmap, im_keypoints, 
                        matching_strategy: MatchingStrategy,
                        is_sparse: bool = True, 
                        subsample: int = 8, 
                        pixel_tol: int = 0, 
                        viz: bool = False, 
                        device: str = 'cuda',
                        collect_im_matches: bool = True):
    """
    Extract image matches from MASt3R predictions.
    
    Core function for match extraction - not gin configurable.
    Parameters are passed from higher-level functions.
    
    Args:
        pred1: Predictions for first image in each pair
        pred2: Predictions for second image in each pair
        pairs: List of image pairs
        image_to_colmap: Mapping from image index to COLMAP IDs
        im_keypoints: Dictionary to store keypoints per image
        matching_strategy: Tuple specifying the filtering strategy
        is_sparse: Whether to use sparse matching (default: True)
        subsample: Subsampling factor (default: 8)
        pixel_tol: Pixel tolerance (default: 0)
        viz: Whether to visualize matches (default: False)
        device: Device to run on (default: 'cuda')
        collect_im_matches: Whether to convert/store COLMAP-ready matches.
            Set to False for fast probe passes that only need match counts.
        
    Returns:
        Tuple of (im_matches, match_stats):
            - im_matches: Dictionary of matches per image pair
            - match_stats: Dict with 'thresholds' list and 'num_matches' list
    """
    im_matches = {}
    match_stats = {'thresholds': [], 'num_matches': [], 'total_candidates': []}
    
    for i in range(len(pred1['pts3d'])):
        imidx0 = pairs[i][0]['idx']
        imidx1 = pairs[i][1]['idx']
        if 'desc' in pred1:  # mast3r
            descs = [pred1['desc'][i], pred2['desc'][i]]
            confidences = [pred1['desc_conf'][i], pred2['desc_conf'][i]]
            desc_dim = descs[0].shape[-1]

            if is_sparse:
                corres = extract_correspondences_nonsym(descs[0], descs[1], confidences[0], confidences[1],
                                                        device=device, subsample=subsample, pixel_tol=pixel_tol)
                conf = corres[2].cpu().numpy()
                mask, threshold = kern_apply_matching_strategy(conf, matching_strategy)
                matches_im0 = corres[0].cpu().numpy()[mask]
                matches_im1 = corres[1].cpu().numpy()[mask]
                
                # Track stats for caller
                match_stats['thresholds'].append(threshold)
                match_stats['num_matches'].append(int(mask.sum()))
                match_stats['total_candidates'].append(len(conf))
            else:
                # For dense matching, we need a threshold for initial filtering
                strategy_type, strategy_value = matching_strategy
                if strategy_type == 'conf_thres':
                    dense_conf_thr = strategy_value
                else:
                    dense_conf_thr = 1.0  # Low threshold for initial dense filtering
                
                confidence_masks = [confidences[0] >= dense_conf_thr, 
                                    confidences[1] >= dense_conf_thr]
                pts2d_list, desc_list = [], []
                for j in range(2):
                    conf_j = confidence_masks[j].cpu().numpy().flatten()
                    true_shape_j = pairs[i][j]['true_shape'][0]
                    pts2d_j = xy_grid(
                        true_shape_j[1], true_shape_j[0]).reshape(-1, 2)[conf_j]
                    desc_j = descs[j].detach().cpu(
                    ).numpy().reshape(-1, desc_dim)[conf_j]
                    pts2d_list.append(pts2d_j)
                    desc_list.append(desc_j)
                if len(desc_list[0]) == 0 or len(desc_list[1]) == 0:
                    continue

                nn0, nn1 = bruteforce_reciprocal_nns(desc_list[0], desc_list[1],
                                                     device=device, dist='dot', block_size=2**13)
                reciprocal_in_P0 = (nn1[nn0] == np.arange(len(nn0)))

                matches_im1 = pts2d_list[1][nn0][reciprocal_in_P0]
                matches_im0 = pts2d_list[0][reciprocal_in_P0]
        else:
            pts3d = [pred1['pts3d'][i], pred2['pts3d_in_other_view'][i]]
            confidences = [pred1['conf'][i], pred2['conf'][i]]

            if is_sparse:
                corres = extract_correspondences_nonsym(pts3d[0], pts3d[1], confidences[0], confidences[1],
                                                        device=device, subsample=subsample, pixel_tol=pixel_tol,
                                                        ptmap_key='3d')
                conf = corres[2].cpu().numpy()
                mask, threshold = kern_apply_matching_strategy(conf, matching_strategy)
                matches_im0 = corres[0].cpu().numpy()[mask]
                matches_im1 = corres[1].cpu().numpy()[mask]
                
                # Track stats for caller
                match_stats['thresholds'].append(threshold)
                match_stats['num_matches'].append(int(mask.sum()))
                match_stats['total_candidates'].append(len(conf))
            else:
                # For dense matching, use threshold for initial filtering
                strategy_type, strategy_value = matching_strategy
                if strategy_type == 'conf_thres':
                    dense_conf_thr = strategy_value
                else:
                    dense_conf_thr = 1.0
                confidence_masks = [confidences[0] >= dense_conf_thr,
                                    confidences[1] >= dense_conf_thr]
                # find 2D-2D matches between the two images
                pts2d_list, pts3d_list = [], []
                for j in range(2):
                    conf_j = confidence_masks[j].cpu().numpy().flatten()
                    true_shape_j = pairs[i][j]['true_shape'][0]
                    pts2d_j = xy_grid(true_shape_j[1], true_shape_j[0]).reshape(-1, 2)[conf_j]
                    pts3d_j = pts3d[j].detach().cpu().numpy().reshape(-1, 3)[conf_j]
                    pts2d_list.append(pts2d_j)
                    pts3d_list.append(pts3d_j)

                PQ, PM = pts3d_list[0], pts3d_list[1]
                if len(PQ) == 0 or len(PM) == 0:
                    continue
                reciprocal_in_PM, nnM_in_PQ, num_matches = find_reciprocal_matches(
                    PQ, PM)

                matches_im1 = pts2d_list[1][reciprocal_in_PM]
                matches_im0 = pts2d_list[0][nnM_in_PQ][reciprocal_in_PM]

        if len(matches_im0) == 0:
            continue
        if collect_im_matches:
            imidx0, imidx1, colmap_matches = convert_im_matches_pairs(pairs[i][0], pairs[i][1],
                                                                      image_to_colmap, im_keypoints,
                                                                      matches_im0, matches_im1, viz)
            im_matches[(imidx0, imidx1)] = colmap_matches
    return im_matches, match_stats
