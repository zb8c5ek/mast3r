"""
MOD_FeatureProcess - MASt3R Feature Matching & COLMAP DB Creation
=================================================================

High-Level API (for BURN scripts):
    - create_db_from_folder(): Image folder -> COLMAP database
    - create_db_from_folder_with_structure(): Pre-organized images -> DB (multi-cam)
    - run_pycolmap_mapping(): Run incremental mapping

Low-Level API:
    - essn_run_mast3r_matching(): Gin-configurable matching entry point
    - kern_get_im_matches(): Core match extraction from MASt3R predictions
    - kern_apply_matching_strategy(): Apply filtering strategy to confidences

Matching Strategies:
    ('conf_thres', float)  - Fixed confidence threshold
    ('ratio', float)       - Keep top X% of matches
    ('num_pts', int)       - Keep top N matches (default)
    [strategy1, strategy2] - Combined (AND logic)

Gin configs: _gconfs/essn_default.gin, essn_conf_thres.gin, essn_ratio.gin
"""
# High-level DB creation API (main entry for BURN scripts)
from .ESSN_CreateDBfromFolder import (
    create_db_from_folder,
    create_db_from_folder_with_structure,
    run_pycolmap_mapping,
)

# Matching API
from .ESSN_FeatureProcess import (
    essn_run_mast3r_matching,
    essn_probe_mast3r_matching,
    probe_mast3r_matching,
    run_mast3r_matching,  # backward-compatible alias
)

# Core functions
from .kern_feature_process import (
    kern_get_im_matches,
    kern_apply_matching_strategy,
    MatchingStrategy,
)

__all__ = [
    # High-level DB creation
    'create_db_from_folder',
    'create_db_from_folder_with_structure',
    'run_pycolmap_mapping',
    # Matching
    'essn_run_mast3r_matching',
    'essn_probe_mast3r_matching',
    'probe_mast3r_matching',
    'run_mast3r_matching',
    # Core
    'kern_get_im_matches',
    'kern_apply_matching_strategy',
    'MatchingStrategy',
]
