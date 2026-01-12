"""
MOD_FeatureProcess Module
Custom feature processing functions for MASt3R matching.

Structure:
    - kern_feature_process.py: Core functions (not gin configurable)
    - ESSN_FeatureProcess.py: High-level gin-configurable entry point

Gin configurable - load configuration from _gconfs/ folder:
    import gin
    gin.parse_config_file('MOD_FeatureProcess/_gconfs/essn_default.gin')
    
Available configurations:
    - essn_default.gin: Keep top 10000 matching points (default)
    - essn_conf_thres.gin: Fixed confidence threshold (1.001)
    - essn_ratio.gin: Keep top 10% of matches
"""
from .ESSN_FeatureProcess import (
    # Gin configurable entry point
    essn_run_mast3r_matching,
    # Backward compatible alias
    run_mast3r_matching,
)

from .kern_feature_process import (
    # Core functions (not gin configurable)
    kern_get_im_matches,
    kern_apply_matching_strategy,
    # Type alias
    MatchingStrategy,
)

__all__ = [
    # High-level gin configurable
    'essn_run_mast3r_matching',
    'run_mast3r_matching',
    # Core functions
    'kern_get_im_matches',
    'kern_apply_matching_strategy',
    # Type
    'MatchingStrategy',
]
