# MOD_ResultsAnalyze
# Module for analyzing reconstruction results from COLMAP/GLOMAP
#
# For full workflow with HTML/JSON reports:
#   python _BURNSCRIPT_Result_Analyze/BURN_ReconstructionAnalyze.py <paths...> --best-only
#
# Core functions for programmatic use:

from .ESSN_ReconstructionAnalyze import (
    load_reconstruction_to_dict,
    get_reconstruction_stats,
    count_images_in_folder,
    find_images_folder,
)

from .kern_trajectory_analyze import (
    CameraPose,
    CameraTrajectory,
    parse_timestamp_from_filename,
    extract_trajectory_from_reconstruction,
    trajectories_to_arrays,
    compute_trajectory_stats,
    # 2D views and rig analysis
    prepare_trajectory_2d_views,
    analyze_rig_relative_poses,
    analyze_single_reconstruction,
    analyze_folder_hierarchical,
    save_analysis_json,
)

__all__ = [
    # Reconstruction analysis
    'load_reconstruction_to_dict',
    'get_reconstruction_stats',
    'count_images_in_folder',
    'find_images_folder',
    # Trajectory analysis
    'CameraPose',
    'CameraTrajectory',
    'parse_timestamp_from_filename',
    'extract_trajectory_from_reconstruction',
    'trajectories_to_arrays',
    'compute_trajectory_stats',
    # 2D views and rig analysis
    'prepare_trajectory_2d_views',
    'analyze_rig_relative_poses',
    'analyze_single_reconstruction',
    'analyze_folder_hierarchical',
    'save_analysis_json',
]
