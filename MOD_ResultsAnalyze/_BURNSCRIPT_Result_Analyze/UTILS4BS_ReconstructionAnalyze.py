#!/usr/bin/env python
"""
UTILS4BS_ReconstructionAnalyze - Utilities for Reconstruction Analysis
======================================================================
Handles:
- Loading pycolmap reconstructions
- Extracting camera trajectories and statistics
- HTML report generation with interactive visualization
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class CameraTrajectory:
    """Single camera pose with intrinsics."""
    image_id: int
    image_name: str
    camera_id: int
    # Extrinsics: world-to-camera transform (4x4 matrix)
    cam_from_world: np.ndarray  # 4x4
    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    camera_model: str
    # Position in world coordinates
    position: np.ndarray  # 3D
    # Additional stats
    num_points2D: int = 0
    num_points3D_visible: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'image_id': self.image_id,
            'image_name': self.image_name,
            'camera_id': self.camera_id,
            'cam_from_world': self.cam_from_world.tolist(),
            'position': self.position.tolist(),
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'camera_model': self.camera_model,
            'num_points2D': self.num_points2D,
            'num_points3D_visible': self.num_points3D_visible,
        }


@dataclass
class ReconstructionStats:
    """Statistics for a reconstruction."""
    name: str
    path: str
    num_images_total: int
    num_images_registered: int
    num_cameras: int
    num_points3D: int
    # Coverage
    registration_ratio: float
    # 3D points stats
    mean_track_length: float
    mean_reproj_error: float
    # Bounding box
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_size: np.ndarray
    # Camera positions
    camera_positions: np.ndarray  # Nx3
    camera_span: float  # max distance between any two cameras
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'name': self.name,
            'path': self.path,
            'num_images_total': self.num_images_total,
            'num_images_registered': self.num_images_registered,
            'num_cameras': self.num_cameras,
            'num_points3D': self.num_points3D,
            'registration_ratio': self.registration_ratio,
            'mean_track_length': self.mean_track_length,
            'mean_reproj_error': self.mean_reproj_error,
            'bbox_min': self.bbox_min.tolist() if self.bbox_min is not None else None,
            'bbox_max': self.bbox_max.tolist() if self.bbox_max is not None else None,
            'bbox_size': self.bbox_size.tolist() if self.bbox_size is not None else None,
            'camera_span': self.camera_span,
        }


# =============================================================================
# CORE LOADING FUNCTIONS
# =============================================================================
def find_all_reconstructions(base_path: Path) -> List[Path]:
    """
    Recursively find all reconstruction folders under a base path.
    
    Looks for folders containing cameras.bin/images.bin or numbered folders
    inside 'sparse' directories.
    
    Args:
        base_path: Root path to search
    
    Returns:
        List of paths to reconstruction folders
    """
    base_path = Path(base_path)
    recon_paths = []
    
    # Check if this is already a reconstruction folder
    if (base_path / 'cameras.bin').exists() and (base_path / 'images.bin').exists():
        recon_paths.append(base_path)
        return recon_paths
    
    # Check for sparse/N folders
    sparse_dir = base_path / 'sparse'
    if sparse_dir.exists():
        for item in sparse_dir.iterdir():
            if item.is_dir() and (item / 'cameras.bin').exists():
                recon_paths.append(item)
    
    # Check for direct numbered folders (0, 1, 2, ...)
    for item in base_path.iterdir():
        if item.is_dir() and item.name.isdigit() and (item / 'cameras.bin').exists():
            recon_paths.append(item)
    
    # Recursively search subdirectories (e.g., group_XXX/camX/sparse/0)
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in ('sparse', 'cache', 'images'):
            # Don't recurse into numbered model folders we already found
            if not (item.name.isdigit() and (item / 'cameras.bin').exists()):
                recon_paths.extend(find_all_reconstructions(item))
    
    return recon_paths


def _count_images_in_folder(folder_path: Path) -> int:
    """Count image files in a folder."""
    if not folder_path.exists():
        return 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    count = 0
    for f in folder_path.iterdir():
        if f.is_file() and f.suffix.lower() in image_extensions:
            count += 1
    return count


def _find_images_folder(recon_path: Path) -> Tuple[Path, int]:
    """
    Find the images folder relative to a reconstruction path.
    
    Structure is always:
        parent_folder/
        ├── sparse/
        │   ├── 0/   <- recon_path points here
        │   ├── 1/
        │   └── ...
        └── images/  <- we want this
    
    So from sparse/N, images is at parent.parent / 'images'
    
    Returns (images_folder_path, image_count).
    """
    # From sparse/N -> go up twice to parent_folder, then into images
    # recon_path = .../sparse/0
    # recon_path.parent = .../sparse
    # recon_path.parent.parent = .../parent_folder
    images_folder = recon_path.parent.parent / 'images'
    
    if images_folder.exists() and images_folder.is_dir():
        count = _count_images_in_folder(images_folder)
        return images_folder, count
    
    return None, 0


def load_reconstruction(recon_path: Path, name_override: str = None) -> Tuple[Dict[int, CameraTrajectory], ReconstructionStats]:
    """
    Load a COLMAP/GLOMAP reconstruction and extract camera trajectories.
    
    Args:
        recon_path: Path to reconstruction folder (containing cameras.bin, images.bin, points3D.bin)
        name_override: Optional name to use instead of path-derived name
    
    Returns:
        trajectories: Dict mapping image_id to CameraTrajectory
        stats: ReconstructionStats with overall statistics
    """
    import pycolmap
    
    recon_path = Path(recon_path)
    
    # Handle path: might be direct recon folder or parent with "0" subfolder
    if not (recon_path / 'cameras.bin').exists():
        if (recon_path / '0').exists() and (recon_path / '0' / 'cameras.bin').exists():
            recon_path = recon_path / '0'
        elif (recon_path / 'sparse' / '0').exists():
            recon_path = recon_path / 'sparse' / '0'
    
    # Load reconstruction
    recon = pycolmap.Reconstruction(str(recon_path))
    
    # Find actual images folder and count
    images_folder, total_images_in_folder = _find_images_folder(recon_path)
    
    trajectories = {}
    camera_positions = []
    
    num_reg_images = recon.num_reg_images()
    for idx, (img_id, image) in enumerate(recon.images.items()):
        camera = recon.cameras[image.camera_id]
        
        # Get transform matrix - handle different pycolmap versions
        # In newer pycolmap, cam_from_world is a method returning Rigid3d
        if callable(image.cam_from_world):
            rigid = image.cam_from_world()
            T = rigid.matrix() if callable(rigid.matrix) else rigid.matrix
        elif hasattr(image.cam_from_world, 'matrix'):
            T = image.cam_from_world.matrix() if callable(image.cam_from_world.matrix) else image.cam_from_world.matrix
        else:
            T = image.cam_from_world
        
        # Camera position in world coordinates: -R^T @ t
        R = T[:3, :3]
        t = T[:3, 3]
        position = -R.T @ t
        camera_positions.append(position)
        
        # Build trajectory entry
        traj = CameraTrajectory(
            image_id=img_id,
            image_name=image.name,
            camera_id=image.camera_id,
            cam_from_world=T,
            position=position,
            fx=camera.focal_length_x,
            fy=camera.focal_length_y,
            cx=camera.principal_point_x,
            cy=camera.principal_point_y,
            width=camera.width,
            height=camera.height,
            camera_model=str(camera.model) if hasattr(camera, 'model') else 'unknown',
            num_points2D=len(image.points2D) if hasattr(image, 'points2D') else 0,
        )
        trajectories[img_id] = traj
        
        if idx + 1 >= num_reg_images:
            break
    
    # Compute 3D points statistics
    points3D_xyz = []
    track_lengths = []
    reproj_errors = []
    
    num_points3D = recon.num_points3D()
    for idx, (pt_id, pt3d) in enumerate(recon.points3D.items()):
        points3D_xyz.append(pt3d.xyz)
        track_lengths.append(pt3d.track.length() if hasattr(pt3d.track, 'length') else len(pt3d.track))
        reproj_errors.append(pt3d.error)
        if idx + 1 >= num_points3D:
            break
    
    points3D_xyz = np.array(points3D_xyz) if points3D_xyz else np.zeros((0, 3))
    camera_positions = np.array(camera_positions) if camera_positions else np.zeros((0, 3))
    
    # Bounding box
    if len(points3D_xyz) > 0:
        bbox_min = points3D_xyz.min(axis=0)
        bbox_max = points3D_xyz.max(axis=0)
        bbox_size = bbox_max - bbox_min
    else:
        bbox_min = bbox_max = bbox_size = np.zeros(3)
    
    # Camera span
    if len(camera_positions) > 1:
        from scipy.spatial.distance import pdist
        camera_span = pdist(camera_positions).max()
    else:
        camera_span = 0.0
    
    # Build stats - derive a meaningful name from path
    if name_override:
        recon_name = name_override
    else:
        # Try to build a meaningful name from the path hierarchy
        parts = recon_path.parts
        # Look for group_XXX, camX, or model number patterns
        meaningful_parts = []
        for p in parts[-5:]:  # Check last 5 path components
            if p.startswith('group_') or p.startswith('cam') or p.startswith('bubble') or p.isdigit() or p == 'sparse':
                meaningful_parts.append(p)
        recon_name = '/'.join(meaningful_parts) if meaningful_parts else recon_path.name
    
    # Use actual image count from folder if available, otherwise fall back to reconstruction count
    actual_total_images = total_images_in_folder if total_images_in_folder > 0 else len(recon.images)
    
    stats = ReconstructionStats(
        name=recon_name,
        path=str(recon_path),
        num_images_total=actual_total_images,
        num_images_registered=num_reg_images,
        num_cameras=len(recon.cameras),
        num_points3D=num_points3D,
        registration_ratio=num_reg_images / max(1, actual_total_images),
        mean_track_length=np.mean(track_lengths) if track_lengths else 0,
        mean_reproj_error=np.mean(reproj_errors) if reproj_errors else 0,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size=bbox_size,
        camera_positions=camera_positions,
        camera_span=camera_span,
    )
    
    return trajectories, stats


def load_multiple_reconstructions(base_paths: List[Path], recursive: bool = True) -> Dict[str, Tuple[Dict, ReconstructionStats]]:
    """
    Load multiple reconstructions for comparison.
    
    Args:
        base_paths: List of base paths to search for reconstructions
        recursive: If True, recursively find all reconstructions under each path
    
    Returns:
        Dict mapping reconstruction name to (trajectories, stats)
    """
    results = {}
    
    # Collect all reconstruction paths
    all_recon_paths = []
    for base_path in base_paths:
        base_path = Path(base_path)
        if recursive:
            found = find_all_reconstructions(base_path)
            if found:
                print(f"  Found {len(found)} reconstruction(s) under {base_path.name}")
                all_recon_paths.extend(found)
            else:
                # Try loading directly if no sub-reconstructions found
                all_recon_paths.append(base_path)
        else:
            all_recon_paths.append(base_path)
    
    print(f"  Total reconstructions to load: {len(all_recon_paths)}")
    
    # Load each reconstruction
    for path in all_recon_paths:
        path = Path(path)
        try:
            trajectories, stats = load_reconstruction(path)
            # Use the stats name (which is derived from path hierarchy)
            name = stats.name
            # Handle duplicate names by appending path info
            if name in results:
                name = f"{name}_{path.parent.name}"
            results[name] = (trajectories, stats)
            print(f"    [OK] {name}: {stats.num_images_registered} images, {stats.num_points3D} points")
        except Exception as e:
            name = str(path)[-50:]  # Truncated path for error display
            print(f"    [!!] Failed {name}: {e}")
    
    return results


# =============================================================================
# BEST MODEL SELECTION
# =============================================================================
def select_best_models(
    reconstructions: Dict[str, Tuple[Dict, ReconstructionStats]]
) -> Dict[str, Tuple[Dict, ReconstructionStats]]:
    """
    Select the best model for each reconstruction location (parent folder).
    
    When multiple models exist in the same sparse folder (e.g., sparse/0, sparse/1),
    this function selects the best one based on:
    1. If any model has 100% registration, prefer that
    2. Otherwise, select the one with most registered images
    
    Args:
        reconstructions: Dict from load_multiple_reconstructions
    
    Returns:
        Dict with only the best model for each parent folder
    """
    # Group reconstructions by their parent folder (the folder containing sparse/)
    # e.g., group_000/cam0/sparse/0 -> parent is group_000/cam0
    groups: Dict[str, List[Tuple[str, Dict, ReconstructionStats]]] = {}
    
    for name, (trajs, stats) in reconstructions.items():
        if stats is None:
            continue
        # Get parent path (folder containing sparse/)
        # From path like .../group_000/cam0/sparse/0, get .../group_000/cam0
        recon_path = Path(stats.path)
        parent_key = str(recon_path.parent.parent)  # Go up from sparse/N to sparse to parent
        
        if parent_key not in groups:
            groups[parent_key] = []
        groups[parent_key].append((name, trajs, stats))
    
    # Select best model from each group
    best_models = {}
    
    for parent_key, models in groups.items():
        if len(models) == 1:
            # Only one model, use it
            name, trajs, stats = models[0]
            best_models[name] = (trajs, stats)
        else:
            # Multiple models - select best
            # First, check for 100% registration
            full_reg = [m for m in models if m[2].registration_ratio >= 0.999]
            
            if full_reg:
                # Pick the one with most 3D points among 100% registered
                best = max(full_reg, key=lambda m: m[2].num_points3D)
            else:
                # Pick the one with most registered images
                best = max(models, key=lambda m: m[2].num_images_registered)
            
            name, trajs, stats = best
            best_models[name] = (trajs, stats)
    
    return best_models


# =============================================================================
# JSON EXPORT
# =============================================================================
def export_analysis_json(
    reconstructions: Dict[str, Tuple[Dict, ReconstructionStats]],
    output_path: Path
) -> Path:
    """
    Export reconstruction analysis to a JSON file.
    
    Args:
        reconstructions: Dict from load_multiple_reconstructions
        output_path: Where to save the JSON file
    
    Returns:
        Path to the generated JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'summary': {
            'total_reconstructions': len(reconstructions),
            'total_registered_images': sum(
                s.num_images_registered for _, (_, s) in reconstructions.items() if s
            ),
            'total_3d_points': sum(
                s.num_points3D for _, (_, s) in reconstructions.items() if s
            ),
        },
        'reconstructions': []
    }
    
    for name, (trajs, stats) in reconstructions.items():
        if stats is not None:
            data['reconstructions'].append(stats.to_dict())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON analysis exported to: {output_path}")
    return output_path


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================
def generate_html_report(
    reconstructions: Dict[str, Tuple[Dict, ReconstructionStats]],
    output_path: Path,
    title: str = "Reconstruction Analysis Report"
) -> Path:
    """
    Generate an HTML report comparing reconstructions.
    
    Args:
        reconstructions: Dict from load_multiple_reconstructions
        output_path: Where to save the HTML file
        title: Report title
    
    Returns:
        Path to the generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for the report
    stats_list = []
    for name, (trajs, stats) in reconstructions.items():
        if stats is not None:
            stats_list.append(stats.to_dict())
    
    stats_json = json.dumps(stats_list, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --border: #30363d;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%);
            border-bottom: 1px solid var(--border);
            padding: 2rem 3rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .header h1 {{
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--accent);
            letter-spacing: -0.5px;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 3rem;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.2rem;
        }}
        
        .summary-card .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-top: 0.3rem;
        }}
        
        .summary-card.success .value {{ color: var(--accent-green); }}
        .summary-card.warning .value {{ color: var(--accent-orange); }}
        .summary-card.error .value {{ color: var(--accent-red); }}
        
        .section {{
            margin-bottom: 2rem;
        }}
        
        .section-title {{
            font-size: 1.1rem;
            color: var(--text-primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: var(--bg-tertiary);
            color: var(--accent);
            font-weight: 600;
            text-align: left;
            padding: 1rem;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 0.9rem 1rem;
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .metric {{
            font-variant-numeric: tabular-nums;
        }}
        
        .metric.good {{ color: var(--accent-green); }}
        .metric.medium {{ color: var(--accent-orange); }}
        .metric.bad {{ color: var(--accent-red); }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: var(--accent-green);
            transition: width 0.3s ease;
        }}
        
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        .bar-chart {{
            display: flex;
            align-items: flex-end;
            height: 200px;
            gap: 1rem;
            padding: 1rem 0;
        }}
        
        .bar-group {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .bar {{
            width: 100%;
            max-width: 60px;
            background: linear-gradient(180deg, var(--accent) 0%, #1a3a5c 100%);
            border-radius: 4px 4px 0 0;
            transition: height 0.5s ease;
        }}
        
        .bar-label {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-align: center;
            max-width: 80px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .bar-value {{
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .timestamp {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            padding: 2rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge.success {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .badge.warning {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-orange); }}
        .badge.error {{ background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 {title}</h1>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <div class="summary-grid" id="summaryGrid"></div>
        
        <div class="section">
            <h2 class="section-title">📈 Registration Comparison</h2>
            <div class="chart-container">
                <div class="bar-chart" id="barChart"></div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📋 Detailed Statistics</h2>
            <table id="statsTable">
                <thead>
                    <tr>
                        <th>Reconstruction</th>
                        <th>Images</th>
                        <th>Registration</th>
                        <th>3D Points</th>
                        <th>Cameras</th>
                        <th>Track Length</th>
                        <th>Reproj Error</th>
                        <th>Camera Span</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <div class="section">
            <h2 class="section-title">📐 Bounding Box Info</h2>
            <table id="bboxTable">
                <thead>
                    <tr>
                        <th>Reconstruction</th>
                        <th>Min X</th>
                        <th>Min Y</th>
                        <th>Min Z</th>
                        <th>Size X</th>
                        <th>Size Y</th>
                        <th>Size Z</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    
    <div class="timestamp">
        Report generated by UTILS4BS_ReconstructionAnalyze
    </div>
    
    <script>
        const stats = {stats_json};
        
        // Summary cards
        const summaryGrid = document.getElementById('summaryGrid');
        const totalImages = stats.reduce((sum, s) => sum + s.num_images_registered, 0);
        const totalPoints = stats.reduce((sum, s) => sum + s.num_points3D, 0);
        const avgRatio = stats.length > 0 
            ? (stats.reduce((sum, s) => sum + s.registration_ratio, 0) / stats.length * 100).toFixed(1)
            : 0;
        
        summaryGrid.innerHTML = `
            <div class="summary-card">
                <div class="label">Reconstructions</div>
                <div class="value">${{stats.length}}</div>
            </div>
            <div class="summary-card success">
                <div class="label">Total Registered</div>
                <div class="value">${{totalImages.toLocaleString()}}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total 3D Points</div>
                <div class="value">${{totalPoints.toLocaleString()}}</div>
            </div>
            <div class="summary-card ${{avgRatio > 80 ? 'success' : avgRatio > 50 ? 'warning' : 'error'}}">
                <div class="label">Avg Registration</div>
                <div class="value">${{avgRatio}}%</div>
            </div>
        `;
        
        // Bar chart
        const barChart = document.getElementById('barChart');
        const maxImages = Math.max(...stats.map(s => s.num_images_registered), 1);
        
        barChart.innerHTML = stats.map(s => {{
            const height = (s.num_images_registered / maxImages * 150) + 20;
            return `
                <div class="bar-group">
                    <div class="bar-value">${{s.num_images_registered}}</div>
                    <div class="bar" style="height: ${{height}}px;"></div>
                    <div class="bar-label" title="${{s.name}}">${{s.name}}</div>
                </div>
            `;
        }}).join('');
        
        // Stats table
        const tbody = document.querySelector('#statsTable tbody');
        tbody.innerHTML = stats.map(s => {{
            const ratio = (s.registration_ratio * 100).toFixed(1);
            const ratioClass = ratio > 80 ? 'good' : ratio > 50 ? 'medium' : 'bad';
            const errorClass = s.mean_reproj_error < 1 ? 'good' : s.mean_reproj_error < 2 ? 'medium' : 'bad';
            
            return `
                <tr>
                    <td>${{s.name}}</td>
                    <td class="metric">${{s.num_images_registered}} / ${{s.num_images_total}}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${{ratio}}%;"></div>
                        </div>
                        <span class="metric ${{ratioClass}}">${{ratio}}%</span>
                    </td>
                    <td class="metric">${{s.num_points3D.toLocaleString()}}</td>
                    <td class="metric">${{s.num_cameras}}</td>
                    <td class="metric">${{s.mean_track_length.toFixed(2)}}</td>
                    <td class="metric ${{errorClass}}">${{s.mean_reproj_error.toFixed(3)}} px</td>
                    <td class="metric">${{s.camera_span.toFixed(2)}} m</td>
                </tr>
            `;
        }}).join('');
        
        // Bbox table
        const bboxBody = document.querySelector('#bboxTable tbody');
        bboxBody.innerHTML = stats.map(s => {{
            const fmt = (v) => v !== null && v !== undefined ? v.toFixed(3) : '-';
            return `
                <tr>
                    <td>${{s.name}}</td>
                    <td class="metric">${{s.bbox_min ? fmt(s.bbox_min[0]) : '-'}}</td>
                    <td class="metric">${{s.bbox_min ? fmt(s.bbox_min[1]) : '-'}}</td>
                    <td class="metric">${{s.bbox_min ? fmt(s.bbox_min[2]) : '-'}}</td>
                    <td class="metric">${{s.bbox_size ? fmt(s.bbox_size[0]) : '-'}}</td>
                    <td class="metric">${{s.bbox_size ? fmt(s.bbox_size[1]) : '-'}}</td>
                    <td class="metric">${{s.bbox_size ? fmt(s.bbox_size[2]) : '-'}}</td>
                </tr>
            `;
        }}).join('');
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report saved to: {output_path}")
    return output_path


def export_trajectories_json(
    trajectories: Dict[int, CameraTrajectory],
    output_path: Path
) -> Path:
    """Export trajectories to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {img_id: traj.to_dict() for img_id, traj in trajectories.items()}
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Trajectories exported to: {output_path}")
    return output_path
