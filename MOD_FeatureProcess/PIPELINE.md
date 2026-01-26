# MOD_FeatureProcess Pipeline

MASt3R-based feature matching and COLMAP database creation.

**Note:** Runs in Docker where `D:\` is mounted as `/d_disk`.

## Architecture

```
BURN-Scripts/                     MOD_FeatureProcess/
┌─────────────────────┐           ┌──────────────────────────────────┐
│ BURN_template_*.py  │──────────▶│ create_db_from_folder()          │
│ (YAML config)       │           │ create_db_from_folder_with_*()   │
└─────────────────────┘           └──────────────┬───────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────────┐
                                  │ _run_matching_pipeline()          │
                                  │   • load_images → make_pairs      │
                                  │   • setup COLMAP DB               │
                                  │   • run_mast3r_matching()  ───────┼──▶ MASt3R inference
                                  │   • verify_matches                │
                                  └──────────────┬───────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────────┐
                                  │ run_pycolmap_mapping()            │
                                  │   (or async via Utils4BurnScript) │
                                  └──────────────────────────────────┘
```

## API Reference

### High-Level (for BURN scripts)

```python
from MOD_FeatureProcess import create_db_from_folder, run_pycolmap_mapping

# Single-camera folder
result = create_db_from_folder(
    dp_images=Path("path/to/images"),
    dp_output=Path("output/dir"),
    model=mast3r_model,
    matching_strategy=('num_pts', 1000),  # or [('conf_thres', 2.5), ('num_pts', 1500)]
    camera_model='PINHOLE',
    batch_size=16
)

# Multi-camera rig (pre-organized as images/cam0/, images/cam1/, ...)
result = create_db_from_folder_with_structure(
    root_path=output_dir,
    filelist_relpath=["images/cam0/001.jpg", "images/cam1/001.jpg", ...],
    dp_output=output_dir,
    model=model,
    share_intrinsics_by_subfolder=True,  # One camera per subfolder
    ...
)

# Run COLMAP mapping
map_result = run_pycolmap_mapping(
    database_path=result['database_path'],
    image_path=output_dir,
    output_path=output_dir / 'sparse',
    mapper_options={'min_num_matches': 15}
)
```

### Matching Strategies

| Strategy | Example | Description |
|----------|---------|-------------|
| `num_pts` | `('num_pts', 1000)` | Keep top N matches by confidence |
| `conf_thres` | `('conf_thres', 2.5)` | Fixed confidence threshold |
| `ratio` | `('ratio', 0.1)` | Keep top X% of matches |
| Combined | `[('conf_thres', 2.5), ('num_pts', 1500)]` | AND logic |

### Camera Models

`SIMPLE_PINHOLE`, `PINHOLE` (default), `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`

## File Structure

```
MOD_FeatureProcess/
├── __init__.py                 # Public API exports
├── ESSN_CreateDBfromFolder.py  # DB creation
├── ESSN_FeatureProcess.py      # Gin-configurable matching
├── kern_feature_process.py     # Core matching functions
├── _gconfs/                    # Gin config presets
└── PIPELINE.md                 # This doc

BURN-Scripts/
├── BURN_template_RopeCapGroup.py     # 1 cam, 1 pose → 1 DB
├── BURN_template_RopeCapMultiPose.py # 1 cam, N poses → 1 DB
├── BURN_template_RigBubble.py        # N cams → 1 DB (shared intrinsics)
├── Utils4BurnScript.py               # MappingJobManager (async w/ dashboard)
└── BURN_*.ps1                        # PowerShell wrappers

configs/
├── ropecap_group_20260123.yaml       # Individual camera config
├── ropecap_multipose_20260125.yaml   # Multi-pose config
└── ropecap_rigbubble_20260123.yaml   # Rig-bubble config
```

## Processing Modes

| Mode | Script | Use Case |
|------|--------|----------|
| Individual | `BURN_template_RopeCapGroup.py` | 1 camera, 1 pose → 1 DB |
| Multi-Pose | `BURN_template_RopeCapMultiPose.py` | 1 camera, N poses → 1 DB |
| Rig-Bubble | `BURN_template_RigBubble.py` | N cameras → 1 DB (shared intrinsics) |

### Multi-Pose Mode

Collects images from multiple pose folders for a single camera:

```yaml
# configs/ropecap_multipose_20260125.yaml
paths:
  base_path: "/d_disk/_DataBuffer/RopeCap0121/17/undistort_fov110_p30y20_20260121_165059"
  output: null  # Auto-generate

processing:
  target_poses:
    - "p+0_y+20_r+0"
    - "p+0_y+0_r+0"
  stride_frame: 1
  run_mapping: true
  async_mapping: true
  mapping_timeout: 1800

matching:
  type: num_pts
  value: 1000

mapper:
  camera_model: PINHOLE
  share_intrinsics_by_pose: false  # true = separate camera per pose
```

```
base_path/group_XXX/cam0/
├── p+0_y+20_r+0/  ─┐
└── p+0_y+0_r+0/   ─┴──→  single database.db
```

**Run:**
```bash
python BURN-Scripts/BURN_template_RopeCapMultiPose.py -c configs/ropecap_multipose_20260125.yaml
```

## Output Structure

```
<output_base>/
├── group_XXX/
│   ├── cam0/ (or bubble_00/)
│   │   ├── images/
│   │   ├── database.db
│   │   ├── pairs.txt
│   │   ├── cache/
│   │   └── sparse/0/  (after mapping)
│   └── ...
├── processing_results.json
├── report.html
├── _dashboard.html  (live mapping status)
└── config_used.yaml
```
