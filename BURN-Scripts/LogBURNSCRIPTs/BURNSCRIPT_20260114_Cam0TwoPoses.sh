#!/bin/bash
# BURNSCRIPT - Docker runner for Cam0 Two Poses Processing
# ==========================================================
# Generated: 20260114
# Script: BatchedDB
# Camera: cam0
# Poses: p+0_y+15_r+0, p-20_y+15_r+0
# Batch: 36, Stride: 12 (overlap: 24)
# 
# Runs batch processing for two camera pose directories in the m3rslam Docker container.
# Output folders will include pose information in their names for easy identification.
#
# Usage:
#   ./BURNSCRIPT_20260114_Cam0TwoPoses.sh

STARTER_SCRIPT="BURN-Scripts/starter_20260114_Cam0TwoPoses.py"

echo "========================================"
echo "BURNSCRIPT - Cam0 Two Poses Runner"
echo "========================================"
echo "Datetime: 20260114"
echo "Script: BatchedDB"
echo "Camera: cam0"
echo "Poses: p+0_y+15_r+0, p-20_y+15_r+0"
echo "Batch: 36, Stride: 12"
echo "Starter: $STARTER_SCRIPT"
echo ""

docker run --rm -it \
    --gpus=all \
    --shm-size=8g \
    -v D:/:/d_disk \
    m3rslam \
    bash -c "cd /d_disk/mast3r && /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 python $STARTER_SCRIPT"
