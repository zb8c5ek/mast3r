#!/bin/bash
# BURNSCRIPT - Docker runner for Two Camera Poses Processing (cam0)
# ===================================================================
# Generated: 20260114
# Script: BatchedDB
# Mode: Two poses (p+0_y+15_r+0 and p-20_y+15_r+0) - cam0
# Batch size: 36, overlap: 24 (stride: 12)
# 
# Runs batch processing for two camera pose directories in the m3rslam Docker container.
# Output folders will include pose information in their names for easy identification.
#
# Usage:
#   ./BURNSCRIPT_20260114_cam0_TwoPoses.sh

STARTER_SCRIPT="BURN-Scripts/starter_20260114_cam0_TwoPoses.py"

echo "========================================"
echo "BURNSCRIPT - Two Poses Docker Runner (cam0)"
echo "========================================"
echo "Datetime: 20260114"
echo "Script: BatchedDB"
echo "Poses: p+0_y+15_r+0, p-20_y+15_r+0"
echo "Batch: 36, Overlap: 24, Stride: 12"
echo "Starter: $STARTER_SCRIPT"
echo ""

docker run --rm -it \
    --gpus=all \
    --shm-size=8g \
    -v D:/:/d_disk \
    m3rslam \
    bash -c "cd /d_disk/mast3r && /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 python $STARTER_SCRIPT"
