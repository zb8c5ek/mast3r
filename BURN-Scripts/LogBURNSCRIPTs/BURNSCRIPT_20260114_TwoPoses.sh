#!/bin/bash
# BURNSCRIPT - Docker runner for Two Camera Poses Processing
# ===========================================================
# Generated: 20260114
# Script: BatchedDB
# Mode: Two poses (p-15_y-10_r+0 and p-5_y+0_r+0)
# 
# Runs batch processing for two camera pose directories in the m3rslam Docker container.
# Output folders will include pose information in their names for easy identification.
#
# Usage:
#   ./BURNSCRIPT_20260114_TwoPoses.sh

STARTER_SCRIPT="BURN-Scripts/starter_20260114_TwoPoses.py"

echo "========================================"
echo "BURNSCRIPT - Two Poses Docker Runner"
echo "========================================"
echo "Datetime: 20260114"
echo "Script: BatchedDB"
echo "Poses: p-15_y-10_r+0, p-5_y+0_r+0"
echo "Starter: $STARTER_SCRIPT"
echo ""

docker run --rm -it \
    --gpus=all \
    --shm-size=8g \
    -v D:/:/d_disk \
    m3rslam \
    bash -c "cd /d_disk/mast3r && /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 python $STARTER_SCRIPT"
