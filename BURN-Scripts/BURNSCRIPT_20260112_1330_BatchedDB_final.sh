#!/bin/bash
# BURNSCRIPT FINAL - Docker runner for SKRIPT_Batched_DB_Generator
# =================================================================
# Generated: 20260112_1330
# Script: BatchedDB
# Mode: FULL DATASET (all batches)
# 
# Runs the full batch processing in the m3rslam Docker container.
#
# Usage:
#   ./BURNSCRIPT_20260112_1330_BatchedDB_final.sh

STARTER_SCRIPT="BURN-Scripts/starter_20260112_1330_BatchedDB_final.py"

echo "========================================"
echo "BURNSCRIPT FINAL - Docker Runner"
echo "========================================"
echo "Datetime: 20260112_1330"
echo "Script: BatchedDB"
echo "Mode: FULL DATASET"
echo "Starter: $STARTER_SCRIPT"
echo ""

docker run --rm -it \
    --gpus=all \
    --shm-size=8g \
    -v D:/:/d_disk \
    m3rslam \
    bash -c "cd /d_disk/mast3r && /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 python $STARTER_SCRIPT"
