#!/bin/bash
# Burnscript - Docker runner for SKRIPT_Batched_DB_Generator
# ==========================================================
# Generated: 20260112_1330
# Script: BatchedDB
# 
# Runs the batch processing in the m3rslam Docker container.
#
# Usage:
#   ./burnscript_20260112_1330_BatchedDB_starter.sh

STARTER_SCRIPT="BURN-Scripts/starter_20260112_1330_BatchedDB_test1batch.py"

echo "========================================"
echo "BURNSCRIPT - Docker Runner"
echo "========================================"
echo "Datetime: 20260112_1330"
echo "Script: BatchedDB"
echo "Starter: $STARTER_SCRIPT"
echo ""

docker run --rm -it \
    --gpus=all \
    --shm-size=8g \
    -v D:/:/d_disk \
    m3rslam \
    bash -c "cd /d_disk/mast3r && /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 python $STARTER_SCRIPT"
