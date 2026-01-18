#!/bin/bash
# =============================================================================
# BURN SCRIPT - undistorted_fov110_center All Groups
# =============================================================================
# Process all groups (group_001 to group_013) with cam0 and cam1
# Each group folder is treated as a "pose" to process
#
# Date: 20260114
# Dataset: undistorted_fov110_center
# Groups: 13 groups × 2 cams = 26 runs
# =============================================================================

# =============================================================================
# PARAMETERS
# =============================================================================

DATE_TAG="20260114"

# We'll process each group separately - cam0 and cam1 as poses within each group
# Since the structure is different (group/cam instead of cam/pose), we need separate runs

# Batch parameters
BATCH_SIZE=30
OVERLAP=15      # stride = 30 - 15 = 15

# Matching strategy
CONF_THRES=2.5
NUM_PTS=1500

# =============================================================================
# DOCKER SETTINGS
# =============================================================================
DOCKER_IMAGE="m3rslam"
CONDA_ENV="m3rslam"
MICROMAMBA="/root/.local/bin/micromamba"

SCRIPT_DIR="BURN-Scripts"
TEMPLATE_SCRIPT="${SCRIPT_DIR}/BURN_template_BatchedDB.py"

echo "========================================"
echo "BURN SCRIPT - undistorted_fov110_center"
echo "========================================"
echo "Date: ${DATE_TAG}"
echo "Batch size: ${BATCH_SIZE}, Overlap: ${OVERLAP}, Stride: $((BATCH_SIZE - OVERLAP))"
echo "Groups: 001-013, Cams: cam0, cam1"
echo "========================================"
echo ""

# Process all groups
for GROUP_NUM in $(seq -w 1 13); do
    GROUP="group_0${GROUP_NUM}"
    # Trim leading zeros for single digits
    GROUP="group_$(printf '%03d' $((10#$GROUP_NUM)))"
    
    BASE_PATH="/d_disk/_DataBuffer/RopeCap/20251224_103638/undistorted_fov110_center/${GROUP}"
    POSES="cam0,cam1"
    OUTPUT_PREFIX="mapping3r_batched_${DATE_TAG}"
    
    echo "========================================"
    echo "Processing ${GROUP}"
    echo "========================================"
    
    PYTHON_CMD="python ${TEMPLATE_SCRIPT} \
        --base_path \"${BASE_PATH}\" \
        --poses \"${POSES}\" \
        --batch_size ${BATCH_SIZE} \
        --overlap ${OVERLAP} \
        --conf_thres ${CONF_THRES} \
        --num_pts ${NUM_PTS} \
        --output_prefix \"${OUTPUT_PREFIX}\""
    
    # Check if we're inside the container
    if [ -f "${MICROMAMBA}" ]; then
        cd /d_disk/mast3r
        eval "${MICROMAMBA} run -n ${CONDA_ENV} ${PYTHON_CMD}"
    else
        docker run --rm -it \
            --gpus=all \
            --shm-size=8g \
            -v D:/:/d_disk \
            ${DOCKER_IMAGE} \
            bash -c "cd /d_disk/mast3r && ${MICROMAMBA} run -n ${CONDA_ENV} ${PYTHON_CMD}"
    fi
done

echo "========================================"
echo "ALL GROUPS COMPLETED"
echo "========================================"
