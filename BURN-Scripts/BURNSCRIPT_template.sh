#!/bin/bash
# =============================================================================
# BURN SCRIPT TEMPLATE - BatchedDB Processing
# =============================================================================
# This is a template .sh file. Copy and modify the parameters below for new runs.
#
# Usage (from host Windows):
#   bash BURN-Scripts/BURNSCRIPT_YYYYMMDD_description.sh
#
# Usage (from inside container):
#   /root/.local/bin/micromamba run -n slam3r_p310_c126_t270 bash BURN-Scripts/BURNSCRIPT_xxx.sh
# =============================================================================

# =============================================================================
# PARAMETERS - Modify these for each run
# =============================================================================

# Date tag for output folders
DATE_TAG="20260114"

# Camera and poses
BASE_PATH="/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_fov110_720sq_final/cam0"
POSES="p+0_y+15_r+0,p-20_y+15_r+0"

# Batch parameters
BATCH_SIZE=36
OVERLAP=24      # stride = BATCH_SIZE - OVERLAP = 12

# Matching strategy
CONF_THRES=2.0
NUM_PTS=2000

# Output prefix (will be combined with sanitized pose name)
OUTPUT_PREFIX="mapping3r_batched_${DATE_TAG}"

# =============================================================================
# DOCKER SETTINGS (only used when running from host)
# =============================================================================
DOCKER_IMAGE="m3rslam"
CONDA_ENV="m3rslam"
MICROMAMBA="/root/.local/bin/micromamba"

# =============================================================================
# EXECUTION
# =============================================================================

SCRIPT_DIR="BURN-Scripts"
TEMPLATE_SCRIPT="${SCRIPT_DIR}/BURN_template_BatchedDB.py"

echo "========================================"
echo "BURN SCRIPT - BatchedDB Processing"
echo "========================================"
echo "Date: ${DATE_TAG}"
echo "Base path: ${BASE_PATH}"
echo "Poses: ${POSES}"
echo "Batch size: ${BATCH_SIZE}, Overlap: ${OVERLAP}, Stride: $((BATCH_SIZE - OVERLAP))"
echo "Conf threshold: ${CONF_THRES}, Num points: ${NUM_PTS}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "========================================"
echo ""

# Build the python command
PYTHON_CMD="python ${TEMPLATE_SCRIPT} \
    --base_path \"${BASE_PATH}\" \
    --poses \"${POSES}\" \
    --batch_size ${BATCH_SIZE} \
    --overlap ${OVERLAP} \
    --conf_thres ${CONF_THRES} \
    --num_pts ${NUM_PTS} \
    --output_prefix \"${OUTPUT_PREFIX}\""

# Check if we're inside the container (check if micromamba exists)
if [ -f "${MICROMAMBA}" ]; then
    echo "Running inside container..."
    cd /d_disk/mast3r
    eval "${MICROMAMBA} run -n ${CONDA_ENV} ${PYTHON_CMD}"
else
    echo "Running from host, launching Docker container..."
    docker run --rm -it \
        --gpus=all \
        --shm-size=8g \
        -v D:/:/d_disk \
        ${DOCKER_IMAGE} \
        bash -c "cd /d_disk/mast3r && ${MICROMAMBA} run -n ${CONDA_ENV} ${PYTHON_CMD}"
fi
