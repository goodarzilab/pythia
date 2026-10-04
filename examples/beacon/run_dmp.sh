#!/usr/bin/env bash
# End-to-end example: train + evaluate PythiaDMP (distance map prediction)
# with the deployed hyperparameters.
#
# Usage:
#   bash run_dmp.sh <distance-map-data-dir> <output-dir>
#
# <distance-map-data-dir> is a BEACON DistanceMap-style directory containing
# train.csv, val.csv (columns: id, input) and a distance_map/ subdirectory
# of {id}.npy distance maps already normalized to [0, 1] -- see the
# top-level README.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_dmp.sh <distance-map-data-dir> <output-dir>" >&2
    exit 1
fi

BEACON_DATA="$1"
OUTPUT_DIR="$2"
TRAIN_CSV="${BEACON_DATA}/train.csv"
VAL_CSV="${BEACON_DATA}/val.csv"
DISTANCE_DIR="${BEACON_DATA}/distance_map"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/../../src/pythia" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python"
fi

# ---- Hyperparameters (deployed configuration) ------------------------------
MAX_LEN=1024
MAX_EPOCHS=100
BATCH_SIZE=1
BATCH_EVAL=1
GRAD_ACCUM=8          # effective batch = BATCH_SIZE * GRAD_ACCUM = 8
LR=1e-4
WEIGHT_DECAY=1e-3
WARMUP_EPOCHS=20
PATIENCE=10
DIL_START=5
DIL_END=24
BULGE_SIZE=2
DROPOUT=0.1
HIDDEN_DIM=256
NUM_RES_LAYERS=9
DIST_ENC_DIM=128
HUBER_DELTA=0.3

mkdir -p "${OUTPUT_DIR}"

echo "=== Step 1: Training PythiaDMP ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/train_beacon.py" \
    --task              dmp \
    --train-csv         "${TRAIN_CSV}" \
    --val-csv           "${VAL_CSV}" \
    --distance-dir      "${DISTANCE_DIR}" \
    --test-splits       RFAM19 DIRECT test \
    --output-dir        "${OUTPUT_DIR}" \
    --max-len           "${MAX_LEN}" \
    --max-epochs        "${MAX_EPOCHS}" \
    --batch-size        "${BATCH_SIZE}" \
    --batch-eval        "${BATCH_EVAL}" \
    --grad-accum        "${GRAD_ACCUM}" \
    --lr                "${LR}" \
    --weight-decay      "${WEIGHT_DECAY}" \
    --warmup-epochs     "${WARMUP_EPOCHS}" \
    --patience          "${PATIENCE}" \
    --dil-start         "${DIL_START}" \
    --dil-end           "${DIL_END}" \
    --bulge-size        "${BULGE_SIZE}" \
    --dropout           "${DROPOUT}" \
    --hidden-dim        "${HIDDEN_DIM}" \
    --num-res-layers    "${NUM_RES_LAYERS}" \
    --dist-enc-dim      "${DIST_ENC_DIM}" \
    --arm1-widths 192 192 128 \
    --arm2-widths 256 256 128 \
    --huber-delta       "${HUBER_DELTA}" \
    --num-workers 4

echo "=== Step 2: Plotting performance ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/plot.py" \
    --task         dmp \
    --metrics-json "${OUTPUT_DIR}/metrics_best.json" \
    --output-dir   "${OUTPUT_DIR}/figures"

echo "=== DMP training complete ==="
