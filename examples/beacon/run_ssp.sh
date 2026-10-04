#!/usr/bin/env bash
# End-to-end example: train + evaluate PythiaSSP (secondary structure
# prediction) with the deployed hyperparameters.
#
# Usage:
#   bash run_ssp.sh <bprna-csv> <output-dir>
#
# <bprna-csv> is a bpRNA-style CSV with columns data_name (TR0/VL0/TS0),
# file_name, seq, dot_string -- see the top-level README for the public
# source of bpRNA data.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_ssp.sh <bprna-csv> <output-dir>" >&2
    exit 1
fi

CSV="$1"
OUTPUT_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/../../src/pythia" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python"
fi

# ---- Hyperparameters (deployed configuration) ------------------------------
MAX_LEN=512
MAX_EPOCHS=300
BATCH_SIZE=4
BATCH_EVAL=4
GRAD_ACCUM=2
LR=3e-4
WEIGHT_DECAY=0
WARMUP_EPOCHS=3
PATIENCE=10
DIL_START=5
DIL_END=24
BULGE_SIZE=2
DROPOUT=0.15
HIDDEN_DIM=384
NUM_RES_LAYERS=8
DIST_ENC_DIM=64

mkdir -p "${OUTPUT_DIR}"

echo "=== Step 1: Training PythiaSSP ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/train_beacon.py" \
    --task           ssp \
    --csv            "${CSV}" \
    --output-dir     "${OUTPUT_DIR}" \
    --max-len        "${MAX_LEN}" \
    --max-epochs     "${MAX_EPOCHS}" \
    --batch-size     "${BATCH_SIZE}" \
    --batch-eval     "${BATCH_EVAL}" \
    --grad-accum     "${GRAD_ACCUM}" \
    --lr             "${LR}" \
    --weight-decay   "${WEIGHT_DECAY}" \
    --warmup-epochs  "${WARMUP_EPOCHS}" \
    --patience       "${PATIENCE}" \
    --dil-start      "${DIL_START}" \
    --dil-end        "${DIL_END}" \
    --bulge-size     "${BULGE_SIZE}" \
    --dropout        "${DROPOUT}" \
    --hidden-dim     "${HIDDEN_DIM}" \
    --num-res-layers "${NUM_RES_LAYERS}" \
    --dist-enc-dim   "${DIST_ENC_DIM}" \
    --arm1-widths 128 128 128 \
    --arm2-widths 256 128 128 \
    --num-workers 4

echo "=== Step 2: Plotting performance ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/plot.py" \
    --task         ssp \
    --metrics-json "${OUTPUT_DIR}/metrics_best.json" \
    --output-dir   "${OUTPUT_DIR}/figures"

echo "=== SSP training complete ==="
