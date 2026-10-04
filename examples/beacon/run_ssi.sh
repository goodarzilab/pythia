#!/usr/bin/env bash
# End-to-end example: train + evaluate PythiaSSI (structural score
# imputation) with the deployed hyperparameters (hyperopt TPE search,
# 50 trials, seed=42: val R2=0.522, test R2=0.396).
#
# Usage:
#   bash run_ssi.sh <ssi-data-dir> <output-dir>
#
# <ssi-data-dir> must contain train.csv, val.csv, test.csv with columns
# sequence, struct (observed scores, -1 = missing), struct_true -- see the
# top-level README.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_ssi.sh <ssi-data-dir> <output-dir>" >&2
    exit 1
fi

SSI_BASE="$1"
OUTPUT_DIR="$2"
TRAIN_CSV="${SSI_BASE}/train.csv"
VAL_CSV="${SSI_BASE}/val.csv"
TEST_CSV="${SSI_BASE}/test.csv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/../../src/pythia" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python"
fi

# ---- Hyperparameters (deployed configuration) ------------------------------
MAX_LEN=440
HIDDEN_DIM=384
NUM_RES_LAYERS=10
DIL_START=2
DIL_END=36
BULGE_SIZE=4
DROPOUT=0.307
MAX_EPOCHS=100
BATCH_SIZE=64
BATCH_EVAL=128
GRAD_ACCUM=1
LR_FEAT=7.071e-4
LR_HEAD=2.598e-4
WEIGHT_DECAY=8.864e-3
WARMUP_EPOCHS=20
PATIENCE=12
LOSS_FN=l1
GRAD_CLIP=1.152

mkdir -p "${OUTPUT_DIR}"

echo "=== Step 1: Training PythiaSSI ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/train_beacon.py" \
    --task           ssi \
    --train-csv      "${TRAIN_CSV}" \
    --val-csv        "${VAL_CSV}" \
    --test-csv       "${TEST_CSV}" \
    --output-dir     "${OUTPUT_DIR}" \
    --max-len        "${MAX_LEN}" \
    --hidden-dim     "${HIDDEN_DIM}" \
    --num-res-layers "${NUM_RES_LAYERS}" \
    --arm1-widths 128 128 128 \
    --arm2-widths 256 128 128 \
    --dil-start      "${DIL_START}" \
    --dil-end        "${DIL_END}" \
    --bulge-size     "${BULGE_SIZE}" \
    --dropout        "${DROPOUT}" \
    --max-epochs     "${MAX_EPOCHS}" \
    --batch-size     "${BATCH_SIZE}" \
    --batch-eval     "${BATCH_EVAL}" \
    --grad-accum     "${GRAD_ACCUM}" \
    --lr-feat        "${LR_FEAT}" \
    --lr-head        "${LR_HEAD}" \
    --weight-decay   "${WEIGHT_DECAY}" \
    --warmup-epochs  "${WARMUP_EPOCHS}" \
    --patience       "${PATIENCE}" \
    --loss-fn        "${LOSS_FN}" \
    --grad-clip      "${GRAD_CLIP}" \
    --num-workers    4

echo "=== Step 2: Inference on test set ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/infer.py" \
    --task        ssi \
    --checkpoint  "${OUTPUT_DIR}/ssi_best.pt" \
    --input-csv   "${TEST_CSV}" \
    --output-csv  "${OUTPUT_DIR}/ssi_predictions.csv" \
    --max-len     "${MAX_LEN}" \
    --hidden-dim     "${HIDDEN_DIM}" \
    --num-res-layers "${NUM_RES_LAYERS}" \
    --arm1-widths 128 128 128 \
    --arm2-widths 256 128 128 \
    --dil-start  "${DIL_START}" \
    --dil-end    "${DIL_END}" \
    --bulge-size "${BULGE_SIZE}" \
    --dropout    "${DROPOUT}" \
    --batch-size "${BATCH_EVAL}" \
    --num-workers 4

echo "=== Step 3: Plotting performance ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/plot.py" \
    --task         ssi \
    --metrics-json "${OUTPUT_DIR}/metrics_best.json" \
    --output-dir   "${OUTPUT_DIR}/figures"

echo "=== SSI training complete ==="
