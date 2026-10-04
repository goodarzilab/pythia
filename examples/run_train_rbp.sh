#!/usr/bin/env bash
# End-to-end example: train + evaluate PythiaRBP for one RBP dataset.
#
# Usage:
#   bash run_train_rbp.sh <data-dir> <rbp-name> <output-dir>
#
# <data-dir> must contain:
#   {rbp-name}_trainingSet.tsv.gz
#   {rbp-name}_tuningSet.tsv.gz
#   {rbp-name}_validationSet.tsv.gz
# in the TSV format documented in the top-level README (Input, Response,
# SeqNames, MFEs columns).
#
# Hyperparameters below are the deployed configuration used for the
# published RBP benchmark results (§3 of the porting notes): binarized
# FixedDilatedConv features, dil_start=2, dil_end=48.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash run_train_rbp.sh <data-dir> <rbp-name> <output-dir>" >&2
    exit 1
fi

DATA_DIR="$1"
RBP="$2"
OUTPUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/../src/pythia" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python"
fi

TRAIN_TSV="${DATA_DIR}/${RBP}_trainingSet.tsv.gz"
VAL_TSV="${DATA_DIR}/${RBP}_tuningSet.tsv.gz"
TEST_TSV="${DATA_DIR}/${RBP}_validationSet.tsv.gz"

for f in "${TRAIN_TSV}" "${VAL_TSV}" "${TEST_TSV}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required input not found: ${f}" >&2
        exit 1
    fi
done

# ---- Hyperparameters (deployed configuration) ------------------------------
INPUTSIZE=256
MAX_EPOCHS=60
BATCH_SIZE=256
LR=0.004
DIL_START=2
DIL_END=48
BULGE_SIZE=2
DP=0.5
PATIENCE=20

mkdir -p "${OUTPUT_DIR}"

echo "=== [${RBP}] training ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/train_rbp.py" \
    --train-tsv   "${TRAIN_TSV}" \
    --val-tsv     "${VAL_TSV}" \
    --output-dir  "${OUTPUT_DIR}" \
    --inputsize   "${INPUTSIZE}" \
    --max-epochs  "${MAX_EPOCHS}" \
    --batch-size  "${BATCH_SIZE}" \
    --lr          "${LR}" \
    --dil-start   "${DIL_START}" \
    --dil-end     "${DIL_END}" \
    --bulge-size  "${BULGE_SIZE}" \
    --dp          "${DP}" \
    --patience    "${PATIENCE}" \
    --binarize-fd \
    --num-workers 4

CKPT=$(ls "${OUTPUT_DIR}/checkpoints/"*.ckpt 2>/dev/null | head -n 1 || true)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: no checkpoint found in ${OUTPUT_DIR}/checkpoints/" >&2
    exit 1
fi
echo "=== [${RBP}] best checkpoint: ${CKPT} ==="

echo "=== [${RBP}] inference on validation set ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/infer.py" \
    --task        rbp \
    --checkpoint  "${CKPT}" \
    --input-csv   "${TEST_TSV}" \
    --output-csv  "${OUTPUT_DIR}/${RBP}_validationPredictions.tsv" \
    --inputsize   "${INPUTSIZE}" \
    --batch-size  "${BATCH_SIZE}" \
    --num-workers 4

echo "=== [${RBP}] plotting ==="
"${VENV_PYTHON}" "${PACKAGE_DIR}/plot.py" \
    --task        rbp \
    --predictions "${OUTPUT_DIR}/${RBP}_validationPredictions.tsv" \
    --output-dir  "${OUTPUT_DIR}/figures"

echo "=== [${RBP}] done — results at ${OUTPUT_DIR}/${RBP}_validationPredictions.tsv ==="
