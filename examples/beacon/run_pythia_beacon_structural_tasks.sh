#!/usr/bin/env bash
# Runs all 4 BEACON structural tasks in sequence:
#   pythia_beacon_structural_tasks.py --task <ssp|ssi|cmp|dmp>
#
# Usage: bash run_pythia_beacon_structural_tasks.sh [config.yaml]
#
# Edit pythia_beacon_config.yaml for your environment before running --
# the placeholder data paths in that file are not shipped with this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    VENV_PYTHON="python"
fi
CONFIG="${1:-${SCRIPT_DIR}/pythia_beacon_config.yaml}"

TASKS=(ssp ssi cmp dmp)

for TASK in "${TASKS[@]}"; do
    echo "=== Running task: ${TASK} ==="
    "${VENV_PYTHON}" "${SCRIPT_DIR}/pythia_beacon_structural_tasks.py" --config "${CONFIG}" --task "${TASK}"
done
