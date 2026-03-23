#!/bin/bash
# GR00T N1.6 inference script for RTX 5090 Docker container
# Usage: bash scripts/run_inference.sh [OPTIONS]
#
# Examples:
#   bash scripts/run_inference.sh
#   bash scripts/run_inference.sh --model-path /path/to/checkpoint
#   bash scripts/run_inference.sh --dataset-path demo_data/gr1.PickNPlace --traj-ids 0 1 2

set -euo pipefail

# Defaults
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-demo_data/gr1.PickNPlace}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GR1}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
INFERENCE_MODE="${INFERENCE_MODE:-pytorch}"

python scripts/deployment/standalone_inference_script.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --traj-ids $TRAJ_IDS \
  --inference-mode "$INFERENCE_MODE" \
  --action-horizon "$ACTION_HORIZON" \
  "$@"
