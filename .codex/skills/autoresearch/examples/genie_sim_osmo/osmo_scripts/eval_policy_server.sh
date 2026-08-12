#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

log_section "Check checkpoint"
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
nvidia-smi

log_section "Set up policy-server environment"
export UV_CACHE_DIR=/tmp/uv-cache
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
bash skills/autoresearch/scripts/setup_env.sh
PYTHON="${PWD}/.venv/bin/python"
uv pip install --python "${PYTHON}" websockets msgpack msgpack-numpy

log_section "Serve policy"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/amlfs-02/shared/ckpts}"
export HF_HUB_OFFLINE=1
export GIT_LFS_SKIP_SMUDGE=1
export OTEL_SDK_DISABLED=true
export PYTHONUNBUFFERED=1
export ENABLE_TENSORRT=True

exec "${PYTHON}" \
  skills/autoresearch/examples/genie_sim_osmo/osmo_scripts/serve_n2_genie_sim_websocket.py \
  --model-path "${CHECKPOINT_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --host 0.0.0.0 \
  --port "${POLICY_PORT}" \
  --device cuda:0 \
  --num-inference-timesteps "${DENOISING_STEPS}" \
  --action-horizon "${ACTION_HORIZON}"
