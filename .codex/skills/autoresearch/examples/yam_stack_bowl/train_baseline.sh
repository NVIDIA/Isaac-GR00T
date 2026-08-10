#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
if [[ -d "${REPO_ROOT}/oss/skills/autoresearch" ]]; then
  PACKAGE_ROOT="${REPO_ROOT}/oss"
else
  PACKAGE_ROOT="${REPO_ROOT}"
fi
PYTHON="${PYTHON:-${PACKAGE_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${PACKAGE_ROOT}/skills/autoresearch/examples/yam_stack_bowl/train_config.yaml}"
RUN_ROOT="${RUN_ROOT:-/mnt/amlfs-01/home/yuhlu/AutoResearch_yam_stack_bowl_full/baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/training}"

mkdir -p "${OUTPUT_DIR}"
exec 9>"${RUN_ROOT}/.train.lock"
flock 9
model_is_complete() {
  [[ -f "${OUTPUT_DIR}/config.json" ]] &&
    compgen -G "${OUTPUT_DIR}/*.safetensors" >/dev/null
}

if model_is_complete; then
  printf '[yam_stack_bowl_train] reusing completed output: %s\n' "${OUTPUT_DIR}"
  exit 0
fi

export REPO_ROOT PYTHON
export DS_IGNORE_CUDA_DETECTION=1
export ENABLE_TENSORRT=True
export SKIP_EXPERIMENT_REGISTRATION=1
unset RAY_ADDRESS

REPO_ROOT="${PACKAGE_ROOT}" bash "${PACKAGE_ROOT}/skills/autoresearch/scripts/train_local.sh" \
  "${CONFIG_PATH}" \
  --training.output-dir "${OUTPUT_DIR}"

# train_local.sh returns only after the final model has been saved at OUTPUT_DIR.
# Numbered checkpoints are no longer needed once that durable final model exists.
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +

if ! model_is_complete; then
  echo "Training completed without a valid model in ${OUTPUT_DIR}" >&2
  exit 1
fi
