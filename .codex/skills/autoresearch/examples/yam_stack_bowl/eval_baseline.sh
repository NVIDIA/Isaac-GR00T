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
CONFIG_PATH="${CONFIG_PATH:-${PACKAGE_ROOT}/skills/autoresearch/examples/yam_stack_bowl/eval_config.yaml}"
RUN_ROOT="${RUN_ROOT:-/mnt/amlfs-01/home/yuhlu/AutoResearch_yam_stack_bowl_full/baseline}"
MODEL_PATH="${CHECKPOINT_PATH:-${MODEL_PATH:-${RUN_ROOT}/training}}"
EVAL_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/eval}"
RESULT_PATH="${RESULT_PATH:-${EVAL_DIR}/results.json}"

mkdir -p "${EVAL_DIR}"
exec 9>"${RUN_ROOT}/.eval.lock"
flock 9
if [[ -s "${RESULT_PATH}" ]]; then
  printf '[yam_stack_bowl_eval] reusing completed result: %s\n' "${RESULT_PATH}"
  exit 0
fi
export REPO_ROOT PYTHON

REPO_ROOT="${PACKAGE_ROOT}" bash "${PACKAGE_ROOT}/skills/autoresearch/scripts/eval_local.sh" \
  "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-json "${RESULT_PATH}" \
  --save-plot-dir "${EVAL_DIR}/plots"
