#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
if [[ -d "${REPO_ROOT}/oss/skills/autoresearch" ]]; then
  PACKAGE_ROOT="${REPO_ROOT}/oss"
else
  PACKAGE_ROOT="${REPO_ROOT}"
fi
SOURCE_EXAMPLE_DIR="${PACKAGE_ROOT}/skills/autoresearch/examples/genie_sim_osmo"
. "${SOURCE_EXAMPLE_DIR}/osmo_scripts/common.sh"
PYTHON="${PYTHON:-${PACKAGE_ROOT}/.venv/bin/python}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-${SOURCE_EXAMPLE_DIR}/eval_config.yaml}"

log() {
  printf '[genie_sim_osmo_eval] %s\n' "$*"
}

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment does not exist: ${PYTHON}" >&2
  exit 1
fi
# Read config lines with a while-read loop so this stays portable to bash 3.2
# (macOS default; the shebang promises any bash). Process substitution keeps the
# loop body in this shell so the array survives.
CONFIG_VALUES=()
while IFS= read -r config_value; do
  CONFIG_VALUES+=("${config_value}")
done < <("${PYTHON}" -c '
import os, sys, yaml
c = yaml.safe_load(os.path.expandvars(open(sys.argv[1], encoding="utf-8").read()))
for section, key in [
    ("osmo", "pool"), ("osmo", "priority"),
    ("simulation", "geniesim_root"), ("simulation", "embodiment_tag"),
    ("simulation", "task_type"), ("simulation", "task_name"),
    ("simulation", "num_episode"), ("simulation", "max_subtasks"),
    ("policy", "action_horizon"), ("policy", "denoising_steps"),
    ("policy", "port"),
]:
    print(c[section][key])
' "${EVAL_CONFIG_PATH}")
if (( ${#CONFIG_VALUES[@]} != 11 )); then
  echo "Invalid eval config: ${EVAL_CONFIG_PATH}" >&2
  exit 2
fi

OSMO="${OSMO:-osmo}"
POOL="${POOL:-${CONFIG_VALUES[0]}}"
PRIORITY="${PRIORITY:-${CONFIG_VALUES[1]}}"
if [[ "${POOL}" == "your-osmo-pool" ]]; then
  echo "Set POOL to an OSMO pool available to your account" >&2
  exit 2
fi
OSMO_POLL_INTERVAL_S="${OSMO_POLL_INTERVAL_S:-30}"
GENIESIM_ROOT="${GENIESIM_ROOT:-${CONFIG_VALUES[2]}}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-${CONFIG_VALUES[3]}}"
TASK_TYPE="${TASK_TYPE:-${CONFIG_VALUES[4]}}"
TASK_NAME="${TASK_NAME:-${CONFIG_VALUES[5]}}"
NUM_EPISODE="${NUM_EPISODE:-${CONFIG_VALUES[6]}}"
MAX_SUBTASKS="${MAX_SUBTASKS:-${CONFIG_VALUES[7]}}"
ACTION_HORIZON="${ACTION_HORIZON:-${CONFIG_VALUES[8]}}"
DENOISING_STEPS="${DENOISING_STEPS:-${CONFIG_VALUES[9]}}"
POLICY_PORT="${POLICY_PORT:-${CONFIG_VALUES[10]}}"

DRY_RUN="${DRY_RUN:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
WORKFLOW="${WORKFLOW:-${SOURCE_EXAMPLE_DIR}/workflows/eval_genie_sim_osmo_n2.yaml}"
if [[ -n "${AUTORESEARCH_TRIAL_ID:-}" ]]; then
  DEFAULT_WORKFLOW_NAME="${AUTORESEARCH_TRIAL_ID}-${AUTORESEARCH_STAGE:-eval}-a${AUTORESEARCH_ATTEMPT:-1}"
else
  DEFAULT_WORKFLOW_NAME="n2-genie-sim-eval-${USER:-user}-$(date -u +%Y%m%d-%H%M%S)"
fi
WORKFLOW_NAME="${WORKFLOW_NAME:-${DEFAULT_WORKFLOW_NAME}}"
GROOT_BRANCH="${GROOT_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
GROOT_REVISION="${GROOT_REVISION:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
GROOT_REPO="${GROOT_REPO:-github.com/NVIDIA/Isaac-GR00T-CRA.git}"

AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:?Set AUTORESEARCH_ROOT to a writable shared directory}"
RUN_ROOT="${RUN_ROOT:-${AUTORESEARCH_ROOT}/genie_sim_osmo/baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_ROOT}/training}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/eval}"
RESULT_PATH="${RESULT_PATH:-${EVAL_OUTPUT_DIR}/results.json}"
if [[ -s "${RESULT_PATH}" ]]; then
  log "reusing completed result: ${RESULT_PATH}"
  exit 0
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

SET_VALUES=(
  "ACTION_HORIZON=${ACTION_HORIZON}"
  "DENOISING_STEPS=${DENOISING_STEPS}"
  "NUM_EPISODE=${NUM_EPISODE}"
  "MAX_SUBTASKS=${MAX_SUBTASKS}"
  "POLICY_PORT=${POLICY_PORT}"
)
SET_STRINGS=(
  "WORKFLOW_NAME=${WORKFLOW_NAME}"
  "AUTORESEARCH_ROOT=${AUTORESEARCH_ROOT}"
  "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
  "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
  "GENIESIM_ROOT=${GENIESIM_ROOT}"
  "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
  "TASK_TYPE=${TASK_TYPE}"
  "TASK_NAME=${TASK_NAME}"
  "GROOT_BRANCH=${GROOT_BRANCH}"
  "GROOT_REVISION=${GROOT_REVISION}"
  "GROOT_REPO=${GROOT_REPO}"
)

log "workflow: ${WORKFLOW}"
log "pool: ${POOL}, priority: ${PRIORITY}"
log "checkpoint: ${CHECKPOINT_PATH}"
log "eval output: ${EVAL_OUTPUT_DIR}"
log "repo ref: ${GROOT_BRANCH} ${GROOT_REVISION}"

OSMO_ARGS=(
  --pool "${POOL}"
  --set "${SET_VALUES[@]}"
  --set-string "${SET_STRINGS[@]}"
)
if [[ "${VALIDATE_ONLY}" == 1 ]]; then
  exec "${OSMO}" workflow validate "${WORKFLOW}" "${OSMO_ARGS[@]}"
fi
if [[ "${DRY_RUN}" == 1 ]]; then
  exec "${OSMO}" workflow submit "${WORKFLOW}" \
    --priority "${PRIORITY}" --dry-run "${OSMO_ARGS[@]}"
fi

workflow_id="${WORKFLOW_NAME}"
if "${OSMO}" workflow query "${workflow_id}" --format-type json >/dev/null 2>&1; then
  log "reattached workflow: ${workflow_id}"
else
  submit_json="$("${OSMO}" workflow submit "${WORKFLOW}" \
    --priority "${PRIORITY}" --format-type json "${OSMO_ARGS[@]}")"
  workflow_id="$(printf '%s' "${submit_json}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("name", ""))')"
  if [[ -z "${workflow_id}" ]]; then
    printf '%s\n' "${submit_json}"
    echo "OSMO submit did not return a workflow name" >&2
    exit 1
  fi
  log "submitted workflow: ${workflow_id}"
fi

wait_for_osmo_workflow "${OSMO}" "${workflow_id}" "${OSMO_POLL_INTERVAL_S}"
if [[ ! -s "${RESULT_PATH}" ]]; then
  echo "OSMO evaluation completed without ${RESULT_PATH}" >&2
  exit 1
fi
