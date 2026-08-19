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

log() {
  printf '[genie_sim_osmo_train] %s\n' "$*"
}

OSMO="${OSMO:-osmo}"
POOL="${POOL:-groot-l40-01}"
PRIORITY="${PRIORITY:-NORMAL}"
OSMO_POLL_INTERVAL_S="${OSMO_POLL_INTERVAL_S:-30}"
WORKFLOW="${WORKFLOW:-${SOURCE_EXAMPLE_DIR}/workflows/train_genie_sim_osmo_n2.yaml}"
if [[ -n "${AUTORESEARCH_TRIAL_ID:-}" ]]; then
  DEFAULT_WORKFLOW_NAME="${AUTORESEARCH_TRIAL_ID}-${AUTORESEARCH_STAGE:-train}-a${AUTORESEARCH_ATTEMPT:-1}"
else
  DEFAULT_WORKFLOW_NAME="n2-genie-sim-train-${USER:-user}-$(date -u +%Y%m%d-%H%M%S)"
fi
WORKFLOW_NAME="${WORKFLOW_NAME:-${DEFAULT_WORKFLOW_NAME}}"
GROOT_BRANCH="${GROOT_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
GROOT_REVISION="${GROOT_REVISION:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
GROOT_REPO="${GROOT_REPO:-gitlab-master.nvidia.com/GR00T/gr00t/gr00t.git}"

RUN_ROOT="${RUN_ROOT:-/mnt/amlfs-01/home/yuhlu/AutoResearch_genie_sim_osmo/baseline}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${SOURCE_EXAMPLE_DIR}/train_config.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-/mnt/amlfs-02/shared/datasets/agibot_data/instruction_preprocessed/pick_billards_color}"
TRAINING_OUTPUT_DIR="${RUN_ROOT}/training"

model_is_complete() {
  [[ -f "${TRAINING_OUTPUT_DIR}/config.json" ]] &&
    compgen -G "${TRAINING_OUTPUT_DIR}/*.safetensors" >/dev/null
}

if model_is_complete; then
  log "reusing completed output: ${TRAINING_OUTPUT_DIR}"
  exit 0
fi

mkdir -p "${RUN_ROOT}/config"
CONFIG_COPY="${RUN_ROOT}/config/train_config.yaml"
cp "${TRAIN_CONFIG_PATH}" "${CONFIG_COPY}"

SET_STRINGS=(
  "WORKFLOW_NAME=${WORKFLOW_NAME}"
  "RUN_ROOT=${RUN_ROOT}"
  "TRAINING_CONFIG=${CONFIG_COPY}"
  "TRAIN_DATASET=${TRAIN_DATASET}"
  "GROOT_BRANCH=${GROOT_BRANCH}"
  "GROOT_REVISION=${GROOT_REVISION}"
  "GROOT_REPO=${GROOT_REPO}"
)

log "workflow: ${WORKFLOW}"
log "pool: ${POOL}, priority: ${PRIORITY}"
log "output: ${TRAINING_OUTPUT_DIR}"
log "repo ref: ${GROOT_BRANCH} ${GROOT_REVISION}"

workflow_id="${WORKFLOW_NAME}"
if "${OSMO}" workflow query "${workflow_id}" --format-type json >/dev/null 2>&1; then
  log "reattached workflow: ${workflow_id}"
else
  submit_json="$("${OSMO}" workflow submit "${WORKFLOW}" \
    --pool "${POOL}" \
    --priority "${PRIORITY}" \
    --format-type json \
    --set-string "${SET_STRINGS[@]}")"
  workflow_id="$(printf '%s' "${submit_json}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("name", ""))')"
  if [[ -z "${workflow_id}" ]]; then
    printf '%s\n' "${submit_json}"
    echo "OSMO submit did not return a workflow name" >&2
    exit 1
  fi
  log "submitted workflow: ${workflow_id}"
fi

wait_for_osmo_workflow "${OSMO}" "${workflow_id}" "${OSMO_POLL_INTERVAL_S}"
if ! model_is_complete; then
  echo "OSMO training completed without a valid model in ${TRAINING_OUTPUT_DIR}" >&2
  exit 1
fi
