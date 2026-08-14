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
. "${SCRIPT_DIR}/common.sh"

# Write live simulator outputs to OSMO storage, then copy small summaries back.
log_section "Prepare eval output"
EVAL_OUTPUT_DIR="${OSMO_OUTPUT_DIR}/eval"
mkdir -p "${EVAL_OUTPUT_DIR}" "${OSMO_OUTPUT_DIR}"

copy_eval_outputs() {
  cp "${EVAL_OUTPUT_DIR}/results.json" "${OSMO_OUTPUT_DIR}/results.json" 2>/dev/null || true
  cp "${EVAL_OUTPUT_DIR}/genie_sim_benchmark.log" "${OSMO_OUTPUT_DIR}/genie_sim_benchmark.log" 2>/dev/null || true
  if sudo install -d -m 0777 "${REQUESTED_EVAL_OUTPUT_DIR}" 2>/dev/null; then
    sudo cp "${EVAL_OUTPUT_DIR}/results.json" "${REQUESTED_EVAL_OUTPUT_DIR}/results.json" 2>/dev/null || true
    sudo cp "${EVAL_OUTPUT_DIR}/genie_sim_benchmark.log" "${REQUESTED_EVAL_OUTPUT_DIR}/genie_sim_benchmark.log" 2>/dev/null || true
  fi
}
trap copy_eval_outputs EXIT

# Wait until the GR00T policy server accepts websocket connections.
log_section "Wait for policy server"
echo "Waiting for Genie Sim policy server at ${POLICY_HOST}:${POLICY_PORT}..."
ready=0
for attempt in $(seq 1 450); do
  if python3 "${SCRIPT_DIR}/probe_websocket.py" "${POLICY_HOST}" "${POLICY_PORT}" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [ "${ready}" != "1" ]; then
  echo "Policy server did not open ${POLICY_HOST}:${POLICY_PORT}" >&2
  exit 2
fi

# Prepare writable Isaac Sim cache/config locations for the Genie Sim image.
log_section "Prepare Genie Sim runtime"
if [ ! -d "${GENIESIM_ROOT}" ]; then
  echo "Genie Sim root does not exist: ${GENIESIM_ROOT}" >&2
  exit 2
fi

sudo mkdir -p \
  /geniesim \
  /isaac-sim/.cache \
  /isaac-sim/.nv/ComputeCache \
  /isaac-sim/.nvidia-omniverse/logs \
  /isaac-sim/.nvidia-omniverse/config \
  /isaac-sim/.local/share/ov/data \
  /isaac-sim/.local/share/ov/pkg
sudo chown -R 1234:1234 \
  /geniesim \
  /isaac-sim/.cache \
  /isaac-sim/.nv \
  /isaac-sim/.nvidia-omniverse \
  /isaac-sim/.local/share/ov
sudo ln -sfnT "${GENIESIM_ROOT}" /geniesim/main
GENIESIM_MAIN=/geniesim/main
"${GENIESIM_MAIN}/scripts/entrypoint.sh" bash -lc "echo Genie Sim runtime ready."

# Select one Genie Sim config; TASK_NAME may be bare or already prefixed.
log_section "Select eval task"
CONFIG_DIR="${GENIESIM_MAIN}/source/geniesim/config"
case "${TASK_TYPE}" in
  if) TASK_PATTERN="if_*.yaml" ;;
  robust) TASK_PATTERN="robust_*.yaml" ;;
  manip) TASK_PATTERN="manip_*.yaml" ;;
  s2r) TASK_PATTERN="s2r_*.yaml" ;;
  spatial) TASK_PATTERN="spatial_*.yaml" ;;
  *)
    echo "Unsupported TASK_TYPE=${TASK_TYPE}" >&2
    exit 2
    ;;
esac

if [ -n "${TASK_NAME}" ]; then
  TASK_NAME="${TASK_NAME%.yaml}"
  case "${TASK_NAME}" in
    "${TASK_TYPE}"_*) CONFIG_NAME="${TASK_NAME}" ;;
    *) CONFIG_NAME="${TASK_TYPE}_${TASK_NAME}" ;;
  esac
  CONFIG_YAML="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
else
  CONFIG_YAML="$(find "${CONFIG_DIR}" -maxdepth 1 -name "${TASK_PATTERN}" -type f | sort | head -n 1)"
  CONFIG_NAME="$(basename "${CONFIG_YAML}" .yaml)"
fi

# Keep compatibility with the historical dataset spelling "billards".
if [ ! -f "${CONFIG_YAML}" ] && echo "${CONFIG_NAME}" | grep -q billards; then
  CONFIG_NAME="$(echo "${CONFIG_NAME}" | sed 's/billards/billiards/g')"
  CONFIG_YAML="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
fi
if [ ! -f "${CONFIG_YAML}" ]; then
  echo "No Genie Sim config selected: ${CONFIG_YAML}" >&2
  exit 2
fi

case "${CONFIG_NAME}" in
  if_*) RUN_TASK_TYPE=if ;;
  robust_*) RUN_TASK_TYPE=robust ;;
  manip_*) RUN_TASK_TYPE=manip ;;
  s2r_*) RUN_TASK_TYPE=s2r ;;
  spatial_*) RUN_TASK_TYPE=spatial ;;
  *) RUN_TASK_TYPE="${TASK_TYPE}" ;;
esac

# Run a tiny closed-loop batch and keep raw benchmark files under OSMO output.
log_section "Run Genie Sim benchmark"
TASK_OUTPUT="${EVAL_OUTPUT_DIR}/${CONFIG_NAME}"
mkdir -p "${TASK_OUTPUT}"
MAX_SUBTASKS_ARGS=()
if [ -n "${MAX_SUBTASKS}" ] && [ "${MAX_SUBTASKS}" != "0" ]; then
  MAX_SUBTASKS_ARGS=(--max-subtasks "${MAX_SUBTASKS}")
fi

set +e
HOME=/tmp/genie_sim_home \
XDG_CACHE_HOME=/tmp/genie_sim_cache \
XDG_CONFIG_HOME=/tmp/genie_sim_config \
XDG_DATA_HOME=/tmp/genie_sim_data \
GENIESIM_OUTPUT_DIR="${TASK_OUTPUT}" bash -lc "
  cd '${GENIESIM_MAIN}'
  bash scripts/run_batch_tasks.sh \
    --num-episode '${NUM_EPISODE}' \
    --type '${RUN_TASK_TYPE}' \
    --task '${CONFIG_NAME}' \
    ${MAX_SUBTASKS_ARGS[*]@Q} \
    --infer-host '${POLICY_HOST}:${POLICY_PORT}' \
    --output-dir '${TASK_OUTPUT}' \
    --headless
" 2>&1 | tee "${EVAL_OUTPUT_DIR}/genie_sim_benchmark.log"
EVAL_EXIT="${PIPESTATUS[0]}"
set -e

# Aggregate Genie Sim outputs and convert them to AutoResearch's metric JSON.
log_section "Record eval outputs"
rm -rf "${EVAL_OUTPUT_DIR}/benchmark"
mkdir -p "${EVAL_OUTPUT_DIR}/benchmark"
if [ -d "${TASK_OUTPUT}/benchmark" ]; then
  cp -a "${TASK_OUTPUT}/benchmark"/. "${EVAL_OUTPUT_DIR}/benchmark"/
fi
python3 "${GENIESIM_MAIN}/scripts/stat_average.py" "${EVAL_OUTPUT_DIR}/benchmark" \
  2>&1 | tee "${EVAL_OUTPUT_DIR}/benchmark/stats.txt" | tee -a "${EVAL_OUTPUT_DIR}/genie_sim_benchmark.log" || true

python3 "${SCRIPT_DIR}/eval_results_to_autoresearch.py" \
  "${EVAL_OUTPUT_DIR}/benchmark" \
  "${EVAL_OUTPUT_DIR}/results.json" \
  "${CHECKPOINT_PATH}" \
  "${CONFIG_NAME}"
copy_eval_outputs

if [ "${EVAL_EXIT}" -ne 0 ]; then
  exit "${EVAL_EXIT}"
fi
