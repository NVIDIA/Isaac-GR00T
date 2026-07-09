#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workflow_yaml> [osmo workflow submit args...]"
  exit 1
fi

WORKFLOW_YAML="$1"
shift
WORKFLOW_ID_FILE="${OSMO_WORKFLOW_ID_FILE:-${CI_PROJECT_DIR:-$PWD}/active_workflows.txt}"

echo "--- Submitting OSMO workflow: ${WORKFLOW_YAML} ---"
SUBMIT_RC=0
SUBMIT_OUTPUT=$(osmo workflow submit "${WORKFLOW_YAML}" "$@" 2>&1) || SUBMIT_RC=$?
echo "${SUBMIT_OUTPUT}"

if [[ "${SUBMIT_RC}" -ne 0 ]]; then
  echo "ERROR: osmo workflow submit failed (exit ${SUBMIT_RC})" >&2
  exit "${SUBMIT_RC}"
fi

WORKFLOW_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'Workflow ID\s+-\s+\K\S+' | head -1 || true)
if [[ -z "${WORKFLOW_ID}" ]]; then
  echo "ERROR: Could not extract workflow ID from submit output" >&2
  exit 1
fi

mkdir -p "$(dirname "${WORKFLOW_ID_FILE}")"
echo "${WORKFLOW_ID}" >> "${WORKFLOW_ID_FILE}"
echo "--- OSMO workflow submitted: ${WORKFLOW_ID} ---"

PREV_LINE_COUNT=0
while true; do
  CURRENT_LOGS=$(osmo workflow logs "${WORKFLOW_ID}" 2>/dev/null || true)
  if [[ -n "${CURRENT_LOGS}" ]]; then
    CURRENT_LINE_COUNT=$(echo "${CURRENT_LOGS}" | wc -l)
    if [[ "${CURRENT_LINE_COUNT}" -gt "${PREV_LINE_COUNT}" ]]; then
      echo "${CURRENT_LOGS}" | tail -n +"$((PREV_LINE_COUNT + 1))"
      PREV_LINE_COUNT="${CURRENT_LINE_COUNT}"
    fi
  fi

  STATUS=$(osmo workflow query "${WORKFLOW_ID}" 2>/dev/null \
    | grep -E '^\s*Status\s*:' \
    | sed 's/.*:\s*//' \
    | tr -d '[:space:]' \
    || true)

  case "${STATUS}" in
    COMPLETED|SUCCEEDED)
      echo "--- OSMO workflow completed: ${WORKFLOW_ID} ---"
      : > "${WORKFLOW_ID_FILE}"
      exit 0
      ;;
    FAILED|FAILED_CANCELED|CANCELED|CANCELLED)
      echo "ERROR: OSMO workflow ${WORKFLOW_ID} failed with status: ${STATUS}" >&2
      exit 1
      ;;
  esac

  sleep "${OSMO_WORKFLOW_LOG_POLL_SECONDS:-15}"
done
