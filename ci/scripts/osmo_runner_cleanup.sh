#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Cleanup: Cancel OSMO workflows if job is cancelled or interrupted
#
# Usage: osmo_runner_cleanup.sh <workflow_dir>
#   workflow_dir: directory containing list_sharded.yaml and workflow YAML files
#
# Required environment variables:
#   CI_PROJECT_DIR
#   BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, BUCKET_REGION
#   OSMO_TOKEN, OSMO_URL

set -euo pipefail # Exit on error, undefined vars, and pipe failures

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <workflow_dir> [--zombie]"
  echo "  workflow_dir: directory containing list_sharded.yaml and workflow YAML files"
  exit 1
fi
WORKFLOW_DIR="${1%/}"
WORKFLOW_LABEL="${OSMO_WORKFLOW_LABEL:-$(basename "${WORKFLOW_DIR}" | sed 's/_workflows$//')}"
ZOMBIE_MODE="${2:-}"
WORKFLOW_IDS_FILE=${CI_PROJECT_DIR}/active_workflows.txt
OSMO_AWS_CREDENTIAL_NAME="${OSMO_AWS_CREDENTIAL_NAME:-gear-aws-cred-${CI_PIPELINE_ID:-manual}-${CI_JOB_ID:-manual}}"

# Setup AWS credentials for s5cmd
export AWS_ACCESS_KEY_ID=${BUCKET_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${BUCKET_SECRET_ACCESS_KEY}
export AWS_REGION=${BUCKET_REGION}
export AWS_ENDPOINT_URL=${BUCKET_ENDPOINT_URL}
export S3_ENDPOINT_URL=${BUCKET_ENDPOINT_URL}

if [[ -f "${CI_PROJECT_DIR}/log.json" ]]; then
  echo "log.json exists, osmo_runner completed successfully. Skipping cleanup."

  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
  fi

  # Persist test results to database
  echo "Persisting test results to database..."
  uv run --active -m groot.infra.ci.main.testdb \
    --osmo-log-json "${CI_PROJECT_DIR}"/log.json \
    --osmo-workflow-list-yaml "${CI_PROJECT_DIR}/${WORKFLOW_DIR}/list_sharded.yaml" \
    --main-branch-only

  if [[ "${CI_COMMIT_BRANCH:-}" = "${CI_DEFAULT_BRANCH:-main}" ]]; then
    echo "Recording postmerge workflow result timestamps..."
    uv run --active -m groot.infra.ci.main.postmerge_test_rate_limit record \
      --log-file "${CI_PROJECT_DIR}/log.json" \
      --workflow-type "${WORKFLOW_LABEL}" \
      --commit-sha "${CI_COMMIT_SHA:-}" \
      --pipeline-url "${CI_PIPELINE_URL:-}" \
      --window-hours "${POSTMERGE_TEST_RATE_LIMIT_HOURS:-24}" \
      || echo "⚠️  Postmerge result timestamp recording failed (non-blocking)"
  fi

  osmo credential delete "${OSMO_AWS_CREDENTIAL_NAME}" || true
  exit 0
fi

if [[ ! -f "${WORKFLOW_IDS_FILE}" ]]; then
  echo "No workflow tracking file found, skipping cleanup"
  osmo credential delete "${OSMO_AWS_CREDENTIAL_NAME}" || true
  exit 0
fi

echo "Checking for submitted workflows to cancel..."

# Install dependencies only for cancellation; successful runs already have
# log.json and do not need osmo_runner in after_script.
command -v s5cmd >/dev/null
bash ci/scripts/osmo_runner_install.sh

# Extract unique workflow IDs from tracking file
workflow_ids=$(cat "${WORKFLOW_IDS_FILE}" | sort -u | grep -v '^$' || true)

if [[ -z "${workflow_ids}" ]]; then
  echo "No workflow IDs found in ${WORKFLOW_IDS_FILE}, skipping cleanup"
  exit 0
fi

echo "Found workflow IDs to cancel:"
echo "${workflow_ids}"
echo ""
echo "Cancelling workflows:"

# Cancel each workflow
echo "${workflow_ids}" | while read -r workflow_id; do
  if [[ -n "${workflow_id}" ]]; then
    echo "  🛑 Cancelling: ${workflow_id}"
    osmo workflow cancel "${workflow_id}" || echo "    ⚠️  Failed to cancel ${workflow_id}"
  fi
done

osmo credential delete "${OSMO_AWS_CREDENTIAL_NAME}" || true
echo "Cleanup complete"
exit 0
