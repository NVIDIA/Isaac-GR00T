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
# Run OSMO workflows and validate results
#
# Usage: osmo_runner_execute.sh <workflow_dir> [--zombie]
#   workflow_dir: directory containing list.template.yaml and workflow YAML files
#   --zombie: run only zombie (quarantined) workflows; without this flag, zombie workflows are excluded
#
# Required environment variables:
#   CI_PROJECT_DIR, CI_DEFAULT_BRANCH
#   OSMO_POOL, IMAGE, GROOT_COMMIT_BRANCH
#   BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, BUCKET_ENDPOINT_URL, BUCKET_REGION
#   OSMO_TOKEN, OSMO_URL
#
# Optional environment variables:
#   CI_NODE_INDEX  - 1-based shard index (set by GitLab CI parallel: N)
#   CI_NODE_TOTAL  - total number of shards (set by GitLab CI parallel: N)
#   CI_COMMIT_BRANCH - used to set workflow priority (may be unset in MR pipelines)
#   OSMO_CACHE_ENABLED - set to "false" to disable per-workflow result caching (default: true)
#   OSMO_WORKFLOW_INCLUDE_REGEX - optional regex selecting workflow keys to keep

set -euo pipefail # Exit on error, undefined vars, and pipe failures

# Result caching: skip already-passed workflows on retry (disable with OSMO_CACHE_ENABLED=false)
OSMO_CACHE_ENABLED="${OSMO_CACHE_ENABLED:-true}"
CACHED_RESULTS_FILE="" # set later after CI_PROJECT_DIR is available

# Python cache CLI — lives in the installable package tree so it can be
# properly tested and imported. Run this script from the package root so uv
# resolves the intended pyproject instead of an enclosing monorepo project.
OSMO_CACHE_CLI=(uv run --active -m groot.infra.ci.main.osmo_cache)
OSMO_WORKFLOW_OWNERS_CLI=(uv run --active -m groot.infra.ci.main.osmo_workflow_owners)

# Validate arguments
ZOMBIE_MODE=false
if [[ $# -eq 2 ]] && [[ "$2" = "--zombie" ]]; then
  ZOMBIE_MODE=true
elif [[ $# -ne 1 ]]; then
  echo "Usage: $0 <workflow_dir> [--zombie]"
  echo "  workflow_dir: directory containing list.template.yaml and workflow YAML files"
  exit 1
fi

WORKFLOW_DIR="${1%/}"
WORKFLOW_LABEL="${OSMO_WORKFLOW_LABEL:-$(basename "${WORKFLOW_DIR}" | sed 's/_workflows$//')}"
if [[ ! -d "${WORKFLOW_DIR}" ]]; then
  echo "❌ Workflow directory does not exist: ${WORKFLOW_DIR}"
  exit 1
fi
export OSMO_AWS_CREDENTIAL_NAME="${OSMO_AWS_CREDENTIAL_NAME:-gear-aws-cred-${CI_PIPELINE_ID:-manual}-${CI_JOB_ID:-manual}}"

# Render workflow list from template
TEMPLATE_FILE="${WORKFLOW_DIR}/list.template.yaml"
RENDERED_FILE="${WORKFLOW_DIR}/list_rendered.yaml"

OSMO_POOL=${OSMO_POOL} IMAGE=${IMAGE} envsubst < "${PWD}/${TEMPLATE_FILE}" > "${PWD}/${RENDERED_FILE}"

# Tach filter: filter or validate workflows and write metrics
TACH_FILTERED_FILE="${CI_PROJECT_DIR}/list_tach_filtered.yaml"
TACH_METRICS_FILE="${CI_PROJECT_DIR}/tach_filter_result.json"
TACH_SCHEDULE_ARGS=()
if [[ "${POSTMERGE_TESTS:-}" = "true" ]]; then
  TACH_SCHEDULE_ARGS=(--schedule-type postmerge)
fi
set +e
if [[ "${CI_PIPELINE_SOURCE:-}" = "merge_request_event" ]]; then
  # Filter mode: only run workflows affected by this MR
  uv run --active -m groot.infra.ci.main.osmo_test_picker \
    --workflow-dir "$(pwd)/${WORKFLOW_DIR}" \
    --input "$(pwd)/${RENDERED_FILE}" \
    --output "${TACH_FILTERED_FILE}" \
    --metrics-output "${TACH_METRICS_FILE}" \
    "${TACH_SCHEDULE_ARGS[@]}" \
    --base "${CI_MERGE_REQUEST_DIFF_BASE_SHA:-main}"
else
  # Validate mode: run all workflows but compute what would be skipped (for observability)
  uv run --active -m groot.infra.ci.main.osmo_test_picker \
    --workflow-dir "$(pwd)/${WORKFLOW_DIR}" \
    --input "$(pwd)/${RENDERED_FILE}" \
    --output "${TACH_FILTERED_FILE}" \
    --metrics-output "${TACH_METRICS_FILE}" \
    "${TACH_SCHEDULE_ARGS[@]}" \
    --validate \
    --base "HEAD~1"
fi
tach_exit=$?
set -e

if [[ ${tach_exit} -eq 2 ]]; then
  echo "⏭️  No workflows affected by this MR — skipping"
  exit 0
elif [[ ${tach_exit} -ne 0 ]]; then
  echo "⚠️  Tach filter failed (exit ${tach_exit}) — falling back to full list"
  cp "$(pwd)/${RENDERED_FILE}" "${TACH_FILTERED_FILE}"
fi
RENDERED_FILE="${TACH_FILTERED_FILE}"

# Extract RUN_ZOMBIE_TESTS value from commit message, e.g. "RUN_ZOMBIE_TESTS=test_a,test_b"
RUN_ZOMBIE_TESTS=$(echo "${CI_COMMIT_MESSAGE:-}" \
  | grep -oP '(?<=RUN_ZOMBIE_TESTS=)[^[:space:]]+' || true)

# Filter zombie/sloth workflows from DB state.
# In normal mode: exclude zombie/sloth (keep passing workflows).
# In zombie mode (--zombie): keep ONLY zombie/sloth (for postmerge zombie jobs).
FILTERED_FILE="${WORKFLOW_DIR}/list_filtered.yaml"
SHARD_FILE="${WORKFLOW_DIR}/list_sharded.yaml"
ZOMBIE_MODE_FLAG=""
if [[ "${ZOMBIE_MODE}" = "true" ]]; then
  ZOMBIE_MODE_FLAG="--zombie-mode"
fi
set +e
uv run --active -m groot.infra.ci.main.ci_state_filter \
  --test-type integration \
  --input "${RENDERED_FILE}" \
  --output "$(pwd)/${FILTERED_FILE}" \
  ${ZOMBIE_MODE_FLAG} \
  --run-zombie-tests "${RUN_ZOMBIE_TESTS}"
filter_exit=$?
set -e
if [[ "${filter_exit}" -eq 2 ]]; then
  echo "⏭️  No workflows to run for ${WORKFLOW_LABEL} (zombie_mode=${ZOMBIE_MODE}) — skipping"
  exit 0
elif [[ "${filter_exit}" -ne 0 ]]; then
  echo "⚠️  ci_state_filter failed (exit ${filter_exit}) — running all workflows"
  cp "${RENDERED_FILE}" "$(pwd)/${FILTERED_FILE}"
fi

if [[ -n "${OSMO_WORKFLOW_INCLUDE_REGEX:-}" ]]; then
  INCLUDE_FILTERED_FILE="${WORKFLOW_DIR}/list_include_filtered.yaml"
  set +e
  python3 - << PYEOF
import os
import re
import sys

input_path = "$(pwd)/${FILTERED_FILE}"
output_path = "$(pwd)/${INCLUDE_FILTERED_FILE}"
pattern = re.compile(os.environ["OSMO_WORKFLOW_INCLUDE_REGEX"])

with open(input_path) as f:
    lines = f.readlines()

blocks, current_key, current_lines = {}, None, []
for line in lines:
    if line and not line[0].isspace() and not line.startswith("#") and ":" in line:
        if current_key is not None:
            blocks[current_key] = current_lines
        current_key = line.split(":", 1)[0].strip()
        current_lines = [line]
    elif current_key is not None:
        current_lines.append(line)
if current_key is not None:
    blocks[current_key] = current_lines

env_block = blocks.pop("environment", None)
workflow_keys = [key for key in blocks if pattern.search(key)]
if not workflow_keys:
    print(f"No workflows matched OSMO_WORKFLOW_INCLUDE_REGEX={pattern.pattern!r}", file=sys.stderr)
    sys.exit(2)

output = []
if env_block:
    output.extend(env_block)
    output.append("\n")
for key in workflow_keys:
    output.extend(blocks[key])
    output.append("\n")

with open(output_path, "w") as f:
    f.write("".join(output))

print(
    f"Workflow include filter kept {len(workflow_keys)}/{len(blocks)} workflows: "
    + ", ".join(workflow_keys),
    file=sys.stderr,
)
PYEOF
  include_filter_exit=$?
  set -e
  if [[ "${include_filter_exit}" -eq 2 ]]; then
    echo "⏭️  No workflows matched OSMO_WORKFLOW_INCLUDE_REGEX=${OSMO_WORKFLOW_INCLUDE_REGEX} — skipping"
    exit 0
  elif [[ "${include_filter_exit}" -ne 0 ]]; then
    echo "❌ Workflow include filter failed with exit code ${include_filter_exit}"
    exit "${include_filter_exit}"
  fi
  FILTERED_FILE="${INCLUDE_FILTERED_FILE}"
fi

# Shard workflows: distribute across CI_NODE_TOTAL parallel runners by index
# If CI_NODE_INDEX / CI_NODE_TOTAL are unset, all workflows are included (no sharding)
set +e
python3 - << PYEOF
import os, sys

filtered_path = "$(pwd)/${FILTERED_FILE}"
shard_file_path = "$(pwd)/${SHARD_FILE}"

node_index = int(os.environ.get("CI_NODE_INDEX", "1")) - 1  # convert to 0-based
node_total = int(os.environ.get("CI_NODE_TOTAL", "1"))
sharding_active = node_total > 1

with open(filtered_path) as f:
    lines = f.readlines()

# Parse top-level YAML blocks (same line-by-line approach as zombie filter)
blocks, current_key, current_lines = {}, None, []
for line in lines:
    if line and not line[0].isspace() and not line.startswith('#') and ':' in line:
        if current_key is not None:
            blocks[current_key] = current_lines
        current_key = line.split(':')[0].strip()
        current_lines = [line]
    elif current_key is not None:
        current_lines.append(line)
if current_key is not None:
    blocks[current_key] = current_lines

env_block = blocks.pop('environment', None)
workflow_keys = list(blocks.keys())
shard_keys = [k for i, k in enumerate(workflow_keys) if i % node_total == node_index]

if not shard_keys and sharding_active:
    print(f"Shard {node_index+1}/{node_total}: no workflows assigned — skipping", file=sys.stderr)
    sys.exit(2)  # sentinel: nothing to run for this shard

output = []
if env_block:
    output.extend(env_block)
    output.append('\n')
for key in shard_keys:
    output.extend(blocks[key])
    output.append('\n')

shard_label = f"{node_index+1}/{node_total}" if sharding_active else "all"
print(f"Shard {shard_label}: {len(shard_keys)}/{len(workflow_keys)} workflows", file=sys.stderr)

with open(shard_file_path, 'w') as f:
    f.write(''.join(output))
PYEOF
shard_exit=$?
set -e
if [[ "${shard_exit}" -eq 2 ]]; then
  echo "⏭️  Shard ${CI_NODE_INDEX:-1}/${CI_NODE_TOTAL:-1}: no workflows assigned — skipping"
  exit 0
elif [[ "${shard_exit}" -ne 0 ]]; then
  echo "❌ Shard filter script failed with exit code ${shard_exit}"
  exit "${shard_exit}"
fi

echo "--------- Sharded ${SHARD_FILE} ---------"
cat "${PWD}/${SHARD_FILE}"
echo "-------------------------------------"

if [[ "${POSTMERGE_TESTS:-}" = "true" ]]; then
  echo "🔎 Applying 24h postmerge result rate limit..."
  set +e
  uv run --active -m groot.infra.ci.main.postmerge_test_rate_limit filter \
    --input "${PWD}/${SHARD_FILE}" \
    --output "${PWD}/${SHARD_FILE}" \
    --window-hours "${POSTMERGE_TEST_RATE_LIMIT_HOURS:-24}"
  postmerge_rate_limit_exit=$?
  set -e

  if [[ "${postmerge_rate_limit_exit}" -eq 2 ]]; then
    echo "⏭️  All postmerge workflows in this shard already produced a result recently — skipping"
    exit 0
  elif [[ "${postmerge_rate_limit_exit}" -ne 0 ]]; then
    echo "❌ Postmerge rate-limit filter failed with exit code ${postmerge_rate_limit_exit}"
    exit "${postmerge_rate_limit_exit}"
  fi
fi

# --- Result caching: skip workflows that already passed for this commit ---
CACHED_RESULTS_FILE="${CI_PROJECT_DIR}/cached_results.json"
echo "[]" > "${CACHED_RESULTS_FILE}"
RUN_FILE="${WORKFLOW_DIR}/list_to_run.yaml"
RUNNER_FILE="${WORKFLOW_DIR}/list_runner.yaml"

if [[ "${OSMO_CACHE_ENABLED}" = "true" ]] && [[ "${CI_COMMIT_SHA:-}" != "" ]]; then
  echo "🔍 Checking S3 cache for previously passed workflows (commit=${CI_COMMIT_SHA:0:12})..."

  set +e
  "${OSMO_CACHE_CLI[@]}" check-cache \
    --shard-file "${PWD}/${SHARD_FILE}" \
    --run-file "${PWD}/${RUN_FILE}" \
    --cached-results-file "${CACHED_RESULTS_FILE}" \
    --commit-sha "${CI_COMMIT_SHA}"
  cache_check_exit=$?
  set -e

  if [[ "${cache_check_exit}" -eq 2 ]]; then
    echo "⏭️  All workflows in this shard already passed (cached). Generating log.json from cache."
    "${OSMO_CACHE_CLI[@]}" generate-log \
      --cached-results-file "${CACHED_RESULTS_FILE}" \
      --log-file "${CI_PROJECT_DIR}/log.json"
    echo "✅ All tests passed (from cache): $(jq -r '.integration.passed_workflows' "${CI_PROJECT_DIR}"/log.json)/$(jq -r '.integration.total_workflows' "${CI_PROJECT_DIR}"/log.json)"
    exit 0
  elif [[ "${cache_check_exit}" -ne 0 ]]; then
    echo "⚠️  Cache check failed (exit=${cache_check_exit}), proceeding without cache"
    cp "${PWD}/${SHARD_FILE}" "${PWD}/${RUN_FILE}"
    echo "[]" > "${CACHED_RESULTS_FILE}"
  fi
else
  echo "ℹ️  Result caching disabled (OSMO_CACHE_ENABLED=${OSMO_CACHE_ENABLED})"
  cp "${PWD}/${SHARD_FILE}" "${PWD}/${RUN_FILE}"
fi

echo "--------- Workflows to run ---------"
cat "${PWD}/${RUN_FILE}"
echo "-------------------------------------"

"${OSMO_WORKFLOW_OWNERS_CLI[@]}" strip-runner-yaml \
  "${PWD}/${RUN_FILE}" \
  "${PWD}/${RUNNER_FILE}"

echo "--------- Runner workflows ---------"
cat "${PWD}/${RUNNER_FILE}"
echo "-------------------------------------"

# Set priority: high for nightly postmerge coverage, low for main branch, high for MRs
# CI_COMMIT_BRANCH may be unset in MR pipelines, so use ${VAR:-} syntax for it
if [[ "${POSTMERGE_TESTS:-}" = "true" ]]; then
  OSMO_PRIORITY=high
elif [[ "${CI_COMMIT_BRANCH:-}" = "${CI_DEFAULT_BRANCH}" ]]; then
  OSMO_PRIORITY=low
else
  OSMO_PRIORITY=high
fi

echo "CI_PROJECT_DIR: ${CI_PROJECT_DIR}"

# Track workflow IDs as they're submitted for cleanup on cancellation
WORKFLOW_IDS_FILE=${CI_PROJECT_DIR}/active_workflows.txt
rm -f "${WORKFLOW_IDS_FILE}"
touch "${WORKFLOW_IDS_FILE}"

# Login to OSMO (required before osmo_runner can call osmo CLI)
osmo login --method token --token "${OSMO_TOKEN}" "${OSMO_URL}"
osmo credential set "${OSMO_AWS_CREDENTIAL_NAME}" --type GENERIC --payload \
  aws_access_key_id="${BUCKET_ACCESS_KEY_ID}" \
  aws_secret_access_key="${BUCKET_SECRET_ACCESS_KEY}" \
  aws_endpoint_url="https://pdx.s8k.io" \
  aws_default_region="${BUCKET_REGION}"
for attempt in $(seq 1 12); do
  if osmo credential list | grep -F "${OSMO_AWS_CREDENTIAL_NAME}" >/dev/null; then
    echo "OSMO credential ${OSMO_AWS_CREDENTIAL_NAME} is visible"
    break
  fi
  if [[ "${attempt}" -eq 12 ]]; then
    echo "Timed out waiting for OSMO credential ${OSMO_AWS_CREDENTIAL_NAME} to become visible"
    exit 1
  fi
  sleep 5
done

# Capture workflow IDs from osmo_runner output while still showing logs
# Pattern matches lines like: [P] workflow_name [workflow-id-123]:
# Extracts only the workflow ID (second bracketed item)
# Use stdbuf to disable buffering so IDs are written immediately
osmo_runner \
  --workflows-base-dir "${CI_PROJECT_DIR}/${WORKFLOW_DIR}" \
  --list "${CI_PROJECT_DIR}/${RUNNER_FILE}" \
  --data-storage-access-key-id "${BUCKET_ACCESS_KEY_ID}" \
  --data-storage-access-key "${BUCKET_SECRET_ACCESS_KEY}" \
  --data-storage-endpoint "${BUCKET_ENDPOINT_URL}" \
  --data-storage-region "${BUCKET_REGION}" \
  --log_file "${CI_PROJECT_DIR}"/log.json \
  --kpi_file "${CI_PROJECT_DIR}"/kpis.json \
  --priority "${OSMO_PRIORITY}" \
  --username "${OSMO_USERNAME}" \
  --token "${OSMO_TOKEN}" \
  2>&1 | tee >(stdbuf -oL sed -un 's/^.*\] [^ ]* \[\([a-z0-9_-]*\)\]:.*/\1/p' >> "${WORKFLOW_IDS_FILE}")

# --- Cache write: persist COMPLETED results to S3, then merge with cached ---
if [[ "${OSMO_CACHE_ENABLED}" = "true" ]] && [[ "${CI_COMMIT_SHA:-}" != "" ]]; then
  echo "💾 Writing passed workflow results to S3 cache..."
  set +e
  "${OSMO_CACHE_CLI[@]}" write-cache \
    --log-file "${CI_PROJECT_DIR}/log.json" \
    --run-file "${PWD}/${RUN_FILE}" \
    --commit-sha "${CI_COMMIT_SHA}" \
    --pipeline-id "${CI_PIPELINE_IID:-}" \
    --job-id "${CI_JOB_ID:-}"
  set -e
fi

# --- Merge cached results with fresh results ---
echo "Checking workflow results..."

"${OSMO_CACHE_CLI[@]}" merge-results \
  --log-file "${CI_PROJECT_DIR}/log.json" \
  --cached-results-file "${CACHED_RESULTS_FILE}"
