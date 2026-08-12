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

# launch_copy_to_object_store.sh — submit copy_to_object_store as a Ray job via gear
#
# Usage:
#   ./launch_copy_to_object_store.sh <workflow_id> <job_name> \
#       (--paths /mnt/aws-lfs-02/DatasetA ... | --paths-file paths.txt) \
#       (--pool groot-gb200-02 | --region us-east-1) \
#       [--depth N] [--walk-workers N] [--workers N] \
#       [--s3-concurrency N] [--rate-limit-mbps N] \
#       [--delete] [--dry-run]
#
# --paths-file is expanded locally into --paths path1 path2 ... before submission
# so the remote Ray job receives explicit paths only.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <workflow_id> <job_name> (--paths PATH ... | --paths-file FILE) (--pool POOL | --region REGION) [options]" >&2
  exit 1
fi

WORKFLOW_ID="$1"
JOB_NAME="$2"
shift 2

# Expand --paths-file FILE into --paths path1 path2 ...
EXPANDED_ARGS=()
args=("$@")
i=0
while [[ ${i} -lt ${#args[@]} ]]; do
  if [[ "${args[${i}]}" == "--paths-file" ]]; then
    i=$((i + 1))
    PATHS_FILE="${args[${i}]}"
    if [[ ! -f "${PATHS_FILE}" ]]; then
      echo "Error: paths file not found: ${PATHS_FILE}" >&2
      exit 1
    fi
    EXPANDED_ARGS+=("--paths")
    while IFS= read -r line; do
      [[ -z "${line}" || "${line}" == \#* ]] && continue
      EXPANDED_ARGS+=("${line}")
    done < "${PATHS_FILE}"
  else
    EXPANDED_ARGS+=("${args[${i}]}")
  fi
  i=$((i + 1))
done

uv run python scripts/launch_ray_job.py "${WORKFLOW_ID}" "${JOB_NAME}" \
  python -m groot.core.data.hydrator.migration.copy_to_object_store \
  "${EXPANDED_ARGS[@]}"
