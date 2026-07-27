#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prune stale Colossus Ray job checkouts and unused runtime images.

set -euo pipefail

job_root="${COLOSSUS_JOB_ROOT:-/data/groot-ol-jobs}"
retention_hours="${COLOSSUS_JOB_RETENTION_HOURS:-168}"

[[ "${job_root}" == /* && "${job_root}" != "/" && "${job_root##*/}" == "groot-ol-jobs" ]] || {
  echo "COLOSSUS_JOB_ROOT must be an absolute path ending in /groot-ol-jobs" >&2
  exit 1
}
case "${retention_hours}" in
  ""|*[!0-9]*) echo "COLOSSUS_JOB_RETENTION_HOURS must be a positive integer" >&2; exit 1 ;;
esac
(( retention_hours > 0 )) || {
  echo "COLOSSUS_JOB_RETENTION_HOURS must be greater than zero" >&2
  exit 1
}

podman image prune --all --force --filter "until=${retention_hours}h"

[[ -d "${job_root}" ]] || exit 0

declare -A active_submissions=()
while IFS= read -r container_id; do
  [[ -n "${container_id}" ]] || continue
  while IFS= read -r value; do
    if [[ "${value}" == SUBMISSION_ID=* ]]; then
      active_submissions["${value#SUBMISSION_ID=}"]=1
    fi
  done < <(podman inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_id}")
done < <(podman ps --quiet)

retention_minutes=$((retention_hours * 60))
while IFS= read -r -d '' checkout; do
  submission_id="${checkout##*/}"
  if [[ -n "${active_submissions[${submission_id}]:-}" ]]; then
    continue
  fi
  rm -rf -- "${checkout}"
done < <(
  find "${job_root}" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -mmin "+${retention_minutes}" \
    -print0
)
