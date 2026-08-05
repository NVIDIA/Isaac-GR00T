#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prune stale Colossus Ray job checkouts and unused runtime images.

set -euo pipefail

job_root="${COLOSSUS_JOB_ROOT:-/data/groot-ol-jobs}"
retention_hours="${COLOSSUS_JOB_RETENTION_HOURS:-168}"
evaluation_checkpoint_root="${COLOSSUS_EVALUATION_CHECKPOINT_ROOT:-/data/groot_ol/evaluation_checkpoints}"
evaluation_checkpoint_retention_hours="${COLOSSUS_EVALUATION_CHECKPOINT_RETENTION_HOURS:-168}"
evaluation_video_root="${COLOSSUS_EVALUATION_VIDEO_ROOT:-/data/groot_ol/evaluation_videos}"
evaluation_video_retention_hours="${COLOSSUS_EVALUATION_VIDEO_RETENTION_HOURS:-168}"

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
[[ "${evaluation_checkpoint_root}" == /* \
  && "${evaluation_checkpoint_root}" != "/" \
  && "${evaluation_checkpoint_root##*/}" == "evaluation_checkpoints" ]] || {
  echo "COLOSSUS_EVALUATION_CHECKPOINT_ROOT must be an absolute path ending in /evaluation_checkpoints" >&2
  exit 1
}
case "${evaluation_checkpoint_retention_hours}" in
  ""|*[!0-9]*) echo "COLOSSUS_EVALUATION_CHECKPOINT_RETENTION_HOURS must be a positive integer" >&2; exit 1 ;;
esac
(( evaluation_checkpoint_retention_hours > 0 )) || {
  echo "COLOSSUS_EVALUATION_CHECKPOINT_RETENTION_HOURS must be greater than zero" >&2
  exit 1
}
[[ "${evaluation_video_root}" == /* \
  && "${evaluation_video_root}" != "/" \
  && "${evaluation_video_root##*/}" == "evaluation_videos" ]] || {
  echo "COLOSSUS_EVALUATION_VIDEO_ROOT must be an absolute path ending in /evaluation_videos" >&2
  exit 1
}
case "${evaluation_video_retention_hours}" in
  ""|*[!0-9]*) echo "COLOSSUS_EVALUATION_VIDEO_RETENTION_HOURS must be a positive integer" >&2; exit 1 ;;
esac
(( evaluation_video_retention_hours > 0 )) || {
  echo "COLOSSUS_EVALUATION_VIDEO_RETENTION_HOURS must be greater than zero" >&2
  exit 1
}
podman image prune --all --force --filter "until=${retention_hours}h"

retention_minutes=$((retention_hours * 60))
if [[ -d "${job_root}" ]]; then
  declare -A active_submissions=()
  while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] || continue
    while IFS= read -r value; do
      if [[ "${value}" == SUBMISSION_ID=* ]]; then
        active_submissions["${value#SUBMISSION_ID=}"]=1
      fi
    done < <(podman inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_id}")
  done < <(podman ps --quiet)

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
fi

if [[ -d "${evaluation_checkpoint_root}" ]]; then
  evaluation_retention_minutes=$((evaluation_checkpoint_retention_hours * 60))
  mkdir -p "${evaluation_checkpoint_root}/.locks"
  while IFS= read -r -d '' cache_identity; do
    digest="${cache_identity##*/}"
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || continue
    lock_path="${evaluation_checkpoint_root}/.locks/${digest}.lock"
    (
      flock --exclusive --nonblock 9 || exit 0
      [[ -d "${cache_identity}" ]] || exit 0
      if [[ -n "$(find "${cache_identity}" -maxdepth 0 -mmin "+${evaluation_retention_minutes}" -print -quit)" ]]; then
        rm -rf -- "${cache_identity}"
      fi
    ) 9>>"${lock_path}"
  done < <(
    find "${evaluation_checkpoint_root}" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -mmin "+${evaluation_retention_minutes}" \
      -print0
  )
fi

if [[ -d "${evaluation_video_root}" ]]; then
  evaluation_video_retention_minutes=$((evaluation_video_retention_hours * 60))
  find "${evaluation_video_root}" \
    -type f \
    -mmin "+${evaluation_video_retention_minutes}" \
    -delete
  find "${evaluation_video_root}" \
    -depth \
    -mindepth 1 \
    -type d \
    -empty \
    -delete
fi
