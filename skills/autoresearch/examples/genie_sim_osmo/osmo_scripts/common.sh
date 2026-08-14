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

log_section() {
  echo
  echo "################################################################################"
  echo "# $1"
  echo "################################################################################"
}


wait_for_osmo_workflow() {
  local osmo="$1"
  local workflow_id="$2"
  local poll_interval_s="$3"
  local query_json status last_status=""
  while true; do
    if ! query_json="$("${osmo}" workflow query "${workflow_id}" --format-type json 2>/dev/null)"; then
      printf '[osmo] workflow query failed; retrying: %s\n' "${workflow_id}"
      sleep "${poll_interval_s}"
      continue
    fi
    status="$(printf '%s' "${query_json}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("status", ""))')"
    if [[ "${status}" != "${last_status}" ]]; then
      printf '[osmo] workflow %s: %s\n' "${workflow_id}" "${status:-UNKNOWN}"
      last_status="${status}"
    fi
    case "${status}" in
      COMPLETED|SUCCEEDED)
        return 0
        ;;
      FAILED|FAILED_CANCELED|CANCELED|CANCELLED)
        printf 'OSMO workflow %s failed with status %s\n' "${workflow_id}" "${status}" >&2
        return 1
        ;;
    esac
    sleep "${poll_interval_s}"
  done
}
