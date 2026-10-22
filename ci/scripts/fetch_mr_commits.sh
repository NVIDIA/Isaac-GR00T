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

# Fetch exactly the commits GitLab lists for the current MR, each with its
# parents so its diff can be scanned, and print their SHAs on one line.
#
# Deepening a shallow clone until the base is reachable from the head is not
# enough: after main is merged into a long branch, the base is reachable
# through the merge while the branch's early commits are still missing.

set -euo pipefail

: "${GITLAB_GR00T_PROJECT_TOKEN:?GITLAB_GR00T_PROJECT_TOKEN is required to list the MR commits}"
commits_url="${CI_API_V4_URL}/projects/${CI_MERGE_REQUEST_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/commits"
head="${CI_COMMIT_SHA}"

# The gitleaks image ships busybox wget; the job installs jq before this script.
list_commits() {
  local page=1 response ids
  while :; do
    response="$(wget -q -T 30 -O- \
      --header "PRIVATE-TOKEN: ${GITLAB_GR00T_PROJECT_TOKEN}" \
      "${commits_url}?per_page=100&page=${page}")" || return
    # Only an empty JSON array ends pagination. Reject malformed pages and
    # invalid commit entries so we never return a partial scan list as success.
    ids="$(jq -ers '
      if length == 1 and (.[0] | type) == "array" then .[0]
      else error("expected one JSON array") end
      | map(
          if type != "object" then error("expected a commit object")
          elif (.id | type) != "string" then error("expected a commit ID")
          elif (.id | test("^[0-9a-f]{40}$") | not) then error("invalid commit ID")
          else .id end
        )
      | join("\n")
    ' <<<"${response}")" || {
      echo "Invalid MR commit list response on page ${page}" >&2
      return 1
    }
    [[ -n "${ids}" ]] || return 0
    echo "${ids}"
    page=$((page + 1))
  done
}

# GitLab refreshes the MR's commit list asynchronously after a push.
for attempt in 1 2 3 4 5; do
  commits="$(list_commits)"
  if grep -qx "${head}" <<<"${commits}"; then
    break
  fi
  if [[ ${attempt} -eq 5 ]]; then
    echo "MR !${CI_MERGE_REQUEST_IID} commit list does not include ${head}" >&2
    exit 1
  fi
  sleep 10
done

shas=()
while read -r sha; do
  shas+=("${sha}")
done <<<"${commits}"

git fetch --no-tags --quiet --depth=2 origin "${shas[@]}"
echo "${shas[*]}"
