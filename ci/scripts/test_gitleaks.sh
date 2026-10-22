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

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
gitleaks_bin="${GITLEAKS_BIN:-gitleaks}"
config_path="${GITLEAKS_CONFIG:-${repo_root}/gitleaks.toml}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

fail() {
  echo "gitleaks self-test failed: $*" >&2
  exit 1
}

expect_nvidia_leak() {
  local log_path="$1"
  shift

  set +e
  "${gitleaks_bin}" "$@" >"${log_path}" 2>&1
  local status=$?
  set -e

  if [[ ${status} -ne 1 ]]; then
    cat "${log_path}" >&2
    fail "expected exit code 1, got ${status}"
  fi
  if ! grep -q "nvidia-api-key" "${log_path}"; then
    cat "${log_path}" >&2
    fail "expected the nvidia-api-key rule to match"
  fi
}

command -v "${gitleaks_bin}" >/dev/null || fail "gitleaks executable not found"
[[ -f "${config_path}" ]] || fail "config not found: ${config_path}"

# Construct test-only keys at runtime so the fixture itself is not a finding.
key_prefix="nvapi"
synthetic_key="${key_prefix}-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
second_synthetic_key="${key_prefix}-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
third_synthetic_key="${key_prefix}-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"

# The custom NVIDIA rule must reject a newly introduced key.
printf 'api_key = "%s"\n' "${synthetic_key}" >"${tmp_dir}/nvidia-key.txt"
expect_nvidia_leak \
  "${tmp_dir}/rule-detection.log" \
  dir --config "${config_path}" --redact --verbose --no-banner "${tmp_dir}/nvidia-key.txt"

# Vendored paths explicitly covered by the config must remain ignored.
rm "${tmp_dir}/nvidia-key.txt"
mkdir -p "${tmp_dir}/external_dependencies/vendor"
printf 'api_key = "%s"\n' "${synthetic_key}" \
  >"${tmp_dir}/external_dependencies/vendor/fixture.txt"
"${gitleaks_bin}" dir \
  --config "${config_path}" \
  --redact \
  --no-banner \
  "${tmp_dir}" \
  >"${tmp_dir}/allowlist.log" 2>&1

# A commit-range scan must ignore a pre-existing key while rejecting a key
# introduced after the selected base commit.
range_repo="${tmp_dir}/range-repo"
git init -q "${range_repo}"
git -C "${range_repo}" config user.email "gitleaks-test@nvidia.com"
git -C "${range_repo}" config user.name "Gitleaks Test"

printf 'api_key = "%s"\n' "${synthetic_key}" >"${range_repo}/existing-key.txt"
git -C "${range_repo}" add existing-key.txt
git -C "${range_repo}" commit -qm "Add baseline fixture"
base_sha="$(git -C "${range_repo}" rev-parse HEAD)"

printf 'safe fixture\n' >"${range_repo}/clean.txt"
git -C "${range_repo}" add clean.txt
git -C "${range_repo}" commit -qm "Add clean change"
"${gitleaks_bin}" git \
  --config "${config_path}" \
  --log-opts="${base_sha}..HEAD" \
  --redact \
  --no-banner \
  "${range_repo}" \
  >"${tmp_dir}/baseline-range.log" 2>&1

printf 'api_key = "%s"\n' "${second_synthetic_key}" >"${range_repo}/new-key.txt"
git -C "${range_repo}" add new-key.txt
git -C "${range_repo}" commit -qm "Add new key fixture"
expect_nvidia_leak \
  "${tmp_dir}/new-key-range.log" \
  git --config "${config_path}" --log-opts="${base_sha}..HEAD" --redact --verbose --no-banner "${range_repo}"

# A commit-list scan, as used for MRs, must reject a key that a later commit
# removed, and must not walk into the listed commits' ancestors.
printf 'api_key = "%s"\n' "${third_synthetic_key}" >"${range_repo}/removed-key.txt"
git -C "${range_repo}" add removed-key.txt
git -C "${range_repo}" commit -qm "Add removed key fixture"
added_sha="$(git -C "${range_repo}" rev-parse HEAD)"
git -C "${range_repo}" rm -q removed-key.txt
git -C "${range_repo}" commit -qm "Remove key fixture"
removed_sha="$(git -C "${range_repo}" rev-parse HEAD)"
"${gitleaks_bin}" git \
  --config "${config_path}" \
  --log-opts="--no-walk --diff-merges=first-parent ${removed_sha}" \
  --redact \
  --no-banner \
  "${range_repo}" \
  >"${tmp_dir}/no-walk-clean.log" 2>&1
expect_nvidia_leak \
  "${tmp_dir}/no-walk-removed-key.log" \
  git --config "${config_path}" \
  --log-opts="--no-walk --diff-merges=first-parent ${removed_sha} ${added_sha}" \
  --redact --verbose --no-banner "${range_repo}"

# A key introduced only while resolving a merge conflict must be scanned,
# even if a later commit removes it. Neither parent contains this key.
merge_repo="${tmp_dir}/merge-repo"
git init -q --initial-branch=main "${merge_repo}"
git -C "${merge_repo}" config user.email "gitleaks-test@nvidia.com"
git -C "${merge_repo}" config user.name "Gitleaks Test"
printf 'api_key = "base"\n' >"${merge_repo}/config.txt"
git -C "${merge_repo}" add config.txt
git -C "${merge_repo}" commit -qm "Add merge fixture"
git -C "${merge_repo}" checkout -qb feature
printf 'api_key = "feature"\n' >"${merge_repo}/config.txt"
git -C "${merge_repo}" commit -qam "Change feature config"
git -C "${merge_repo}" checkout -q main
printf 'api_key = "main"\n' >"${merge_repo}/config.txt"
git -C "${merge_repo}" commit -qam "Change main config"
git -C "${merge_repo}" checkout -q feature
if git -C "${merge_repo}" merge --no-edit main >"${tmp_dir}/merge.log" 2>&1; then
  fail "expected a merge conflict"
fi
git -C "${merge_repo}" rev-parse --verify MERGE_HEAD >/dev/null || fail "expected a pending merge"
printf 'api_key = "%s"\n' "${third_synthetic_key}" >"${merge_repo}/config.txt"
git -C "${merge_repo}" add config.txt
git -C "${merge_repo}" commit -qm "Resolve conflict with key fixture"
merge_sha="$(git -C "${merge_repo}" rev-parse HEAD)"
printf 'api_key = "removed"\n' >"${merge_repo}/config.txt"
git -C "${merge_repo}" commit -qam "Remove merge key fixture"
removed_merge_sha="$(git -C "${merge_repo}" rev-parse HEAD)"
expect_nvidia_leak \
  "${tmp_dir}/merge-key.log" \
  git --config "${config_path}" \
  --log-opts="--no-walk --diff-merges=first-parent ${removed_merge_sha} ${merge_sha}" \
  --redact --verbose --no-banner "${merge_repo}"

echo "Gitleaks configuration self-tests passed."
