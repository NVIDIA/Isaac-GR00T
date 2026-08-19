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
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
if [[ -d "${REPO_ROOT}/oss/skills/autoresearch" ]]; then
  PACKAGE_ROOT="${REPO_ROOT}/oss"
else
  PACKAGE_ROOT="${REPO_ROOT}"
fi
PYTHON="${PYTHON:-${PACKAGE_ROOT}/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${PACKAGE_ROOT}/skills/autoresearch/examples/yam_stack_bowl/eval_config.yaml}"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:?Set AUTORESEARCH_ROOT to a writable shared directory}"
RUN_ROOT="${RUN_ROOT:-${AUTORESEARCH_ROOT}/yam_stack_bowl/baseline}"
MODEL_PATH="${CHECKPOINT_PATH:-${MODEL_PATH:-${RUN_ROOT}/training}}"
EVAL_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/eval}"
RESULT_PATH="${RESULT_PATH:-${EVAL_DIR}/results.json}"

mkdir -p "${EVAL_DIR}"
exec 9>"${RUN_ROOT}/.eval.lock"
flock 9
if [[ -s "${RESULT_PATH}" ]]; then
  printf '[yam_stack_bowl_eval] reusing completed result: %s\n' "${RESULT_PATH}"
  exit 0
fi
export REPO_ROOT PYTHON

REPO_ROOT="${PACKAGE_ROOT}" bash "${PACKAGE_ROOT}/skills/autoresearch/scripts/eval_local.sh" \
  "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-json "${RESULT_PATH}" \
  --save-plot-dir "${EVAL_DIR}/plots"
