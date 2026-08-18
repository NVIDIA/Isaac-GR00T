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
. "${SCRIPT_DIR}/common.sh"

log_section "Prepare training output"
TRAINING_OUTPUT_DIR="${RUN_ROOT}/training"
mkdir -p "${TRAINING_OUTPUT_DIR}" "${OSMO_OUTPUT_DIR}"
trap 'cp "${TRAINING_OUTPUT_DIR}/train.log" "${OSMO_OUTPUT_DIR}/train.log" 2>/dev/null || true' EXIT
nvidia-smi

if [[ ! -f "${TRAINING_CONFIG}" ]]; then
  echo "Training config does not exist: ${TRAINING_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${TRAIN_DATASET}" ]]; then
  echo "Training dataset does not exist: ${TRAIN_DATASET}" >&2
  exit 2
fi

log_section "Set up training environment"
export UV_CACHE_DIR=/tmp/uv-cache
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
bash skills/autoresearch/scripts/setup_env.sh 2>&1 | tee "${TRAINING_OUTPUT_DIR}/setup_env.log"
PYTHON="${PWD}/.venv/bin/python"
uv pip freeze --python "${PYTHON}" > "${TRAINING_OUTPUT_DIR}/python-freeze.txt"

log_section "Prepare training config"
RUN_CONFIG="${TRAINING_OUTPUT_DIR}/train_config.yaml"
"${PYTHON}" "${SCRIPT_DIR}/prepare_train_config.py" \
  "${TRAINING_CONFIG}" "${TRAIN_DATASET}" "${RUN_CONFIG}"

log_section "Train for a few steps"
export REPO_ROOT="${PWD}"
export PYTHON
export DS_IGNORE_CUDA_DETECTION=1
export ENABLE_TENSORRT=True
export SKIP_EXPERIMENT_REGISTRATION=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/amlfs-02/shared/ckpts}"
export HF_HUB_OFFLINE=1
export GIT_LFS_SKIP_SMUDGE=1
export OTEL_SDK_DISABLED=true
export WANDB_MODE=disabled
export WANDB_API_KEY=disabled
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
unset RAY_ADDRESS

set +e
bash skills/autoresearch/scripts/train_local.sh \
  "${RUN_CONFIG}" \
  --training.output-dir "${TRAINING_OUTPUT_DIR}" \
  2>&1 | tee "${TRAINING_OUTPUT_DIR}/train.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e
if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  exit "${TRAIN_EXIT}"
fi

log_section "Remove resumable checkpoints after final model save"
find "${TRAINING_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +

log_section "Record train outputs"
find "${TRAINING_OUTPUT_DIR}" -maxdepth 3 -type f | sort > "${TRAINING_OUTPUT_DIR}/train_files.txt"
cp "${TRAINING_OUTPUT_DIR}/python-freeze.txt" "${OSMO_OUTPUT_DIR}/python-freeze.txt"
cp "${TRAINING_OUTPUT_DIR}/train_files.txt" "${OSMO_OUTPUT_DIR}/train_files.txt"
