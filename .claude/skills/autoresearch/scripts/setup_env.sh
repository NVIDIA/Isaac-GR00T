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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
if [[ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_FLASH_ARCHS=""
  while IFS="." read -r major minor; do
    flash_arch="${major}${minor}"
    case "${flash_arch}" in
      80|86|89|90|100|120) ;;
      *) continue ;;
    esac
    if [[ ";${DETECTED_FLASH_ARCHS};" != *";${flash_arch};"* ]]; then
      DETECTED_FLASH_ARCHS="${DETECTED_FLASH_ARCHS:+${DETECTED_FLASH_ARCHS};}${flash_arch}"
    fi
  done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
  if [[ -n "${DETECTED_FLASH_ARCHS}" ]]; then
    export FLASH_ATTN_CUDA_ARCHS="${DETECTED_FLASH_ARCHS}"
  fi
fi

printf '[setup_env] build settings: MAX_JOBS=%s NVCC_THREADS=%s FLASH_ATTN_CUDA_ARCHS=%s\n' \
  "${MAX_JOBS}" "${NVCC_THREADS}" "${FLASH_ATTN_CUDA_ARCHS:-default}"

# Sync all locked dependencies in case the public image does not provide them.
# Install Torch first, then build FlashAttention against that pinned Torch.
uv sync --locked --extra training --extra dev --no-install-package flash-attn
exec uv sync --locked --extra training --extra dev \
  --no-build-isolation-package flash-attn
