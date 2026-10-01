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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: the Thor environment requires aarch64" >&2
    exit 1
fi
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
    echo "ERROR: the Thor environment requires Python 3.12" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! ldconfig -p | grep 'libnvpl_lapack_lp64_gomp.so.0' >/dev/null; then
    if ! sudo -n true 2>/dev/null; then
        echo "ERROR: installing the Thor Torch runtime dependency NVPL requires sudo" >&2
        exit 1
    fi
    if ! apt-cache show libnvpl-lapack0 >/dev/null 2>&1; then
        keyring="$(mktemp /tmp/groot-nvpl-keyring.XXXXXX.deb)"
        trap 'rm -f "$keyring"' EXIT
        curl -fsSL \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
            -o "$keyring"
        sudo dpkg -i "$keyring"
        sudo apt-get update
    fi
    sudo apt-get install -y --no-install-recommends libnvpl-blas0 libnvpl-lapack0
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
uv sync --locked --project "$SCRIPT_DIR"

echo "Thor install complete. Activate it with:"
echo "  source .venv/bin/activate"
echo "  source scripts/deployment/thor/activate_thor.sh"
