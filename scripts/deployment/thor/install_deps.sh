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

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
    SUDO="sudo"
fi

can_sudo() {
    if [[ -z "$SUDO" ]]; then
        return 0
    fi
    $SUDO -n true >/dev/null 2>&1
}

has_cuda_sbsa_repo() {
    local pattern='developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa'
    grep -Rqs "$pattern" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null
}

if ! ldconfig -p | grep 'libnvpl_lapack_lp64_gomp.so.0' >/dev/null || ! dpkg -s libnuma1 >/dev/null 2>&1; then
    if ! can_sudo; then
        echo "ERROR: installing the Thor Torch runtime dependencies NVPL/NUMA requires sudo" >&2
        exit 1
    fi
    if ! apt-cache show libnvpl-lapack0 >/dev/null 2>&1; then
        if ! has_cuda_sbsa_repo; then
            keyring="$(mktemp /tmp/groot-nvpl-keyring.XXXXXX.deb)"
            trap 'rm -f "$keyring"' EXIT
            curl -fsSL \
                https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
                -o "$keyring"
            $SUDO dpkg -i "$keyring"
        fi
        $SUDO apt-get update
    fi
    $SUDO apt-get install -y --no-install-recommends libnvpl-blas0 libnvpl-lapack0 libnuma1
fi

unset UV_NO_SYNC
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
# uv resolves relative environment paths against the selected project.
if [[ "$VENV_DIR" != /* ]]; then
    VENV_DIR="$SCRIPT_DIR/$VENV_DIR"
fi
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --locked --project "$SCRIPT_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"
SITE_PKGS="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Repair incomplete CUDA installs before probing TorchCodec, which imports Torch.
if ! find "$SITE_PKGS/nvidia" -name 'libcudss.so.0' -print -quit 2>/dev/null | grep -q .; then
    echo "Repairing missing CUDA cuDSS runtime"
    CUDSS_VERSION="$("$VENV_PYTHON" -c 'from importlib.metadata import version; print(version("nvidia-cudss-cu13"))')"
    uv pip install --python "$VENV_PYTHON" --reinstall --no-deps "nvidia-cudss-cu13==$CUDSS_VERSION"
fi
if ! find "$SITE_PKGS/nvidia" -name 'libcudss.so.0' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: libcudss.so.0 is missing from the Thor environment after installation." >&2
    exit 1
fi

ensure_torchcodec_system_packages() {
    local package
    local missing_packages=()
    for package in "$@"; do
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    if [[ "${#missing_packages[@]}" -eq 0 ]]; then
        return
    fi
    if ! can_sudo; then
        echo "ERROR: TorchCodec requires these system packages; install with sudo: ${missing_packages[*]}" >&2
        exit 1
    fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends "${missing_packages[@]}"
}

ensure_torchcodec_system_packages ffmpeg
NVIDIA_LIB_DIRS="$(find "$SITE_PKGS/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

# The Jetson wheel may target a different system FFmpeg major version.
TORCHCODEC_VERSION=""
if ! "$VENV_PYTHON" -c 'from torchcodec.decoders import VideoDecoder' >/dev/null 2>&1; then
    echo "Building TorchCodec for the installed Torch and FFmpeg"
    ensure_torchcodec_system_packages \
        build-essential cmake git pkg-config python3-dev pybind11-dev \
        libavcodec-dev libavdevice-dev libavformat-dev libavfilter-dev libavutil-dev \
        libswresample-dev libswscale-dev
    TORCHCODEC_VERSION="$("$VENV_PYTHON" -c 'from importlib.metadata import version; print(version("torchcodec"))')"
    uv pip install --python "$VENV_PYTHON" --no-deps setuptools wheel pybind11
    (
        torchcodec_source="$(mktemp -d /tmp/groot-torchcodec.XXXXXX)"
        trap 'rm -rf "$torchcodec_source"' EXIT
        git clone --depth 1 --branch "v$TORCHCODEC_VERSION" https://github.com/pytorch/torchcodec.git "$torchcodec_source"
        cd "$torchcodec_source"
        BUILD_VERSION="$TORCHCODEC_VERSION" I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
            CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
            uv pip install --python "$VENV_PYTHON" . --no-build-isolation --no-deps
    )
fi
if ! "$VENV_PYTHON" -c 'from torchcodec.decoders import VideoDecoder'; then
    echo "ERROR: TorchCodec cannot load after installation; check the Torch/FFmpeg libraries above." >&2
    exit 1
fi
if [[ -n "${TORCHCODEC_VERSION:-}" ]]; then
    touch "$VENV_DIR/.groot-torchcodec-rebuilt"
fi

echo "Thor install complete. Activate it with:"
echo "  source .venv/bin/activate"
echo "  source scripts/deployment/thor/activate_thor.sh"
