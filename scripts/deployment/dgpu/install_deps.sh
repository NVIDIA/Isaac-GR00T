#!/bin/bash
# install_deps.sh — One-time install of GR00T deps on dGPU systems (x86_64 or aarch64 GB200, CUDA 12.8+)
# Requires an NVIDIA discrete GPU with a CUDA 12.x or 13.x driver already installed.
# After install, activate with: source .venv/bin/activate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Use sudo only when not already root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

ARCH=$(uname -m)

# ──────────────────────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────────────────────

# FFmpeg runtime libs — required by torchcodec at runtime
# libaio-dev — required by deepspeed async I/O ops
echo "Installing system dependencies..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends ffmpeg libaio-dev

# CUDA toolkit — required by deepspeed (needs CUDA_HOME / nvcc to check op compatibility)
# Skip if already installed
if [ ! -d "/usr/local/cuda" ]; then
    echo "CUDA toolkit not found. Installing cuda-toolkit-12-8..."
    # Add NVIDIA CUDA apt repo if not already configured
    if ! apt-cache show cuda-toolkit-12-8 &>/dev/null; then
        UBUNTU_VERSION=$(. /etc/os-release && echo "${VERSION_ID//.}")
        # aarch64 GB200 uses the sbsa (server base system architecture) repo
        if [ "$ARCH" = "aarch64" ]; then
            CUDA_REPO_ARCH="sbsa"
        else
            CUDA_REPO_ARCH="x86_64"
        fi
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CUDA_REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
        echo "Adding NVIDIA CUDA apt repository..."
        curl -fsSL "$KEYRING_URL" -o /tmp/cuda-keyring.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
    fi
    $SUDO apt-get install -y --no-install-recommends cuda-toolkit-12-8
else
    echo "CUDA toolkit already installed at /usr/local/cuda."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Python environment
# ──────────────────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

echo "Running uv sync (torch==2.7.1+cu128 from pytorch-cu128 index)..."
uv sync

echo "Installing package in editable mode..."
uv pip install -e .

# ──────────────────────────────────────────────────────────────────────────────
# torchcodec — build from source on aarch64 (no Linux aarch64 wheel on PyPI)
# ──────────────────────────────────────────────────────────────────────────────
if [ "$ARCH" = "aarch64" ]; then
    echo "Building torchcodec from source (aarch64)..."
    $SUDO apt-get install -y --no-install-recommends \
        libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
        libavutil-dev libswresample-dev libswscale-dev \
        pkg-config cmake build-essential pybind11-dev python3-dev
    TORCHCODEC_GIT_URL="https://github.com/meta-pytorch/torchcodec.git"
    TORCHCODEC_GIT_REF="release/0.4"
    rm -rf /tmp/torchcodec
    git clone --depth 1 --branch "${TORCHCODEC_GIT_REF}" "${TORCHCODEC_GIT_URL}" /tmp/torchcodec
    cd /tmp/torchcodec
    VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
    ENABLE_CUDA=1 uv pip install --python "$VENV_DIR/bin/python" . --no-build-isolation
    cd "$REPO_ROOT" && rm -rf /tmp/torchcodec
    echo "torchcodec built and installed."
fi

echo ""
echo "Install complete! Activate with:"
echo "  source .venv/bin/activate"
