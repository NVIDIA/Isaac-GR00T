#!/bin/bash
# install_deps.sh — One-time install of GR00T deps on dGPU systems (x86_64 or aarch64 GB200, CUDA 13.0+)
# Requires an NVIDIA discrete GPU with a CUDA 13.x driver (580+) already installed.
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
    echo "CUDA toolkit not found. Installing cuda-toolkit-13-0..."
    # Add NVIDIA CUDA apt repo if not already configured
    if ! apt-cache show cuda-toolkit-13-0 &>/dev/null; then
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
    $SUDO apt-get install -y --no-install-recommends cuda-toolkit-13-0
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

echo "Running uv sync (torch==2.12.0+cu130 from pytorch-cu130 index, Python 3.12)..."
uv sync --python 3.12

echo "Installing package in editable mode..."
uv pip install -e .

if [ "$(uname -s)" = "Linux" ]; then
    if [ "$ARCH" = "aarch64" ]; then
        echo "Installing TensorRT (CUDA 13) for aarch64..."
        uv pip install --index-url https://pypi.nvidia.com "tensorrt-cu13==10.16.1.11"
    else
        echo "Installing TensorRT (CUDA 13) for x86_64..."
        uv pip install --index-url https://pypi.nvidia.com "tensorrt-cu13==10.16.1.11"
    fi
fi

echo ""
echo "Install complete! Activate with:"
echo "  source .venv/bin/activate"
