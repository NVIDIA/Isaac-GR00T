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
case "$ARCH" in
    aarch64 | arm64) ARCH=aarch64 ;;
    x86_64 | amd64) ARCH=x86_64 ;;
esac

# ──────────────────────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────────────────────

# FFmpeg runtime/build libs — required by torchcodec on aarch64
# libaio-dev — required by deepspeed async I/O ops
# git-lfs — retrieves the repository-provided aarch64 torchcodec wheel
echo "Installing system dependencies..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git-lfs \
    libaio-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev

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

if [ "$ARCH" = "aarch64" ]; then
    git lfs install --local
    if ! git lfs pull \
        --include="scripts/deployment/dgpu/wheels/torchcodec-*.whl" \
        --exclude=""; then
        echo "WARNING: Git LFS wheel download failed; falling back to a source build." >&2
    fi
fi
echo "Preparing the aarch64 torchcodec wheel required by uv resolution..."
bash "$SCRIPT_DIR/bootstrap_wheels.sh"

echo "Running uv sync..."
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    uv sync
else
    echo "Skipping flash-attn; GR00T inference will use PyTorch SDPA."
    uv sync --no-install-package flash-attn
fi

echo ""
echo "Install complete! Activate with:"
echo "  source .venv/bin/activate"
