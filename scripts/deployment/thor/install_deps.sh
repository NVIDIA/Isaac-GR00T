#!/bin/bash
# install_deps.sh — One-time install of GR00T deps on Jetson AGX Thor (aarch64, Python 3.12)
# Used by both bare metal and scripts/deployment/thor/Dockerfile.
# After install, use `source scripts/activate_thor.sh` in each new shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Use sudo only when not already root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# Validate platform
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: This script is intended for aarch64 (Jetson Thor). Detected: $ARCH"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "WARNING: Expected Python 3.12 for Thor, detected Python $PYTHON_VERSION"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Copy Thor-specific pyproject.toml to repo root
# ──────────────────────────────────────────────────────────────────────────────
echo "Copying Thor pyproject.toml and uv.lock to repo root..."
cp "$SCRIPT_DIR/pyproject.toml" "$REPO_ROOT/pyproject.toml"
cp "$SCRIPT_DIR/uv.lock" "$REPO_ROOT/uv.lock"

# ──────────────────────────────────────────────────────────────────────────────
# NVPL LAPACK/BLAS — required by the jetson torch wheel
# ──────────────────────────────────────────────────────────────────────────────
if ! ldconfig -p | grep -q libnvpl_lapack_lp64_gomp; then
    echo "Installing NVPL libs (required by torch on aarch64)..."
    # Add NVIDIA CUDA apt repo if not already configured
    if ! apt-cache show libnvpl-lapack0 &>/dev/null; then
        echo "Adding NVIDIA CUDA apt repository..."
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
            -o /tmp/cuda-keyring.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update
    fi
    $SUDO apt-get install -y libnvpl-lapack0 libnvpl-blas0
else
    echo "NVPL libs already installed."
fi

# ──────────────────────────────────────────────────────────────────────────────
# CUDA dev packages — Thor BSP only ships runtime libs; cmake builds (e.g.
# torchcodec) need the compiler and headers.
# ──────────────────────────────────────────────────────────────────────────────
if ! dpkg -s cuda-nvcc-13-0 &>/dev/null; then
    echo "Installing CUDA dev packages (nvcc, cudart-dev, nvrtc-dev)..."
    $SUDO apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0
else
    echo "CUDA dev packages already installed."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Python environment
# ──────────────────────────────────────────────────────────────────────────────

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Running uv sync with Thor pyproject.toml..."
cd "$REPO_ROOT"
uv sync

VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
SITE_PKGS="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

echo "Installing package in editable mode..."
uv pip install --python "$VENV_PYTHON" -e .

# ──────────────────────────────────────────────────────────────────────────────
# torchcodec — build from source against system FFmpeg
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing FFmpeg runtime and dev libs for torchcodec build..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev \
    pkg-config pybind11-dev

echo "Ensuring setuptools is available for torchcodec build..."
uv pip install --python "$VENV_PYTHON" setuptools

echo "Building torchcodec from source (v0.10.0 against system FFmpeg)..."
# torchcodec needs PyTorch and NVIDIA runtime libs on LD_LIBRARY_PATH during build.
NVIDIA_LIB_DIRS="$(find "${SITE_PKGS}/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
rm -rf /tmp/torchcodec
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git /tmp/torchcodec
cd /tmp/torchcodec
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install --python "$VENV_PYTHON" . --no-build-isolation
cd - && rm -rf /tmp/torchcodec

echo ""
echo "Install complete! In each new shell, activate with:"
echo "  source .venv/bin/activate"
echo "  source scripts/activate_thor.sh"
