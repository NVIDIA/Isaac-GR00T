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

# The Thor-specific pyproject.toml and uv.lock are consumed in place from
# $SCRIPT_DIR via `uv sync --project` below — we no longer copy them over
# the repo root, which used to leave the working tree dirty after install.

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

# Install platform-specific deps from the Thor pyproject without mutating
# the repo-root pyproject.toml / uv.lock. See the Orin installer for
# details on UV_PROJECT_ENVIRONMENT + --no-install-project.
# Respect a pre-set UV_PROJECT_ENVIRONMENT from the Dockerfile
# (/opt/gr00t-venv, matched by the VIRTUAL_ENV + PATH ENV lines there);
# fall back to $REPO_ROOT/.venv on bare metal.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
echo "Running uv sync with the Thor pyproject at $SCRIPT_DIR (venv: $UV_PROJECT_ENVIRONMENT)..."
uv sync --project "$SCRIPT_DIR" --no-install-project

VENV_DIR="$UV_PROJECT_ENVIRONMENT"
VENV_PYTHON="$VENV_DIR/bin/python"
SITE_PKGS="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

echo "Installing gr00t in editable mode from the repo root (--no-deps)..."
uv pip install --python "$VENV_PYTHON" --no-deps -e "$REPO_ROOT"

# ──────────────────────────────────────────────────────────────────────────────
# torchcodec — prebuilt wheel (shared with Spark, both FFmpeg 6) or source build
# Thor and Spark share the same cp312 aarch64 wheel since both run Ubuntu 24.04
# with FFmpeg 6. The wheel lives under spark/wheels/.
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing FFmpeg runtime..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends ffmpeg

SPARK_DIR="$SCRIPT_DIR/../spark"
if [ ! -d "$SPARK_DIR/wheels" ]; then
    echo "Warning: Spark wheels directory not found at $SPARK_DIR/wheels — will attempt source build"
fi
TORCHCODEC_WHL=$(find "$SPARK_DIR/wheels" -name 'torchcodec-*.whl' -print -quit 2>/dev/null || true)
if [ -n "$TORCHCODEC_WHL" ]; then
    echo "Installing torchcodec from prebuilt wheel: $TORCHCODEC_WHL"
    uv pip install --python "$VENV_PYTHON" --force-reinstall --no-deps "$TORCHCODEC_WHL"
else
    echo "No prebuilt torchcodec wheel found — building from source..."
    $SUDO apt-get install -y --no-install-recommends \
        libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
        libavutil-dev libswresample-dev libswscale-dev \
        pkg-config pybind11-dev python3-dev
    uv pip install --python "$VENV_PYTHON" setuptools
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
fi

echo ""
echo "Install complete! In each new shell, activate with:"
echo "  source .venv/bin/activate"
echo "  source scripts/activate_thor.sh"
