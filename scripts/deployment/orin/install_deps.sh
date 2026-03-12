#!/bin/bash
# install_deps.sh — One-time install of GR00T deps on Jetson Orin (aarch64, JetPack 6.2, Python 3.10)
# Used by both bare metal and scripts/deployment/orin/Dockerfile.
# After install, use `source scripts/activate_orin.sh` in each new shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TMP_DIR="$(mktemp -d)"
RESTORE_ROOT_FILES=0

restore_repo_root_files() {
    if [ "$RESTORE_ROOT_FILES" -ne 1 ]; then
        rm -rf "$TMP_DIR"
        return
    fi

    if [ -f "$TMP_DIR/pyproject.toml.orig" ]; then
        mv "$TMP_DIR/pyproject.toml.orig" "$REPO_ROOT/pyproject.toml"
    else
        rm -f "$REPO_ROOT/pyproject.toml"
    fi

    if [ -f "$TMP_DIR/uv.lock.orig" ]; then
        mv "$TMP_DIR/uv.lock.orig" "$REPO_ROOT/uv.lock"
    else
        rm -f "$REPO_ROOT/uv.lock"
    fi

    rm -rf "$TMP_DIR"
}

trap restore_repo_root_files EXIT

# Use sudo only when not already root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# Validate platform
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: This script is intended for aarch64 (Jetson Orin). Detected: $ARCH"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "WARNING: Expected Python 3.10 for Orin, detected Python $PYTHON_VERSION"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Copy Orin-specific pyproject.toml to repo root
# ──────────────────────────────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    cp "$REPO_ROOT/pyproject.toml" "$TMP_DIR/pyproject.toml.orig"
fi
if [ -f "$REPO_ROOT/uv.lock" ]; then
    cp "$REPO_ROOT/uv.lock" "$TMP_DIR/uv.lock.orig"
fi

RESTORE_ROOT_FILES=1
echo "Temporarily copying Orin pyproject.toml and uv.lock to repo root..."
cp "$SCRIPT_DIR/pyproject.toml" "$REPO_ROOT/pyproject.toml"
if [ -f "$SCRIPT_DIR/uv.lock" ]; then
    cp "$SCRIPT_DIR/uv.lock" "$REPO_ROOT/uv.lock"
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

echo "Running uv sync with Orin pyproject.toml..."
cd "$REPO_ROOT"
uv sync

VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
SITE_PKGS="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages"

echo "Installing package in editable mode..."
uv pip install --python "$VENV_PYTHON" -e .

# ──────────────────────────────────────────────────────────────────────────────
# nvidia-cudss-cu12 — needed by torch 2.10.0 at runtime (libcudss.so.0)
# Installed with --no-deps to avoid pulling in nvidia-cublas-cu12 which
# conflicts with the system CUDA 12.6 libs on JetPack 6.2.
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing nvidia-cudss-cu12 (no-deps to avoid system CUDA conflicts)..."
uv pip install --python "$VENV_PYTHON" --no-deps nvidia-cudss-cu12

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

echo "Building torchcodec from source (v0.10.0, CPU decode against system FFmpeg)..."
# torchcodec needs PyTorch and NVIDIA runtime libs on LD_LIBRARY_PATH during build.
NVIDIA_LIB_DIRS="$(find "${SITE_PKGS}/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
rm -rf /tmp/torchcodec
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git /tmp/torchcodec
cd /tmp/torchcodec
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install --python "$VENV_PYTHON" . --no-build-isolation
cd "$REPO_ROOT" && rm -rf /tmp/torchcodec

echo ""
echo "Install complete! In each new shell, activate with:"
echo "  source .venv/bin/activate"
echo "  source scripts/activate_orin.sh"
