#!/bin/bash
# install_deps.sh — One-time install of GR00T deps on Jetson Orin (aarch64, JetPack 7.2+, CUDA 13, Python 3.10+)
# Used by both bare metal and scripts/deployment/orin/Dockerfile.
# After install, use `source scripts/activate_orin.sh` in each new shell.
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
    echo "ERROR: This script is intended for aarch64 (Jetson Orin). Detected: $ARCH"
    exit 1
fi

python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 15), (
    f"Python 3.10–3.14 required, got {sys.version_info.major}.{sys.version_info.minor}"
)'
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
UV_PYTHON="${UV_PYTHON:-$PYTHON_VERSION}"
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "NOTE: Orin CI targets Python 3.12; detected Python $PYTHON_VERSION"
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

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
echo "Running uv sync with the Orin pyproject at $SCRIPT_DIR (venv: $UV_PROJECT_ENVIRONMENT)..."
uv sync --project "$SCRIPT_DIR" --no-install-project --python "$UV_PYTHON" --extra dev

VENV_DIR="$UV_PROJECT_ENVIRONMENT"
VENV_PYTHON="$VENV_DIR/bin/python"
SITE_PKGS="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

echo "Installing gr00t in editable mode from the repo root (--no-deps)..."
uv pip install --python "$VENV_PYTHON" --no-deps -e "$REPO_ROOT"

# ──────────────────────────────────────────────────────────────────────────────
# JetPack system packages (TensorRT, etc.) — expose to the venv via .pth file.
# ──────────────────────────────────────────────────────────────────────────────
echo "Linking JetPack system packages (TensorRT) into venv..."
echo "/usr/lib/python${PYTHON_VERSION}/dist-packages" \
    > "${SITE_PKGS}/jetpack-system-packages.pth"

# ──────────────────────────────────────────────────────────────────────────────
# torchcodec — prebuilt wheel from platform wheels/ dir or source build
# ──────────────────────────────────────────────────────────────────────────────
echo "Installing FFmpeg runtime..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends ffmpeg

TORCHCODEC_WHL=$(find "$SCRIPT_DIR/wheels" -name 'torchcodec-*.whl' -print -quit 2>/dev/null || true)
if [ -n "$TORCHCODEC_WHL" ]; then
    echo "Installing torchcodec from prebuilt wheel: $TORCHCODEC_WHL"
    uv pip install --python "$VENV_PYTHON" --force-reinstall --no-deps "$TORCHCODEC_WHL"
else
    echo "No prebuilt torchcodec wheel found — installing from jetson-sbsa-cu130 index via uv sync."
fi

echo ""
echo "Install complete! In each new shell, activate with:"
echo "  source .venv/bin/activate"
echo "  source scripts/activate_orin.sh"
