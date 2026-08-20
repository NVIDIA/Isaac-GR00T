#!/bin/bash
# install_deps.sh — One-time GR00T install for JetPack 7.2 (aarch64, Python 3.12).
# Used by both bare metal and the Orin/Thor Docker profiles.
# After install, use `source scripts/activate_jetpack72.sh` in each new shell.
set -euo pipefail

TMP_BUILD_DIRS=()
trap 'for _d in "${TMP_BUILD_DIRS[@]:-}"; do rm -rf "$_d"; done' EXIT

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
    echo "ERROR: This script is intended for aarch64 JetPack 7.2 systems. Detected: $ARCH"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "WARNING: Expected Python 3.12 for JetPack 7.2, detected Python $PYTHON_VERSION"
fi

if [ "${DOCKER_CONTAINER:-0}" != "1" ]; then
    if [ ! -r /etc/nv_tegra_release ]; then
        echo "ERROR: /etc/nv_tegra_release not found. Deployment requires JetPack 7.2 / Jetson Linux 39.2."
        exit 1
    fi

    L4T_RELEASE=$(sed -n 's/^# R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release | head -n1)
    L4T_REVISION=$(sed -n 's/.*REVISION: \([0-9.]\+\).*/\1/p' /etc/nv_tegra_release | head -n1)
    if [ "$L4T_RELEASE" != "39" ]; then
        echo "ERROR: Deployment supports JetPack 7.x / Jetson Linux R39 only."
        echo "       Tested target: JetPack 7.2 / Jetson Linux 39.2."
        echo "       Detected: R${L4T_RELEASE:-unknown}, revision ${L4T_REVISION:-unknown}."
        echo "       JetPack 6.x / Jetson Linux R36 is no longer supported."
        exit 1
    fi
    if [ "${L4T_REVISION%%.*}" != "2" ]; then
        echo "WARNING: Tested target is JetPack 7.2 / Jetson Linux 39.2."
        echo "         Detected: R${L4T_RELEASE}, revision ${L4T_REVISION:-unknown}."
    fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────────────────────

echo "Installing system dependencies..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends ffmpeg

# torch.compile needs ptxas; bare-metal JetPack images may only ship runtime
# libraries. The Docker image already has the full CUDA 13.2 development stack.
if [ ! -x /usr/local/cuda-13.2/bin/ptxas ] && [ ! -x /usr/local/cuda/bin/ptxas ]; then
    echo "Installing CUDA 13.2 dev packages (nvcc, cudart-dev, nvrtc-dev)..."
    if ! apt-cache show cuda-nvcc-13-2 &>/dev/null; then
        echo "Adding NVIDIA CUDA 13.2 apt repository..."
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
            -o /tmp/cuda-keyring.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
    fi
    $SUDO apt-get install -y --no-install-recommends \
        cuda-nvcc-13-2 cuda-cudart-dev-13-2 cuda-nvrtc-dev-13-2
else
    echo "CUDA ptxas already available."
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

# Install JetPack 7.2 deps from the shared pyproject without mutating
# the repo-root pyproject.toml / uv.lock.
#
# UV_PROJECT_ENVIRONMENT pins the venv location. Respect a pre-set value
# from the Docker build (the Orin and Thor Dockerfiles set it to
# /opt/gr00t-venv and adds /opt/gr00t-venv/bin to PATH); fall back to
# $REPO_ROOT/.venv on bare metal so activate_jetpack72.sh still finds the venv
# where users expect.
#
# --no-install-project skips installing "gr00t" from the JetPack 7.2 pyproject
# (its source layout points at the platform dir, which has no gr00t src);
# the real editable install comes from $REPO_ROOT below.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
echo "Running uv sync with the JetPack 7.2 pyproject at $SCRIPT_DIR (venv: $UV_PROJECT_ENVIRONMENT)..."
uv sync --project "$SCRIPT_DIR" --no-install-project --extra dev

VENV_DIR="$UV_PROJECT_ENVIRONMENT"
VENV_PYTHON="$VENV_DIR/bin/python"
SITE_PKGS="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

# Make the CUDA and PyTorch libraries visible while compiling the extension.
NVIDIA_LIB_DIRS="$(find "${SITE_PKGS}/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

# Build a CUDA 13.2 wheel locally because upstream does not publish a cu132
# aarch64 wheel containing the Orin and Thor GPU targets.
source "$SCRIPT_DIR/../build_flash_attn.sh"
build_flash_attn "$SCRIPT_DIR/wheels" jetpack-cu132-torch2.13 "87;110" "8.7;11.0"

echo "Installing gr00t in editable mode from the repo root (--no-deps)..."
uv pip install --python "$VENV_PYTHON" --no-deps -e "$REPO_ROOT"

echo ""
echo "Install complete! In each new shell, activate with:"
echo "  source .venv/bin/activate"
echo "  source scripts/activate_jetpack72.sh"
