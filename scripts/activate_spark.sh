#!/bin/bash
# activate_spark.sh — Source this in each new shell to configure the Spark environment.
# Usage: source scripts/activate_spark.sh
#
# Sets TRITON_PTXAS_PATH and LD_LIBRARY_PATH needed for inference on DGX Spark.
# Docker users don't need this — the Dockerfile sets these via ENV directives.

# torch.compile needs ptxas from the CUDA toolkit
if [ -f /usr/local/cuda-13.0/bin/ptxas ]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
    export CUDA_HOME=/usr/local/cuda-13.0
    export CUDA_PATH=/usr/local/cuda-13.0
elif [ -f /usr/local/cuda/bin/ptxas ]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export CUDA_HOME=/usr/local/cuda
    export CUDA_PATH=/usr/local/cuda
else
    echo "WARNING: ptxas not found. torch.compile may fail."
    echo "  Install CUDA toolkit or set TRITON_PTXAS_PATH manually."
fi

# Triton and torchcodec need PyTorch's shared libraries on the runtime linker path.
TORCH_LIB_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "${TORCH_LIB_DIR:-}" ] && [ -d "${TORCH_LIB_DIR}/torch/lib" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}/torch/lib:${LD_LIBRARY_PATH:-}"
fi

# Triton compiles small CUDA helper modules and expects cuda.h to be discoverable.
if [ -d "${CUDA_HOME:-}/include" ]; then
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
fi

# Add nvidia pip package libs (cudnn, cudss, etc.) to LD_LIBRARY_PATH
SITE_PKGS="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "${SITE_PKGS:-}" ]; then
    NVIDIA_LIB_DIRS=$(find "$SITE_PKGS/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
    if [ -n "$NVIDIA_LIB_DIRS" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

echo "Spark environment configured."
echo "  TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-<not set>}"
echo "  CUDA_HOME=${CUDA_HOME:-<not set>}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"
