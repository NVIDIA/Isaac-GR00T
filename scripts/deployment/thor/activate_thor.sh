#!/bin/bash
# Source this in each new shell after activating the Thor environment.

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
fi

SITE_PKGS="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"

# Strip system CUDA lib64 paths from any pre-existing LD_LIBRARY_PATH before adding pip CUDA
# libs. The pip libcudss.so.0 has a hard-coded RPATH ($ORIGIN) that forces pip libcublas to
# load regardless of LD_LIBRARY_PATH order. If system libcublasLt then loads from a path
# earlier in LD_LIBRARY_PATH we end up with mismatched cuBLAS libraries, which causes
# CUBLAS_STATUS_NOT_INITIALIZED. Stripping system CUDA ensures both libcublas and libcublasLt
# come from the same (pip) source.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _filtered_ld="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda' | paste -sd: -)"
    export LD_LIBRARY_PATH="${_filtered_ld}"
    unset _filtered_ld
fi

if [ -n "${SITE_PKGS:-}" ] && [ -d "${SITE_PKGS}/torch/lib" ]; then
    export LD_LIBRARY_PATH="${SITE_PKGS}/torch/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -d "${CUDA_HOME:-}/include" ]; then
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
fi

if [ -n "${SITE_PKGS:-}" ]; then
    NVIDIA_LIB_DIRS="$(find "$SITE_PKGS/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
    if [ -n "${NVIDIA_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if [ -d "${VIRTUAL_ENV:-}/cyclonedds" ]; then
    export CYCLONEDDS_HOME="${VIRTUAL_ENV}/cyclonedds"
    export CMAKE_PREFIX_PATH="${CYCLONEDDS_HOME}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "Thor environment configured."
echo "  TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-<not set>}"
echo "  CUDA_HOME=${CUDA_HOME:-<not set>}"
echo "  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<not set>}"
