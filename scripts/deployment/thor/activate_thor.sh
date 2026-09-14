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

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

_thor_python=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    _thor_python="${VIRTUAL_ENV}/bin/python"
elif [ -n "${THOR_ENV:-}" ] && [ -x "${THOR_ENV}/bin/python" ]; then
    _thor_python="${THOR_ENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    _thor_python="$(command -v python3)"
fi

SITE_PKGS=""
if [ -n "${_thor_python:-}" ]; then
    SITE_PKGS="$("$_thor_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
fi
unset _thor_python

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

NVIDIA_LIB_DIRS=""
CUDSS_LIB_DIR=""
if [ -n "${SITE_PKGS:-}" ]; then
    NVIDIA_LIB_DIRS="$(find "$SITE_PKGS/nvidia" -name lib -type d 2>/dev/null | paste -sd: -)"
    if [ -n "${NVIDIA_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    # The Jetson Torch build links cuDSS during import. Resolve it from the
    # selected environment rather than whichever python3 happens to be on PATH.
    CUDSS_LIB_DIR="$(find "$SITE_PKGS/nvidia" -name 'libcudss.so.0' -printf '%h\n' -quit 2>/dev/null || true)"
    if [ -n "${CUDSS_LIB_DIR:-}" ]; then
        export LD_LIBRARY_PATH="${CUDSS_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if [ -d "${VIRTUAL_ENV:-}/cyclonedds" ]; then
    export CYCLONEDDS_HOME="${VIRTUAL_ENV}/cyclonedds"
    export CMAKE_PREFIX_PATH="${CYCLONEDDS_HOME}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Share the activation/tmux contract, including ROS variables sourced by the caller.
_thor_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while IFS= read -r _thor_env_name; do
    if [[ -v "$_thor_env_name" ]]; then
        export "$_thor_env_name"
    fi
done < "${_thor_script_dir}/../runtime_environment.txt"
unset _thor_script_dir _thor_env_name

echo "Thor environment configured."
echo "  TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-<not set>}"
echo "  CUDA_HOME=${CUDA_HOME:-<not set>}"
echo "  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<not set>}"
echo "  SITE_PKGS=${SITE_PKGS:-<not found>}"
echo "  CUDSS_LIB_DIR=${CUDSS_LIB_DIR:-<not found>}"
