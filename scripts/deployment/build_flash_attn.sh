#!/bin/bash
# Source this file from a platform installer, then call build_flash_attn.

build_flash_attn() {
    local wheel_dir="$1"
    local cache_key="$2"
    local cuda_archs="$3"
    local torch_cuda_arch="$4"
    local flash_attn_whl=""
    local local_whl
    local cache_whl
    local build_dir
    local built_whl

    : "${VENV_PYTHON:?VENV_PYTHON must be set by the platform installer}"
    if [ -z "${SUDO+x}" ]; then
        echo "ERROR: SUDO must be set by the platform installer" >&2
        return 1
    fi

    local_whl=$(find "$wheel_dir" -name 'flash_attn-*.whl' -print -quit 2>/dev/null || true)
    if [ -n "$local_whl" ]; then
        echo "Found local flash-attn wheel: $local_whl"
        flash_attn_whl="$local_whl"
    fi

    if [ -z "$flash_attn_whl" ] && [ -n "${GROOT_CACHE_DIR:-}" ]; then
        cache_whl=$(find "${GROOT_CACHE_DIR}/wheels/${cache_key}" -name 'flash_attn-*.whl' -print -quit 2>/dev/null || true)
        if [ -n "$cache_whl" ]; then
            echo "Found cached flash-attn wheel: $cache_whl"
            flash_attn_whl="$cache_whl"
        fi
    fi

    if [ -n "$flash_attn_whl" ]; then
        echo "Installing flash-attn from prebuilt wheel..."
        uv pip install --python "$VENV_PYTHON" --force-reinstall --no-deps "$flash_attn_whl"
        return
    fi

    echo "No prebuilt flash-attn wheel found — building v2.8.3 for sm_${cuda_archs//;/, sm_}..."
    echo "To skip this in the future, commit the built wheel to $wheel_dir/"

    local missing_packages=()
    local package
    for package in cmake git ninja-build python3-dev; do
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    if [ "${#missing_packages[@]}" -gt 0 ]; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y --no-install-recommends "${missing_packages[@]}"
    fi

    uv pip install --python "$VENV_PYTHON" pip
    export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
    export FLASH_ATTN_CUDA_ARCHS="$cuda_archs"
    export TORCH_CUDA_ARCH_LIST="$torch_cuda_arch"

    build_dir=$(mktemp -d /tmp/gr00t-flash-attn.XXXXXX)
    TMP_BUILD_DIRS+=("$build_dir")
    git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git "$build_dir/flash-attn"

    # The source checkout does not populate git submodules. Pin CUTLASS so a
    # rebuild is repeatable instead of silently following its moving main branch.
    local cutlass_sha="f74fea9ce35868d3ae9f8d1dce1969d7250d3f90"
    rm -rf "$build_dir/flash-attn/csrc/cutlass"
    mkdir -p "$build_dir/flash-attn/csrc/cutlass"
    git -C "$build_dir/flash-attn/csrc/cutlass" init --quiet
    git -C "$build_dir/flash-attn/csrc/cutlass" remote add origin https://github.com/NVIDIA/cutlass.git
    git -C "$build_dir/flash-attn/csrc/cutlass" fetch --depth 1 --quiet origin "$cutlass_sha"
    git -C "$build_dir/flash-attn/csrc/cutlass" checkout --quiet FETCH_HEAD
    rm -rf "$build_dir/flash-attn/.git"

    # v2.8.3's build script only knows the upstream default architectures.
    # Add platform targets explicitly and fail if that upstream layout changes.
    CUDA_ARCHS="$cuda_archs" "$VENV_PYTHON" - "$build_dir/flash-attn/setup.py" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
archs = os.environ["CUDA_ARCHS"].split(";")
text = path.read_text()
old_default = 'os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")'
if old_default not in text:
    raise SystemExit(f"ERROR: flash-attn architecture default changed in {path}")
text = text.replace(old_default, f'os.getenv("FLASH_ATTN_CUDA_ARCHS", "{";".join(archs)}")', 1)

for arch in archs:
    if arch == "87":
        anchor = '''    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
'''
        addition = '''    if "87" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
'''
    elif arch in {"110", "121"}:
        anchor = '''        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
'''
        addition = f'''        if bare_metal_version >= Version("12.8") and "{arch}" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_{arch},code=sm_{arch}")
'''
    else:
        raise SystemExit(f"ERROR: unsupported flash-attn architecture: {arch}")

    if addition not in text:
        if anchor not in text:
            raise SystemExit(f"ERROR: expected flash-attn compiler block not found in {path}")
        text = text.replace(anchor, anchor + addition, 1)

path.write_text(text)
PY

    mkdir -p "$build_dir/wheels"
    "$VENV_PYTHON" -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir "$build_dir/wheels" "$build_dir/flash-attn"
    built_whl=$(find "$build_dir/wheels" -name 'flash_attn-*.whl' -print -quit)
    if [ -z "$built_whl" ]; then
        echo "ERROR: flash-attn wheel build completed without producing a wheel" >&2
        return 1
    fi
    uv pip install --python "$VENV_PYTHON" --force-reinstall --no-deps "$built_whl"

    if [ -n "${GROOT_CACHE_DIR:-}" ]; then
        mkdir -p "${GROOT_CACHE_DIR}/wheels/${cache_key}"
        cp "$built_whl" "${GROOT_CACHE_DIR}/wheels/${cache_key}/"
        echo "Cached built wheel to ${GROOT_CACHE_DIR}/wheels/${cache_key}/"
    fi

    mkdir -p "$wheel_dir"
    cp "$built_whl" "$wheel_dir/"
    echo "Saved built wheel to $wheel_dir/"

    rm -rf "$build_dir"
}
