#!/bin/bash
# Build missing aarch64 dGPU wheels before the root pyproject's path sources are
# resolved by uv sync. Dependency pins are the source of truth; the pyproject
# path sources are updated to point at the matching generated wheel names.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
WHEEL_DIR="$SCRIPT_DIR/wheels"

mkdir -p "$WHEEL_DIR"

# Target CPython derived from the project's requires-python, so the build venv
# and the wheel's cpXY tag track the pinned interpreter. A literal cp310/python3.10
# silently builds an unusable wheel after a runtime bump (e.g. the py3.12 migration).
resolve_python_version() {
    python3 - "$PYPROJECT" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
# Accept both TOML quote styles (" and ', written as hex \x22 / \x27).
m = re.search(r"requires-python\s*=\s*[\x22\x27]([^\x22\x27]+)[\x22\x27]", text)
if m is None:
    raise SystemExit("ERROR: requires-python not found in pyproject.toml")
v = re.search(r"(\d+)\.(\d+)", m.group(1))
if v is None:
    raise SystemExit(f"ERROR: could not parse a CPython version from {m.group(1)!r}")
print(f"{v.group(1)}.{v.group(2)}")
PY
}

PY_VERSION="$(resolve_python_version)"
PYTHON_BIN="python${PY_VERSION}"
CP_TAG="cp${PY_VERSION/./}"

resolve_dependency_version() {
    python3 - "$PYPROJECT" "$1" <<'PY'
import pathlib
import re
import sys

pyproject_path = pathlib.Path(sys.argv[1])
package = sys.argv[2]
text = pyproject_path.read_text()

dep_line = None
for line in text.splitlines():
    if f'"{package}==' in line and "platform_machine" in line and "aarch64" in line:
        dep_line = line
        break
if dep_line is None:
    for line in text.splitlines():
        if f'"{package}==' in line:
            dep_line = line
            break
if dep_line is None:
    raise SystemExit(f"ERROR: pinned project dependency not found for {package}")

dep_match = re.search(rf'"{re.escape(package)}==([^";]+)', dep_line)
if dep_match is None:
    raise SystemExit(f"ERROR: could not parse pinned dependency version for {package}")

print(dep_match.group(1))
PY
}

update_pyproject_path_source() {
    python3 - "$PYPROJECT" "$1" "$2" <<'PY'
import pathlib
import re
import sys

pyproject_path = pathlib.Path(sys.argv[1])
package = sys.argv[2]
expected_path = sys.argv[3]
text = pyproject_path.read_text()

source_block_match = re.search(
    rf"^{re.escape(package)}\s*=\s*\[(.*?)^\]",
    text,
    flags=re.MULTILINE | re.DOTALL,
)
if source_block_match is None:
    raise SystemExit(f"ERROR: [tool.uv.sources] entry not found for {package}")

block_start, block_end = source_block_match.span(1)
block = source_block_match.group(1)

entry_pattern = re.compile(r"\{[^\n]*path\s*=\s*\"([^\"]+)\"[^\n]*\}")
matches = list(entry_pattern.finditer(block))
aarch64_matches = [
    match
    for match in matches
    if "platform_machine == 'aarch64'" in match.group(0)
    and "scripts/deployment/dgpu/wheels/" in match.group(1)
]
if len(aarch64_matches) != 1:
    raise SystemExit(
        f"ERROR: expected exactly one dGPU aarch64 path source for {package}, "
        f"found {len(aarch64_matches)}"
    )

match = aarch64_matches[0]
current_path = match.group(1)
if current_path == expected_path:
    print(f"{package} path source already points to {expected_path}")
    raise SystemExit(0)

entry = match.group(0)
updated_entry = entry.replace(f'path = "{current_path}"', f'path = "{expected_path}"')
updated_block = block[: match.start()] + updated_entry + block[match.end() :]
updated_text = text[:block_start] + updated_block + text[block_end:]
pyproject_path.write_text(updated_text)
print(f"Updated {package} path source: {current_path} -> {expected_path}")
PY
}

wheel_path_for() {
    package_prefix="$1"
    version="$2"
    printf "%s/%s-%s-${CP_TAG}-${CP_TAG}-linux_aarch64.whl" "$WHEEL_DIR" "$package_prefix" "$version"
}

FLASH_ATTN_VERSION="$(resolve_dependency_version "flash-attn")"
TORCHCODEC_VERSION="$(resolve_dependency_version "torchcodec")"
TORCHCODEC_SOURCE_VERSION="$(printf "%s" "$TORCHCODEC_VERSION" | sed -E 's/a[0-9]+$//')"

FLASH_ATTN_WHEEL="$(wheel_path_for "flash_attn" "$FLASH_ATTN_VERSION")"
TORCHCODEC_WHEEL="$(wheel_path_for "torchcodec" "$TORCHCODEC_VERSION")"
FLASH_ATTN_PATH="scripts/deployment/dgpu/wheels/$(basename "$FLASH_ATTN_WHEEL")"
TORCHCODEC_PATH="scripts/deployment/dgpu/wheels/$(basename "$TORCHCODEC_WHEEL")"

echo "Expected dGPU aarch64 wheels from dependency pins:"
echo "  flash-attn==$FLASH_ATTN_VERSION -> $FLASH_ATTN_PATH"
echo "  torchcodec==$TORCHCODEC_VERSION -> $TORCHCODEC_PATH"

update_pyproject_path_source "flash-attn" "$FLASH_ATTN_PATH"
update_pyproject_path_source "torchcodec" "$TORCHCODEC_PATH"

if [ "${DGPU_WHEEL_BOOTSTRAP_VALIDATE_ONLY:-0}" = "1" ]; then
    exit 0
fi

if [ "$(uname -m)" != "aarch64" ]; then
    echo "dGPU wheel bootstrap is only needed on aarch64; skipping."
    exit 0
fi

if [ -f "$FLASH_ATTN_WHEEL" ] && [ -f "$TORCHCODEC_WHEEL" ]; then
    echo "Matching dGPU aarch64 wheels already exist; skipping source builds."
    exit 0
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is required to bootstrap dGPU wheels." >&2
    exit 1
fi

BUILD_VENV="${DGPU_WHEEL_BUILD_VENV:-/tmp/gr00t-dgpu-wheel-build-venv}"
BUILD_PYTHON="$BUILD_VENV/bin/python"
TMP_BUILD_DIRS=()
trap 'rm -rf "$BUILD_VENV"; for _d in "${TMP_BUILD_DIRS[@]:-}"; do rm -rf "$_d"; done' EXIT

rm -rf "$BUILD_VENV"
"$PYTHON_BIN" -m venv "$BUILD_VENV"

uv_pip_install_retry() {
    for attempt in 1 2 3 4 5; do
        if uv pip install "$@"; then
            return 0
        fi
        sleep $((attempt * 10))
    done
    return 1
}

uv_pip_install_retry --python "$BUILD_PYTHON" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch==2.7.1 triton==3.3.1 numpy==1.26.4
uv_pip_install_retry --python "$BUILD_PYTHON" pip setuptools wheel packaging ninja

SITE_PKGS=$("$BUILD_PYTHON" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)
NVIDIA_LIB_DIRS="$(find "${SITE_PKGS}/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

build_flash_attn() {
    if [ -f "$FLASH_ATTN_WHEEL" ]; then
        echo "flash-attn wheel already exists: $FLASH_ATTN_WHEEL"
        return
    fi

    echo "No dGPU aarch64 flash-attn wheel found; building from source..."
    rm -rf /tmp/flash-attn
    TMP_BUILD_DIRS+=(/tmp/flash-attn)
    git clone --depth 1 --branch "v${FLASH_ATTN_VERSION}" \
        https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn
    rm -rf /tmp/flash-attn/.git

    FLASH_ATTENTION_FORCE_BUILD=TRUE "$BUILD_PYTHON" -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "$WHEEL_DIR" \
        /tmp/flash-attn

    if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
        echo "ERROR: flash-attn source build did not produce expected wheel:" >&2
        echo "  $FLASH_ATTN_WHEEL" >&2
        echo "Available flash-attn wheels:" >&2
        find "$WHEEL_DIR" -maxdepth 1 -name 'flash_attn-*.whl' -print >&2
        exit 1
    fi
}

build_torchcodec() {
    if [ -f "$TORCHCODEC_WHEEL" ]; then
        echo "torchcodec wheel already exists: $TORCHCODEC_WHEEL"
        return
    fi

    echo "No dGPU aarch64 torchcodec wheel found; building from source..."
    rm -rf /tmp/torchcodec
    TMP_BUILD_DIRS+=(/tmp/torchcodec)
    git clone --depth 1 --branch "v${TORCHCODEC_SOURCE_VERSION}" \
        https://github.com/pytorch/torchcodec.git /tmp/torchcodec
    rm -rf /tmp/torchcodec/.git

    I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 "$BUILD_PYTHON" -m pip wheel \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "$WHEEL_DIR" \
        /tmp/torchcodec

    if [ ! -f "$TORCHCODEC_WHEEL" ]; then
        echo "ERROR: torchcodec source build did not produce expected wheel:" >&2
        echo "  $TORCHCODEC_WHEEL" >&2
        echo "Available torchcodec wheels:" >&2
        find "$WHEEL_DIR" -maxdepth 1 -name 'torchcodec-*.whl' -print >&2
        exit 1
    fi
}

cd "$REPO_ROOT"
build_flash_attn
build_torchcodec

echo "dGPU aarch64 wheel bootstrap complete:"
ls -lh "$FLASH_ATTN_WHEEL" "$TORCHCODEC_WHEEL"
