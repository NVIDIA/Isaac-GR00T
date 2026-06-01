#!/usr/bin/env bash
# Patch Triton 3.3.1 to recognize CUDA major version 13+.
# PyTorch 2.7 pins Triton to 3.3.1, which does not handle CUDA 13.x,
# causing a RuntimeError in ptx_get_version(). This script:
#   1. Patches compiler.py directly (works until uv reinstalls triton).
#   2. Installs a .pth file that monkey-patches triton at Python startup,
#      so the fix survives `uv run` reinstalls.
#
# Usage:
#   bash scripts/patch_triton_cuda13.sh            # auto-detect site-packages
#   bash scripts/patch_triton_cuda13.sh /path/to/compiler.py  # explicit path

set -euo pipefail

if [ $# -ge 1 ]; then
    COMPILER_PY="$1"
else
    COMPILER_PY="$(python -c "import triton.backends.nvidia.compiler as c; print(c.__file__)")"
fi

if [ ! -f "$COMPILER_PY" ]; then
    echo "ERROR: Cannot find Triton compiler.py at: $COMPILER_PY" >&2
    exit 1
fi

# --- Step 1: File-level patch (best-effort, may be overwritten by uv) ---

if grep -q 'major == 13' "$COMPILER_PY"; then
    echo "Triton compiler.py already patched for CUDA 13.x"
else
    if ! grep -q 'major == 12' "$COMPILER_PY"; then
        echo "ERROR: Cannot find 'major == 12' in $COMPILER_PY — unexpected Triton version?" >&2
        exit 1
    fi
    # Insert "if major == 13: return 90 + minor" before the existing "if major == 12:" line.
    sed -i '/if major == 12:/i\    if major == 13:' "$COMPILER_PY"
    sed -i '/if major == 13:/a\        return 90 + minor' "$COMPILER_PY"
    echo "Patched $COMPILER_PY to support CUDA 13.x"
fi

# --- Step 2: Install .pth startup hook (survives uv reinstalls) ---

SITE_PACKAGES="$(python -c "import site; print(site.getsitepackages()[0])")"
PTH_FILE="${SITE_PACKAGES}/triton_cuda13_patch.pth"

cat > "$PTH_FILE" << 'PTHEOF'
import triton_cuda13_patch
PTHEOF

cat > "${SITE_PACKAGES}/triton_cuda13_patch.py" << 'PYEOF'
"""Monkey-patch Triton to support CUDA 13.x (installed by patch_triton_cuda13.sh)."""
def _apply():
    try:
        from triton.backends.nvidia import compiler as _c
        _orig = _c.ptx_get_version
        def _patched(cuda_version):
            major, minor = map(int, cuda_version.split('.'))
            if major == 13:
                return 90 + minor
            return _orig(cuda_version)
        _c.ptx_get_version = _patched
    except (ImportError, AttributeError):
        pass
_apply()
del _apply
PYEOF

echo "Installed ${PTH_FILE} (runtime monkey-patch, survives uv reinstalls)"
