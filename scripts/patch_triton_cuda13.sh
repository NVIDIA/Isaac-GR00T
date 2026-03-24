#!/usr/bin/env bash
# Patch Triton 3.3.1 to recognize CUDA major version 13+.
# PyTorch 2.7 pins Triton to 3.3.1, which does not handle CUDA 13.x,
# causing a RuntimeError in ptx_get_version(). This script adds the
# missing branch so that CUDA 13.x maps to PTX version 90+minor.
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

if grep -q 'major == 13' "$COMPILER_PY"; then
    echo "Triton compiler.py already patched for CUDA 13.x"
    exit 0
fi

if ! grep -q 'major == 12' "$COMPILER_PY"; then
    echo "ERROR: Cannot find 'major == 12' in $COMPILER_PY — unexpected Triton version?" >&2
    exit 1
fi

# Insert "if major == 13: return 90 + minor" before the existing "if major == 12:" line.
sed -i '/if major == 12:/i\    if major == 13:' "$COMPILER_PY"
sed -i '/if major == 13:/a\        return 90 + minor' "$COMPILER_PY"

echo "Patched $COMPILER_PY to support CUDA 13.x"
