#!/bin/bash
# Desktop-friendly launcher for the GR00T N1.7 inference server.
# No SLURM required. Works on the lab cluster head node, on a workstation
# (RTX 5090 etc.), or any machine where the gr00tN17 conda env exists.
#
# Usage:
#   bash run_inference_server.sh <RUN_NAME> [<CHECKPOINT_STEP>] [<PORT>]
#   bash run_inference_server.sh mousebox_packing_0421_ABS_VT 10000 5555
#
# Or via env vars for an arbitrary checkpoint path:
#   MODEL_PATH=/path/to/checkpoint-10000 PORT=5555 bash run_inference_server.sh
#
# All optional env vars:
#   MODEL_PATH       absolute checkpoint path (overrides RUN_NAME / CHECKPOINT_STEP)
#   EMBODIMENT_TAG   default: new_embodiment
#   HOST             default: 0.0.0.0  (listen on all interfaces)
#   PORT             default: 5555
#   STRICT           default: 1  (set 0 to disable strict input/output validation)
#   CONDA_ENV        default: gr00tN17
#   GR00T_ROOT       default: directory containing this script
#   SKIP_CONDA       set 1 to skip auto-activation (use the active env as-is)

set -euo pipefail

# ---- Locate repo ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${GR00T_ROOT:-$SCRIPT_DIR}"
cd "$GR00T_ROOT"

# ---- Optional conda activation ---------------------------------------------
CONDA_ENV="${CONDA_ENV:-gr00tN17}"
if [ "${SKIP_CONDA:-0}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    CONDA_BASE_CANDIDATES=(
        "${CONDA_PREFIX:-}"
        "${CONDA_BASE:-}"
        "$HOME/miniconda3"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
        "$HOME/anaconda3"
        "/opt/conda"
        "/workspace/home/rlwrld/david/miniconda3"
    )
    for path in "${CONDA_BASE_CANDIDATES[@]}"; do
        if [ -n "$path" ] && [ -f "$path/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$path/etc/profile.d/conda.sh"
            conda activate "$CONDA_ENV"
            echo "Activated conda env '$CONDA_ENV' (from $path)"
            break
        fi
    done
    if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
        echo "Warning: could not auto-activate conda env '$CONDA_ENV'." >&2
        echo "         Activate it yourself or set SKIP_CONDA=1." >&2
    fi
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTHONPATH="$GR00T_ROOT"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

python -c "import gr00t; print('gr00t:', gr00t.__file__)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# ---- Argument resolution ---------------------------------------------------
RUN_NAME="${1:-mousebox_packing_0421_ABS_VT}"
CHECKPOINT_STEP="${2:-10000}"
PORT_ARG="${3:-${PORT:-5555}}"

EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
HOST="${HOST:-0.0.0.0}"
STRICT="${STRICT:-1}"

if [ -n "${MODEL_PATH:-}" ]; then
    CKPT="$MODEL_PATH"
else
    CKPT="output/train/${RUN_NAME}/checkpoint-${CHECKPOINT_STEP}"
fi

if [ ! -d "$CKPT" ]; then
    echo "Error: checkpoint directory not found: $CKPT" >&2
    echo "       Set MODEL_PATH=/abs/path/to/checkpoint-N to override." >&2
    exit 2
fi

echo "========================================="
echo "  GR00T N1.7 inference server"
echo "  checkpoint: $CKPT"
echo "  embodiment: $EMBODIMENT_TAG"
echo "  host:port:  $HOST:$PORT_ARG"
echo "  strict:     $STRICT"
echo "========================================="

SERVER_ARGS=(
    --model-path "$CKPT"
    --embodiment-tag "$EMBODIMENT_TAG"
    --host "$HOST"
    --port "$PORT_ARG"
)

if [ "$STRICT" = "0" ] || [ "$STRICT" = "false" ]; then
    SERVER_ARGS+=(--no-strict)
else
    SERVER_ARGS+=(--strict)
fi

exec python -m gr00t.eval.run_gr00t_server "${SERVER_ARGS[@]}"
