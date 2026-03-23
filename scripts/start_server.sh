#!/bin/bash
# GR00T inference server startup script
# Usage: docker exec -it gr00t bash /workspace/gr00t/scripts/start_server.sh

EMBODIMENT_TAG="${EMBODIMENT_TAG:-GR1}"
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"

echo "=== GR00T Inference Server ==="
echo "Embodiment: $EMBODIMENT_TAG"
echo "Model:      $MODEL_PATH"
echo "Device:     $DEVICE"
echo "Listen:     $HOST:$PORT"
echo "=============================="

python -m gr00t.eval.run_gr00t_server \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --use-sim-policy-wrapper
