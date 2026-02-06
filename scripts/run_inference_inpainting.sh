#!/bin/bash

if [ -z "$GROOT_CHECKPOINT_PATH" ]; then
    echo "Error: GROOT_CHECKPOINT_PATH environment variable is not set" >&2
    exit 1
fi

# Default to 32 action steps, execute 16, inpaint 16
NUM_ACTION_STEPS="${NUM_ACTION_STEPS:-32}"
NUM_ACTION_EXECUTE_STEPS="${NUM_ACTION_EXECUTE_STEPS:-$(( NUM_ACTION_STEPS / 2 ))}"
NUM_PREFIX_STEPS="${NUM_PREFIX_STEPS:-$(( NUM_ACTION_STEPS / 2 ))}"

echo "Starting GR00T HTTP server with INPAINTING mode"
echo "  Action horizon:     $NUM_ACTION_STEPS"
echo "  Execute steps:      $NUM_ACTION_EXECUTE_STEPS"
echo "  Prefix (inpaint):   $NUM_PREFIX_STEPS"

python scripts/inference_service.py \
    --server \
    --http-server \
    --host 10.111.83.67 \
    --port 8001 \
    --model-path "$GROOT_CHECKPOINT_PATH" \
    --num-action-steps "$NUM_ACTION_STEPS" \
    --use-inpainting \
    --n-action-execute-steps "$NUM_ACTION_EXECUTE_STEPS" \
    --num-prefix-steps "$NUM_PREFIX_STEPS"
