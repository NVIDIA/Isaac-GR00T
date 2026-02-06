#!/bin/bash

if [ -z "$GROOT_CHECKPOINT_PATH" ]; then
    echo "Error: GROOT_CHECKPOINT_PATH environment variable is not set" >&2
    exit 1
fi

python scripts/inference_service.py --server --http-server --host 10.111.83.67 --port 8000 --model-path "$GROOT_CHECKPOINT_PATH" --num-action-steps 32
