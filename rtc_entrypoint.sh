#!/bin/bash

# Use the virtual environment from /workspace/venv
export UV_PROJECT_ENVIRONMENT=/workspace/venv

# Set defaults for environment variables if not set
MODEL_PATH=${HF_MODEL_PATH:-"/model"}   # if huggingface model path is not provided, use the locally mounted model instead
INFERENCE_PORT=${INFERENCE_PORT:-8000}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-"new_embodiment"}
DATA_CONFIG=${DATA_CONFIG:-"so101_custom_config"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Starting inference with:"
echo "MODEL_PATH: $MODEL_PATH"
echo "INFERENCE_PORT: $INFERENCE_PORT" 
echo "EMBODIMENT_TAG: $EMBODIMENT_TAG"
echo "DATA_CONFIG: $DATA_CONFIG"

uv run python3 -m tng_rtc.rtc_inference_script --model-path $MODEL_PATH --port $INFERENCE_PORT --embodiment-tag $EMBODIMENT_TAG --data-config $DATA_CONFIG
