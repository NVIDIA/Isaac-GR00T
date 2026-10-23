#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash examples/naviai_wa1_head_lr_wf/launch_feature_server.sh CONFIG.env" >&2
    exit 2
fi

config_path=$1
if [[ ! -f "$config_path" ]]; then
    echo "configuration file does not exist: $config_path" >&2
    exit 2
fi

set -a
source "$config_path"
set +a

: "${MODEL_PATH:?set MODEL_PATH to the parcel_4f_v5.9 checkpoint directory}"
: "${SPEED_RL_MODEL_ID:?set SPEED_RL_MODEL_ID}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "checkpoint directory does not exist: $MODEL_PATH" >&2
    exit 2
fi
for checkpoint_file in config.json processor_config.json embodiment_id.json model.safetensors.index.json; do
    if [[ ! -f "$MODEL_PATH/$checkpoint_file" ]]; then
        echo "checkpoint is missing $checkpoint_file: $MODEL_PATH" >&2
        exit 2
    fi
done

python_bin=${PYTHON_BIN:-.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable is unavailable: $python_bin (run uv sync first)" >&2
    exit 2
fi

exec "$python_bin" -m gr00t.eval.run_gr00t_server \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "${EMBODIMENT_TAG:-naviai_wa1_head_lr_wf}" \
    --device "${DEVICE:-cuda}" \
    --host "${SERVER_BIND_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-47866}" \
    --speed-rl-features \
    --speed-rl-model-id "$SPEED_RL_MODEL_ID"
