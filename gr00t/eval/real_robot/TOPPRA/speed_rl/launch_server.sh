#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash gr00t/eval/real_robot/TOPPRA/speed_rl/launch_server.sh CONFIG.local.env" >&2
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

: "${MODEL_PATH:?set MODEL_PATH in the server configuration}"
: "${SPEED_RL_MODEL_ID:?set SPEED_RL_MODEL_ID in the server configuration}"

python_bin=${PYTHON_BIN:-.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable is unavailable: $python_bin (run uv sync --all-extras)" >&2
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
