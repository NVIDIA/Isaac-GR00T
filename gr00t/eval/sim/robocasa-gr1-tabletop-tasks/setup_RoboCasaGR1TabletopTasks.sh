#!/usr/bin/env bash
set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
ROBOCASA_GR1_TABLETOP_TASKS_REPO="$PROJECT_REPO/external_dependencies/robocasa-gr1-tabletop-tasks"
UV_ENV="$SCRIPT_DIR/robocasa_uv"

# Optional: if you want to avoid hardlink warnings with uv cache
# export UV_LINK_MODE=copy

git submodule update --init $ROBOCASA_GR1_TABLETOP_TASKS_REPO

# Ensure build tools (mirrors your LIBERO flow)
# python -m pip install cmake==3.18.4

# Fresh venv
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"

# Make sure the venv has a build backend
uv pip install setuptools wheel

# Heavy deps first
uv pip install torch==2.5.1 torchvision==0.20.1

# Preinstall flash-attn to avoid builds inside other installs.
# Guard it to Linux only (flash-attn not supported on macOS).
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
else
  echo "Skipping flash-attn (non-Linux or INSTALL_FLASH_ATTN=0)"
fi

# Core sim deps: robosuite first (as per README), then this repo editable
# README: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@master"

# The repo’s requirements.txt only contains "-e .", so just install editable.
uv pip install -e "$ROBOCASA_GR1_TABLETOP_TASKS_REPO" --config-settings editable_mode=compat

# Optional: your eval stack uses gymnasium
uv pip install gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8

# Make your project importable without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Optional: lower robosuite simulation timestep to 0.005 for headless stability
# python - <<'PY'
# import importlib, re
# try:
#     rs = importlib.import_module("robosuite.macros_private")
#     path = rs.__file__
# except Exception:
#     rs = importlib.import_module("robosuite.macros")
#     path = rs.__file__
# txt = open(path, "r", encoding="utf-8").read()
# new = re.sub(r"(SIMULATION_TIMESTEP\s*=\s*)([0-9.]+)", r"\g<1>0.005", txt)
# if txt != new:
#     open(path, "w", encoding="utf-8").write(new)
#     print(f"Updated SIMULATION_TIMESTEP in {path}")
# else:
#     print("No SIMULATION_TIMESTEP change needed")
# PY

# Assets (per README)
python "$ROBOCASA_GR1_TABLETOP_TASKS_REPO/robocasa/scripts/download_tabletop_assets.py" -y

# Sanity import
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
import robocasa.utils.gym_utils.gymnasium_groot
print("Imports OK:", robosuite.__version__)
env = gym.make("gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env", enable_render=True)
print("Env OK:", type(env))
PY


#pydantic
#av
#zmq
