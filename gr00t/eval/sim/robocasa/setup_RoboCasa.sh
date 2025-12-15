#!/usr/bin/env bash
set -euxo pipefail

# Where this script lives (put it inside your repo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
ROBOCASA_REPO="$PROJECT_REPO/external_dependencies/robocasa"
UV_ENV="$SCRIPT_DIR/robocasa_uv"

git submodule update --init $ROBOCASA_REPO

# Build helpers
# python -m pip install cmake==3.18.4
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install setuptools wheel

# Core deps
uv pip install torch==2.5.1 torchvision==0.20.1
# Linux-only: preinstall flash-attn to avoid compiling inside other wheels
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
fi

# Sim stack
uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@master"
uv pip install -e "$ROBOCASA_REPO" --config-settings editable_mode=compat
uv pip install gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8

# Make your project importable in this venv without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Stable headless timestep (optional but recommended)
# python - <<'PY'
# import importlib, re
# try:
#     rs = importlib.import_module("robosuite.macros_private"); path = rs.__file__
# except Exception:
#     rs = importlib.import_module("robosuite.macros"); path = rs.__file__
# txt = open(path, "r", encoding="utf-8").read()
# new = re.sub(r"(SIMULATION_TIMESTEP\s*=\s*)([0-9.]+)", r"\g<1>0.005", txt)
# if txt != new:
#     open(path, "w", encoding="utf-8").write(new); print(f"Updated SIMULATION_TIMESTEP in {path}")
# else:
#     print("No SIMULATION_TIMESTEP change needed")
# PY

# Assets for RoboCasa (kitchen)
python "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py" -y

# Sanity import & env construction
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
import robocasa.utils.gym_utils.gymnasium_groot
print("Imports OK:", robosuite.__version__)
env = gym.make("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", enable_render=True)
print("Env OK:", type(env))
PY
