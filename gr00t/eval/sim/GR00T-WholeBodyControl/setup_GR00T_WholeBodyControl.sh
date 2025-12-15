#!/usr/bin/env bash
set -euxo pipefail

# Where this script lives (put it inside your repo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
GR00T_WHOLEBODYCONTROL_REPO="$PROJECT_REPO/external_dependencies/GR00T-WholeBodyControl"
UV_ENV="$SCRIPT_DIR/GR00T-WholeBodyControl_uv"

git submodule update --init $GR00T_WHOLEBODYCONTROL_REPO

# Build helpers
# python -m pip install cmake==3.18.4
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install setuptools wheel

# # Sim stack
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS not installed. Please install: https://git-lfs.github.com/"
    exit 1
fi
git -C "$GR00T_WHOLEBODYCONTROL_REPO" lfs pull
rm -rf "$GR00T_WHOLEBODYCONTROL_REPO/gr00t_wbc/dexmg/gr00trobosuite"
git clone https://github.com/xieleo5/robosuite.git "$GR00T_WHOLEBODYCONTROL_REPO/gr00t_wbc/dexmg/gr00trobosuite" -b leo/support_g1_locomanip
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "$GR00T_WHOLEBODYCONTROL_REPO" --config-settings editable_mode=compat
uv pip install -e "$GR00T_WHOLEBODYCONTROL_REPO/gr00t_wbc/dexmg/gr00trobosuite" --config-settings editable_mode=compat
uv pip install -e "$GR00T_WHOLEBODYCONTROL_REPO/gr00t_wbc/dexmg/gr00trobocasa" --config-settings editable_mode=compat
uv pip install mujoco==3.2.6 transformers==4.51.3

uv pip install --editable "$PROJECT_REPO" --no-deps

# Sanity import & env construction
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robocasa, robosuite
import gr00t_wbc.control.envs.robocasa.sync_env
print("Imports OK:", robosuite.__version__)
env = gym.make("gr00tlocomanip_g1_sim/LMBottlePnP_G1_gear_wbc", enable_render=True)
print("Env OK:", type(env))
PY
