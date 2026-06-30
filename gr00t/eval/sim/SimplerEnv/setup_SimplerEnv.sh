#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
SIMPLER_REPO="$PROJECT_REPO/external_dependencies/SimplerEnv"
UV_ENV="$SCRIPT_DIR/simpler_uv"

# Some transitive sim deps (e.g. egl_probe) ship a CMakeLists.txt that calls
# cmake_minimum_required(VERSION <3.5). CMake >= 4.0 removed compatibility with
# those, failing the build with "Compatibility with CMake < 3.5 has been
# removed". CMake reads this env var natively (>= 3.31) and treats it as the
# minimum policy version, letting the legacy build configure anyway.
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

git submodule update --init --recursive $SIMPLER_REPO

# Numpy pin: cluster uses 1.26.4; SimplerEnv README mentions 1.24.4 for pinocchio IK.
# Override by exporting SIMPLER_NUMPY=1.24.4 if needed.
SIMPLER_NUMPY="${SIMPLER_NUMPY:-1.26.4}"

# python -m pip install -U uv
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install "setuptools<71"

# Core deps (match cluster’s pyproject pattern)
uv pip install \
  gymnasium==0.29.1 \
  json-numpy>=2.1.1 \
  numpy=="$SIMPLER_NUMPY" \
  opencv-python-headless==4.10.0.84 \
  ray==2.48.0

# Install SimplerEnv sources (editable)
# uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim" --config-settings editable_mode=compat
# uv pip install -e "$SIMPLER_REPO" --config-settings editable_mode=compat

uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install -e "$SIMPLER_REPO"

# Make your OSS project importable.
# --python-version 3.12: gr00t's pyproject pins requires-python >=3.12,<3.15,
# but this sim venv must be 3.10 (sapien/maniskill ship cp310 wheels only).
# Resolving against 3.12 satisfies the metadata check while still installing
# into the active 3.10 interpreter (pure-python package, no compiled bits).
uv pip install --editable "$PROJECT_REPO" --no-deps --python-version 3.12

uv pip install tianshou==0.5.1 pydantic av zmq msgpack==1.1.0 msgpack-numpy==0.4.8 torchvision==0.22.0 transformers==4.57.3 tyro setuptools==80.9.0

# Sanity check
python - <<'PY'
from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
register_simpler_envs()
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
print("SimplerEnv import OK")
import gymnasium as gym
env = gym.make("simpler_env_google/google_robot_pick_object")
env.reset()
env.close()
print("Env OK:", type(env))
PY

echo "SimplerEnv ready at: $UV_ENV/.venv/bin/python3"
