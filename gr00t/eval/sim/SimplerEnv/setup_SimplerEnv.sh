#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
SIMPLER_REPO="$PROJECT_REPO/external_dependencies/SimplerEnv"
UV_ENV="$SCRIPT_DIR/simpler_uv"

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
BUILD_CONSTRAINTS="$UV_ENV/build-constraints.txt"
printf 'scikit-build-core<0.10\n' > "$BUILD_CONSTRAINTS"

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

uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e "$SIMPLER_REPO"

# gr00t pins python>=3.12, so editable-installing it into this 3.10 sim island
# fails dependency resolution. Drop a .pth instead: gr00t imports from the repo
# root using the island's own deps (the prior install was --no-deps anyway).
python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('$PROJECT_REPO').resolve().as_posix() + '\n')"

uv pip install \
  tianshou==0.5.1 \
  pydantic \
  av \
  zmq \
  torchvision==0.22.0 \
  transformers==4.57.3 \
  tyro \
  setuptools==80.9.0 \
  pandas==2.2.3 \
  dm-tree==0.1.9 \
  einops==0.8.1 \
  albumentations==1.4.18 \
  diffusers==0.35.1 \
  scipy==1.15.3

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
