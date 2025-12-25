#!/usr/bin/env bash
set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
LIBERO_REPO="$SCRIPT_DIR/../../../../external_dependencies/LIBERO"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"

git submodule update --init $LIBERO_REPO

# python -m pip install cmake==3.18.4
rm -rf $LIBERO_UV_ENV
mkdir -p $LIBERO_UV_ENV
uv venv $LIBERO_UV_ENV/.venv --python 3.10
source $LIBERO_UV_ENV/.venv/bin/activate
# uv pip install gymnasium==1.2.0 # -> 2.9.1 -> 
uv pip install --requirements $LIBERO_REPO/requirements.txt
uv pip install -e $LIBERO_REPO --config-settings editable_mode=compat
uv pip install --editable $PROJECT_REPO --no-deps
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install numpy==1.26.4

uv pip install --editable "$PROJECT_REPO" --no-deps

rm -rf $HOME/.libero
echo "y\n" | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs"
python - <<'PY'
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY

#final_info -> 2.9.1 -> final_info