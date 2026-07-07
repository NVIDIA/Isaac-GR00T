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
uv venv $LIBERO_UV_ENV/.venv --python 3.12
source $LIBERO_UV_ENV/.venv/bin/activate
# LIBERO's pinned requirements predate Python 3.12. Patch only the pins that
# otherwise build from source or pull source-only transitive deps on py3.12.
PATCHED_REQUIREMENTS="$LIBERO_UV_ENV/requirements-py312.txt"
python - <<PY
from pathlib import Path

replacements = {
    "hydra-core": "hydra-core==1.3.2",
    "numpy": "numpy==1.26.4",
    "transformers": "transformers==4.57.3",
    "opencv-python": "opencv-python==4.10.0.84",
    "matplotlib": "matplotlib==3.9.4",
    "wandb": "wandb==0.18.7",  # py3.12: 0.13.1 -> pathtools -> removed `imp`
}

src = Path("$LIBERO_REPO/requirements.txt")
dst = Path("$PATCHED_REQUIREMENTS")
lines = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    name = stripped.split("==", 1)[0].strip().lower()
    lines.append(replacements.get(name, raw))
dst.write_text("\n".join(lines) + "\n")
PY
uv pip install --requirements $PATCHED_REQUIREMENTS
uv pip install -e $LIBERO_REPO --config-settings editable_mode=compat
# py3.12 pins: stop the resolver backtracking numba/llvmlite to the 3.10-only build
uv pip install torch==2.9.0 torchvision==0.24.0 pydantic av tianshou==0.5.1 numba==0.65.1 llvmlite==0.47.0 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
# Pin mujoco: robosuite 1.4.0 (pulled by LIBERO's requirements) calls
# mj_fullM(model, dst, M), whose signature changed in mujoco 3.10.0 (2026-06-22)
# to mj_fullM(model, data, dst). mujoco is otherwise unpinned here, so it floats
# to the latest release and crashes env creation. Pin below the break (matches
# the RoboCasa island's existing mujoco==3.3.1 pin).
uv pip install numpy==1.26.4 mujoco==3.3.1

# Expose gr00t from the repo root via a .pth: no dependency re-resolution, and
# the island supplies gr00t's runtime deps itself (matches the old --no-deps).
python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('$PROJECT_REPO').resolve().as_posix() + '\n')"

rm -rf $HOME/.libero
printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs"
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY

#final_info -> 2.9.1 -> final_info
