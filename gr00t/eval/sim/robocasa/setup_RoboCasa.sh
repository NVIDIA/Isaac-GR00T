#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
ROBOCASA_SETUP_VARIANT="${ROBOCASA_SETUP_VARIANT:-robocasa}"

COMMON_DEPS=(
  gymnasium==0.29.1
  pydantic
  av==15.0.0
  zmq
  transformers==4.57.3
  msgpack==1.1.0
  msgpack-numpy==0.4.8
  tyro==1.0.13
)
ROBOCASA365_DEPS=(
  numpy==2.2.5
  numba
  scipy
  mujoco==3.3.1
  pygame
  Pillow
  opencv-python
  pyyaml
  pynput
  tqdm
  termcolor
  imageio
  h5py
  lxml
  hidapi
  tianshou
  loguru==0.7.3
  tenacity==9.1.4
  sqlalchemy==2.0.50
  psycopg2-binary==2.9.12
  openai==2.41.0
  ray==2.55.1
  pandas==2.2.3
  diffusers==0.35.1
  albumentations==1.4.18
  dm-tree==0.1.9
)
ASSET_DOWNLOAD_ARGS=()
ASSETS_CACHE_ROOT=""
INSTALL_ROBOCASA_NO_DEPS=0

case "$ROBOCASA_SETUP_VARIANT" in
  robocasa)
    ROBOCASA_REPO="$PROJECT_REPO/external_dependencies/robocasa"
    ROBOCASA_PATH="external_dependencies/robocasa"
    UV_ENV="$SCRIPT_DIR/robocasa_uv"
    SANITY_GYM_IMPORT="import robocasa.utils.gym_utils.gymnasium_groot"
    SANITY_ENV_ID="robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env"

    if [ ! -e "$ROBOCASA_REPO/.git" ]; then
        if ! git -C "$PROJECT_REPO" submodule update --init "$ROBOCASA_PATH"; then
            # Cache setup can leave asset directories under the submodule path
            # before the submodule is initialized. Git refuses to clone into that
            # non-empty directory, so clear the pre-submodule path and retry.
            rm -rf "$ROBOCASA_REPO"
            git -C "$PROJECT_REPO" submodule update --init "$ROBOCASA_PATH"
        fi
    fi
    ;;
  robocasa365)
    ROBOCASA_REPO="$PROJECT_REPO/external_dependencies/robocasa365"
    ROBOCASA365_PIN="${ROBOCASA365_PIN:-be22d659b02db8f6d7f3a3c3edc742934fdcbaae}"
    ASSETS_CACHE_ROOT="${ROBOCASA365_ASSETS_CACHE_ROOT:-}"
    UV_ENV="$PROJECT_REPO/gr00t/eval/sim/robocasa365/robocasa365_uv"
    INSTALL_ROBOCASA_NO_DEPS=1
    ASSET_DOWNLOAD_ARGS=(--type tex tex_generative fixtures_lw objs_lw objs_objaverse objs_aigen)
    SANITY_GYM_IMPORT="import gr00t.eval.sim.robocasa365.gymnasium_groot"
    SANITY_ENV_ID="robocasa365_panda_omron/CloseFridge_PandaOmron_Env"

    if [ ! -e "$ROBOCASA_REPO/.git" ]; then
        git clone https://github.com/robocasa/robocasa.git "$ROBOCASA_REPO"
    fi

    if [ "$(git -C "$ROBOCASA_REPO" rev-parse HEAD)" != "$ROBOCASA365_PIN" ]; then
        git -C "$ROBOCASA_REPO" fetch origin "$ROBOCASA365_PIN" || git -C "$ROBOCASA_REPO" fetch origin
        git -C "$ROBOCASA_REPO" checkout "$ROBOCASA365_PIN"
    fi
    ;;
  *)
    echo "Unknown ROBOCASA_SETUP_VARIANT: $ROBOCASA_SETUP_VARIANT" >&2
    exit 1
    ;;
esac

rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install setuptools wheel

uv pip install torch==2.5.1 torchvision==0.20.1
# Linux-only: preinstall flash-attn to avoid compiling inside other wheels
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
if [[ "$(uname -s)" == "Linux" && "$INSTALL_FLASH_ATTN" == "1" ]]; then
  uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
fi

uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@master"
if [ "$INSTALL_ROBOCASA_NO_DEPS" = "1" ]; then
    uv pip install -e "$ROBOCASA_REPO" --no-deps --config-settings editable_mode=compat
else
    uv pip install -e "$ROBOCASA_REPO" --config-settings editable_mode=compat
fi

uv pip install "${COMMON_DEPS[@]}"
if [ "$ROBOCASA_SETUP_VARIANT" = "robocasa365" ]; then
    uv pip install "${ROBOCASA365_DEPS[@]}"
fi

# Make your project importable in this venv without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# Assets for RoboCasa (kitchen)
SKIP_DOWNLOAD_ASSETS=${SKIP_DOWNLOAD_ASSETS:-0}
if [[ "$SKIP_DOWNLOAD_ASSETS" == "1" && -n "$ASSETS_CACHE_ROOT" ]]; then
    ROBOCASA_ASSETS_REPO_DIR="$ROBOCASA_REPO/robocasa/models/assets"
    mkdir -p "$ROBOCASA_ASSETS_REPO_DIR"
    for shared_dir in "$ASSETS_CACHE_ROOT"/*/; do
        [ -d "$shared_dir" ] || continue
        name=$(basename "$shared_dir")
        repo_dir="$ROBOCASA_ASSETS_REPO_DIR/$name"
        [ -L "$repo_dir" ] && rm "$repo_dir"
        [ -e "$repo_dir" ] && rm -rf "$repo_dir"
        ln -s "$shared_dir" "$repo_dir"
    done
elif [[ "$SKIP_DOWNLOAD_ASSETS" == "0" ]]; then
    printf 'y\n' | python "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py" "${ASSET_DOWNLOAD_ARGS[@]}"
fi

# Sanity import & env construction
python - <<PY
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym
import robocasa
import robosuite
${SANITY_GYM_IMPORT}
print("Imports OK:", robosuite.__version__)
env = gym.make("${SANITY_ENV_ID}", enable_render=True)
print("Env OK:", type(env))
env.close()
PY
