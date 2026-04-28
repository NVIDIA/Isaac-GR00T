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

"""Setup helpers for COMPASS, IsaacLab, and X-Mobility."""

from __future__ import annotations

import os
import pathlib

from test_support.runtime import TEST_CACHE_PATH, get_root, hf_hub_download_cmd, run_subprocess_step


_REPO_ROOT = get_root()

# COMPASS repo — cloned directly into shared storage (not a git submodule).
_COMPASS_GIT_URL = "https://github.com/NVlabs/COMPASS.git"
SHARED_COMPASS_REPO = TEST_CACHE_PATH / "repos/COMPASS"

# IsaacLab repo + installation — cloned and installed into shared storage.
_ISAACLAB_GIT_URL = "https://github.com/isaac-sim/IsaacLab.git"
_ISAACLAB_PIP_INDEX = "https://pypi.nvidia.com"
_ISAACLAB_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
SHARED_ISAACLAB_REPO = TEST_CACHE_PATH / "repos/IsaacLab"

# Dedicated venv for the IsaacLab/Isaac Sim Python environment.
# Must live on local disk (/tmp) rather than the shared NFS drive because the
# aarch64 torch wheel (~2.8 GB) fails with Errno 2 (ENOENT) when pip tries to
# move extracted .so files across filesystem boundaries.
ISAACLAB_VENV = pathlib.Path("/tmp/isaaclab_venv")

# X-Mobility: wheel ships inside the COMPASS repo; checkpoint from HuggingFace.
_X_MOBILITY_WHL = "x_mobility/x_mobility-0.1.0-py3-none-any.whl"
_X_MOBILITY_HF_REPO = "nvidia/X-Mobility"
_X_MOBILITY_HF_FILE = "x_mobility-nav2-semantic_action_path.ckpt"
SHARED_X_MOBILITY_CKPT = TEST_CACHE_PATH / f"models/x_mobility/{_X_MOBILITY_HF_FILE}"


def isaaclab_env(env: dict[str, str], extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a clean env for running isaaclab.sh commands.

    Strips VIRTUAL_ENV, UV_PROJECT_ENVIRONMENT, and the workspace venv bin
    directory from PATH so isaaclab.sh -p uses its own Isaac Sim Python rather
    than the workspace venv (which lacks pip).
    """
    venv = env.get("UV_PROJECT_ENVIRONMENT") or env.get("VIRTUAL_ENV") or ""
    result = {}
    for k, v in env.items():
        if k in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            continue
        if k == "PATH" and venv:
            venv_bin = str(pathlib.Path(venv) / "bin")
            # Remove "<venv_bin>:" and any leading/trailing colons.
            v = v.replace(f"{venv_bin}:", "").replace(f":{venv_bin}", "").replace(venv_bin, "")
        result[k] = v
    result["TERM"] = "xterm-256color"
    if extra:
        result.update(extra)
    return result


def prepare_isaaclab(env: dict[str, str]) -> pathlib.Path:
    """Return the IsaacLab repo path, cloning and installing into shared storage if needed.

    Priority:
    1. ISAACLAB_PATH env var (user-supplied, assumed already installed)
    2. Shared cache hit — repo + install already done
    3. Clone from GitHub and run ``./isaaclab.sh --install``
    """
    env_path_str = os.environ.get("ISAACLAB_PATH", "")
    if env_path_str:
        env_path = pathlib.Path(env_path_str)
        if (env_path / "isaaclab.sh").is_file():
            return env_path

    # Require both the repo AND the Isaac Sim installation (_isaac_sim/python.sh)
    # to exist before treating the cache as ready.  The repo may be cloned but
    # the install step may have failed, leaving isaaclab.sh present but the
    # Isaac Sim Python environment absent.
    if (SHARED_ISAACLAB_REPO / "_isaac_sim" / "python.sh").is_file():
        return SHARED_ISAACLAB_REPO

    SHARED_ISAACLAB_REPO.parent.mkdir(parents=True, exist_ok=True)
    if not SHARED_ISAACLAB_REPO.exists():
        run_subprocess_step(
            ["git", "clone", _ISAACLAB_GIT_URL, str(SHARED_ISAACLAB_REPO)],
            step="isaaclab_clone",
            cwd=_REPO_ROOT,
            env=env,
            log_prefix="compass",
        )
    # isaaclab.sh --install requires an active Python env to install Isaac Sim
    # into.  The venv must live on LOCAL disk, not on the shared NFS drive:
    # the aarch64 torch wheel bundles libcublas.so.12 (~2.8 GB) and NFS
    # filesystems fail with Errno 2 (ENOENT) when pip tries to move the
    # extracted .so files from the temp staging area to the venv site-packages.
    # Isaac Sim 5.x Kit extensions are built for cp311 only; a Python 3.10 venv
    # causes "platform incompatible" failures when resolving omni.* dependencies.
    # Use `uv venv --python 3.11` with UV_PYTHON_DOWNLOADS=automatic so uv
    # downloads Python 3.11 if it is not already installed on the runner.
    run_subprocess_step(
        ["uv", "venv", "--python", "3.11", "--seed", str(ISAACLAB_VENV)],
        step="isaaclab_venv_create",
        cwd=SHARED_ISAACLAB_REPO,
        env={**env, "UV_PYTHON_DOWNLOADS": "automatic"},
        log_prefix="compass",
    )

    # Pre-install torch/torchvision via uv pip so isaaclab.sh sees them already
    # satisfied and skips its own download step.  Use uv pip rather than plain pip
    # so the install benefits from uv's wheel cache on the shared drive.
    # Clear PIP_CONSTRAINT so the NVIDIA-custom torch build constraint in GB200
    # containers doesn't block IsaacLab's torch version.
    _venv_env = {**env, "PIP_CONSTRAINT": ""}
    _pip = str(ISAACLAB_VENV / "bin" / "pip")
    _venv_python = str(ISAACLAB_VENV / "bin" / "python")
    # Use uv pip for the torch pre-install so it benefits from uv's wheel cache.
    # uv also handles the aarch64 wheel name-casing correctly, so the separate
    # pip-upgrade step that was needed to fix old pip's resolution bug is gone.
    run_subprocess_step(
        [
            "uv",
            "pip",
            "install",
            "--python",
            _venv_python,
            "torch==2.7.0",
            "torchvision==0.22.0",
            "--index-url",
            _ISAACLAB_TORCH_INDEX,
            "--extra-index-url",
            _ISAACLAB_PIP_INDEX,
        ],
        step="isaaclab_torch_preinstall",
        cwd=SHARED_ISAACLAB_REPO,
        env={**_venv_env, "UV_PYTHON_DOWNLOADS": "automatic"},
        log_prefix="compass",
        stream_output=True,
    )

    # isaacsim-rl creates the _isaac_sim symlink inside the IsaacLab repo.
    # setup_vscode.py (called by isaaclab.sh --install) requires that symlink;
    # without it the install fails with "Could not find the isaac-sim directory".
    # ACCEPT_EULA=Y suppresses the interactive NVIDIA Omniverse EULA prompt that
    # the isaacsim-rl post-install script would otherwise block waiting for input.
    run_subprocess_step(
        [_pip, "install", "isaacsim-rl", "--extra-index-url", _ISAACLAB_PIP_INDEX],
        step="isaaclab_isaacsim_install",
        cwd=SHARED_ISAACLAB_REPO,
        env={**_venv_env, "ACCEPT_EULA": "Y"},
        log_prefix="compass",
        stream_output=True,
    )

    run_subprocess_step(
        [str(SHARED_ISAACLAB_REPO / "isaaclab.sh"), "--install"],
        step="isaaclab_install",
        cwd=SHARED_ISAACLAB_REPO,
        env=isaaclab_env(
            env,
            {
                "VIRTUAL_ENV": str(ISAACLAB_VENV),
                "PATH": f"{ISAACLAB_VENV / 'bin'}:{env.get('PATH', os.environ.get('PATH', ''))}",
                "PIP_EXTRA_INDEX_URL": _ISAACLAB_PIP_INDEX,
                # GB200 containers carry a system pip constraint that pins torch to
                # an NVIDIA-custom build.  Clear it so IsaacLab can install its own
                # torch version into the dedicated venv.
                "PIP_CONSTRAINT": "",
                # Auto-accept the NVIDIA Omniverse EULA without interactive prompt.
                "ACCEPT_EULA": "Y",
            },
        ),
        log_prefix="compass",
        stream_output=True,
    )
    return SHARED_ISAACLAB_REPO


def prepare_compass_repo(env: dict[str, str]) -> pathlib.Path:
    """Return the COMPASS repo path, cloning into shared storage if needed.

    Priority:
    1. COMPASS_REPO_PATH env var (user-supplied)
    2. Shared cache hit — reuse without re-cloning
    3. Clone from GitHub into shared storage
    """
    env_path_str = os.environ.get("COMPASS_REPO_PATH", "")
    if env_path_str:
        env_path = pathlib.Path(env_path_str)
        if (env_path / "run.py").is_file():
            return env_path

    if (SHARED_COMPASS_REPO / "run.py").is_file():
        return SHARED_COMPASS_REPO

    SHARED_COMPASS_REPO.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess_step(
        ["git", "clone", _COMPASS_GIT_URL, str(SHARED_COMPASS_REPO)],
        step="compass_repo_clone",
        cwd=_REPO_ROOT,
        env=env,
        log_prefix="compass",
    )
    return SHARED_COMPASS_REPO


def prepare_x_mobility(
    compass_repo: pathlib.Path,
    env: dict[str, str],
) -> pathlib.Path:
    """Install the X-Mobility wheel and return the cached checkpoint path.

    The wheel ships inside the COMPASS repo and is installed into the IsaacLab
    Python environment. The checkpoint is downloaded from HuggingFace once and
    cached in shared storage.
    """
    # isaaclab.sh -p requires _isaac_sim/python.sh which only exists in the full
    # Isaac Sim desktop install, not the pip-based isaacsim-rl package.  Install
    # X-Mobility directly into ISAACLAB_VENV (where isaacsim-rl lives) instead.
    _pip = str(ISAACLAB_VENV / "bin" / "pip")
    run_subprocess_step(
        [_pip, "install", "--verbose", str(compass_repo / _X_MOBILITY_WHL)],
        step="x_mobility_install",
        cwd=compass_repo,
        env={**env, "PIP_CONSTRAINT": ""},
        log_prefix="compass",
        stream_output=True,
    )

    if not SHARED_X_MOBILITY_CKPT.is_file():
        SHARED_X_MOBILITY_CKPT.parent.mkdir(parents=True, exist_ok=True)
        run_subprocess_step(
            hf_hub_download_cmd(
                _X_MOBILITY_HF_REPO, _X_MOBILITY_HF_FILE, str(SHARED_X_MOBILITY_CKPT.parent)
            ),
            step="x_mobility_ckpt_download",
            cwd=_REPO_ROOT,
            env=env,
            log_prefix="compass",
        )

    return SHARED_X_MOBILITY_CKPT
