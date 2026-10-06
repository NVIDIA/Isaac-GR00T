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

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    TEST_CACHE_PATH,
    assert_port_available,
    demo_dataset_tree_ready,
    get_root,
    resolve_model_checkpoint_path,
    run_subprocess_step,
    start_server_process,
    timed,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

LOGGER = logging.getLogger(__name__)


README = REPO_ROOT / "examples/robocasa-gr1-tabletop-tasks/README.md"
ROBOCASA_GR1_CHECKPOINT_PATH = os.environ.get("ROBOCASA_GR1_CHECKPOINT_PATH", "")
ROBOCASA_GR1_EMBODIMENT_TAG = os.environ.get("ROBOCASA_GR1_EMBODIMENT_TAG", "ROBOCASA_GR1_TABLETOP")
TRAINING_STEPS = 2
MODEL_CHECKPOINT = pathlib.Path(f"/tmp/gr1_tabletop_finetune/checkpoint-{TRAINING_STEPS}")
_ROBOCASA_GR1_DATASET_ENV = "ROBOCASA_GR1_DATASET_PATH"
_SHARED_ROBOCASA_GR1_DATASET = TEST_CACHE_PATH / "datasets/robocasa-gr1-tabletop-tasks"

ROBOCASA_SUBMODULE_PATH = REPO_ROOT / "external_dependencies/robocasa-gr1-tabletop-tasks"
SHARED_ROBOCASA_REPO = TEST_CACHE_PATH / "repos/robocasa-gr1-tabletop-tasks"

ROBOCASA_ASSETS_REPO_DIR = (
    REPO_ROOT / "external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
)
ROBOCASA_ASSETS_SHARED_DIR = TEST_CACHE_PATH / "robocasa-gr1-tabletop-tasks/assets"
# Version file written alongside cached assets to detect submodule updates.
_ASSETS_VERSION_FILE = ROBOCASA_ASSETS_SHARED_DIR / ".robocasa_gr1_commit"


def _robocasa_submodule_commit() -> str:
    """Return the robocasa-gr1-tabletop-tasks submodule commit hash recorded in HEAD."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "HEAD", "external_dependencies/robocasa-gr1-tabletop-tasks"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        parts = result.stdout.split()
        return parts[2] if len(parts) >= 3 else "unknown"
    except Exception:
        return "unknown"


def _shared_asset_dirs() -> list[pathlib.Path]:
    """Return all top-level subdirectories in the shared asset cache."""
    if not ROBOCASA_ASSETS_SHARED_DIR.is_dir():
        return []
    return [p for p in ROBOCASA_ASSETS_SHARED_DIR.iterdir() if p.is_dir()]


def _shared_assets_ready() -> bool:
    """Return True when the shared asset cache is present, non-empty, and version-matched."""
    if not _ASSETS_VERSION_FILE.is_file():
        return False
    if _ASSETS_VERSION_FILE.read_text().strip() != _robocasa_submodule_commit():
        return False
    for d in _shared_asset_dirs():
        try:
            if next((f for f in d.rglob("*") if f.is_file()), None) is not None:
                return True
        except OSError:
            pass
    return False


def _assert_required_assets_present() -> None:
    """Raise if the repo asset directory is empty."""
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir() or not any(ROBOCASA_ASSETS_REPO_DIR.iterdir()):
        raise RuntimeError(f"RoboCasa GR1 assets missing at {ROBOCASA_ASSETS_REPO_DIR}")


def _point_repo_assets_to_shared() -> None:
    """Symlink all shared asset subdirectories into the repo asset path."""
    ROBOCASA_ASSETS_REPO_DIR.mkdir(parents=True, exist_ok=True)
    for shared_dir in _shared_asset_dirs():
        repo_dir = ROBOCASA_ASSETS_REPO_DIR / shared_dir.name
        if repo_dir.is_symlink():
            if repo_dir.resolve() == shared_dir.resolve():
                continue
            repo_dir.unlink()
        elif repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.symlink_to(shared_dir, target_is_directory=True)


def _move_repo_assets_to_shared() -> None:
    """Move all downloaded repo asset subdirectories into the shared cache."""
    ROBOCASA_ASSETS_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir():
        return
    for src in ROBOCASA_ASSETS_REPO_DIR.iterdir():
        if not src.is_dir() or src.is_symlink():
            continue
        dst = ROBOCASA_ASSETS_SHARED_DIR / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        # Use cp -r + rm -rf instead of shutil.move: the repo and the shared
        # PVC may be on different filesystems, so shutil.move would fall back
        # to a slow Python-level copytree.
        subprocess.run(["cp", "-r", str(src), str(dst)], check=True)
        shutil.rmtree(str(src))
    _ASSETS_VERSION_FILE.write_text(_robocasa_submodule_commit())


def _remove_dangling_repo_asset_symlinks() -> None:
    """Delete repo asset symlinks that point to missing targets."""
    if not ROBOCASA_ASSETS_REPO_DIR.is_dir():
        return
    for repo_dir in ROBOCASA_ASSETS_REPO_DIR.iterdir():
        if repo_dir.is_symlink() and not repo_dir.exists():
            repo_dir.unlink()


def _robocasa_submodule_initialized() -> bool:
    return (ROBOCASA_SUBMODULE_PATH / ".git").is_file()


def _git_modules_path(submodule_path: pathlib.Path) -> pathlib.Path | None:
    git_file = submodule_path / ".git"
    if not git_file.is_file():
        return None
    content = git_file.read_text().strip()
    if not content.startswith("gitdir:"):
        return None
    rel = content[len("gitdir:") :].strip()
    return (submodule_path / rel).resolve()


def _prepare_robocasa_repo(env: dict[str, str]) -> None:
    """Populate external_dependencies/robocasa-gr1-tabletop-tasks from cache, or init+cache."""
    if _robocasa_submodule_initialized():
        return

    wt_cache = SHARED_ROBOCASA_REPO / "wt"
    modules_cache = SHARED_ROBOCASA_REPO / "modules"

    if (wt_cache / ".git").is_file() and modules_cache.exists():
        print(f"[robocasa-gr1] restoring submodule from cache {wt_cache}", flush=True)
        shutil.copytree(wt_cache, ROBOCASA_SUBMODULE_PATH, dirs_exist_ok=True)
        modules_path = _git_modules_path(ROBOCASA_SUBMODULE_PATH)
        if modules_path is not None:
            modules_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_cache, modules_path, dirs_exist_ok=True)
        return

    if ROBOCASA_SUBMODULE_PATH.exists() and not _robocasa_submodule_initialized():
        shutil.rmtree(ROBOCASA_SUBMODULE_PATH)

    run_subprocess_step(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "external_dependencies/robocasa-gr1-tabletop-tasks",
        ],
        step="robocasa_gr1_repo_init",
        cwd=REPO_ROOT,
        env=env,
        log_prefix="robocasa-gr1",
    )
    if TEST_CACHE_PATH.exists():
        modules_path = _git_modules_path(ROBOCASA_SUBMODULE_PATH)
        print(f"[robocasa-gr1] caching submodule to {wt_cache}", flush=True)
        wt_cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROBOCASA_SUBMODULE_PATH, wt_cache, dirs_exist_ok=True)
        if modules_path is not None:
            modules_cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_path, modules_cache, dirs_exist_ok=True)


def _build_runtime_env(skip_download_assets: str) -> dict[str, str]:
    """Build the runtime environment used by setup, model server, and rollout."""
    return {**os.environ, "SKIP_DOWNLOAD_ASSETS": skip_download_assets, "INSTALL_FLASH_ATTN": "0"}


def _resolve_robocasa_gr1_dataset() -> pathlib.Path:
    """Return the RoboCasa GR1 LeRobot dataset used for short finetuning."""
    env_path_str = os.environ.get(_ROBOCASA_GR1_DATASET_ENV, "").strip()
    if env_path_str:
        env_path = pathlib.Path(env_path_str).expanduser().resolve()
        assert demo_dataset_tree_ready(env_path), (
            f"{_ROBOCASA_GR1_DATASET_ENV} does not point to a complete LeRobot dataset: {env_path}"
        )
        return env_path

    if demo_dataset_tree_ready(_SHARED_ROBOCASA_GR1_DATASET):
        return _SHARED_ROBOCASA_GR1_DATASET

    pytest.skip(
        "RoboCasa GR1 finetune dataset not found. Set "
        f"{_ROBOCASA_GR1_DATASET_ENV} or populate {_SHARED_ROBOCASA_GR1_DATASET}."
    )


def _resolve_robocasa_gr1_checkpoint(blocks, env: dict[str, str]) -> str:
    """Use ROBOCASA_GR1_CHECKPOINT_PATH or create a short fine-tuned checkpoint."""
    if ROBOCASA_GR1_CHECKPOINT_PATH:
        return ROBOCASA_GR1_CHECKPOINT_PATH

    dataset_path = _resolve_robocasa_gr1_dataset()
    with timed("step 1a: base model prep"):
        base_model_path = resolve_model_checkpoint_path(
            hf_repo_id="nvidia/GR00T-N1.7-3B",
            path_override_env="GROOT_MODEL_PATH",
            repo_root=REPO_ROOT,
        )

    if MODEL_CHECKPOINT.parent.exists():
        shutil.rmtree(MODEL_CHECKPOINT.parent)

    finetune_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    replace_once(
                        replace_once(
                            replace_once(
                                replace_once(
                                    find_block(
                                        blocks,
                                        "--output-dir /tmp/gr1_tabletop_finetune",
                                        language="bash",
                                    ).code,
                                    "NUM_GPUS=8",
                                    "NUM_GPUS=1",
                                ),
                                "MAX_STEPS=60000",
                                f"MAX_STEPS={TRAINING_STEPS}",
                            ),
                            "SAVE_STEPS=2000",
                            f"SAVE_STEPS={TRAINING_STEPS}",
                        ),
                        "GLOBAL_BATCH_SIZE=512",
                        "GLOBAL_BATCH_SIZE=2",
                    ),
                    "nvidia/GR00T-N1.7-3B",
                    str(base_model_path),
                ),
                "<path-to-gr1-tabletop-lerobot-dataset>",
                str(dataset_path),
            ),
            "--embodiment-tag ROBOCASA_GR1_TABLETOP",
            f"--embodiment-tag {ROBOCASA_GR1_EMBODIMENT_TAG}",
        ),
        "--output-dir /tmp/gr1_tabletop_finetune",
        f"--output-dir {MODEL_CHECKPOINT.parent}",
    )
    finetune_code = finetune_code.rstrip() + " -- --skip_weight_loading"
    with timed("step 1b: finetune"):
        run_bash_blocks(
            [finetune_code],
            cwd=REPO_ROOT,
            env={
                **env,
                "USE_WANDB": "0",
                "DATALOADER_NUM_WORKERS": "0",
                "SHARD_SIZE": "64",
                "NUM_SHARDS_PER_EPOCH": "1",
            },
        )
    assert MODEL_CHECKPOINT.exists(), (
        f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
    )
    return str(MODEL_CHECKPOINT)


@pytest.mark.gpu
@pytest.mark.timeout(2700)
def test_robocasa_gr1_tabletop_readme_eval_flow() -> None:
    """Run the RoboCasa GR1 Tabletop README finetune/server/client eval flow."""

    shared_assets_ready = _shared_assets_ready()
    if shared_assets_ready:
        _point_repo_assets_to_shared()
    else:
        _remove_dangling_repo_asset_symlinks()

    skip_download_assets = "1" if shared_assets_ready else "0"
    env = _build_runtime_env(skip_download_assets=skip_download_assets)
    blocks = extract_code_blocks(README)

    with timed("step 1: checkpoint prep"):
        checkpoint_path = _resolve_robocasa_gr1_checkpoint(blocks, env)

    with timed("step 2: robocasa-gr1 repo prep"):
        _prepare_robocasa_repo(env)

    with timed("step 3: sim venv setup (setup_RoboCasaGR1TabletopTasks.sh)"):
        run_bash_blocks(
            [find_block(blocks, "setup_RoboCasaGR1TabletopTasks.sh", language="bash")],
            cwd=REPO_ROOT,
            env=env,
            force_yes=True,
        )

    if not shared_assets_ready:
        _move_repo_assets_to_shared()
        _point_repo_assets_to_shared()

    _assert_required_assets_present()

    model_server_host = "127.0.0.1"
    model_server_port = 5556

    # Step 4: Server — N1.7 RoboCasa GR1 evaluation requires a finetuned checkpoint.
    server_code = replace_once(
        replace_once(
            find_block(blocks, "<path-to-finetuned-gr1-tabletop-checkpoint>", language="bash").code,
            "<path-to-finetuned-gr1-tabletop-checkpoint>",
            checkpoint_path,
        ),
        "--embodiment-tag ROBOCASA_GR1_TABLETOP",
        f"--embodiment-tag {ROBOCASA_GR1_EMBODIMENT_TAG}",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    # Step 5: Rollout — substitute test-safe values
    rollout_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "rollout_policy.py", language="bash").code,
                    "--n-episodes 10",
                    "--n-episodes 1",
                ),
                "--policy-client-port 5555",
                f"--policy-client-port {model_server_port}",
            ),
            "--max-episode-steps 720",
            "--max-episode-steps 2",
        ),
        "--n-envs 5",
        "--n-envs 1",
    )

    assert_port_available(model_server_host, model_server_port)
    model_server_proc, server_log = start_server_process(server_code, cwd=REPO_ROOT, env=env)
    with timed("step 4: server startup"):
        wait_for_server_ready(
            proc=model_server_proc,
            host=model_server_host,
            port=model_server_port,
            timeout_s=float(
                os.getenv(
                    "ROBOCASA_GR1_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS)
                )
            ),
            server_log=server_log,
        )

    try:
        with timed("step 5: rollout"):
            simulation_result, _ = run_subprocess_step(
                ["bash", "-c", rollout_code],
                step="robocasa_gr1_rollout",
                cwd=REPO_ROOT,
                env=env,
                log_prefix="robocasa-gr1",
                failure_prefix="RoboCasa GR1 Tabletop rollout failed",
                output_tail_chars=4000,
            )
        simulation_output = (simulation_result.stdout or "") + (simulation_result.stderr or "")
        assert "results:" in simulation_output, (
            "Simulation output did not include expected 'results:' marker.\n"
            f"output_tail=\n{simulation_output[-4000:]}"
        )
        assert "success rate:" in simulation_output, (
            "Simulation output did not include expected 'success rate:' marker.\n"
            f"output_tail=\n{simulation_output[-4000:]}"
        )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)
