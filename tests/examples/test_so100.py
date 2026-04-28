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

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import TEST_CACHE_PATH, get_root, timed


logger = logging.getLogger(__name__)


REPO_ROOT = get_root()
TRAINING_STEPS = 2

README = REPO_ROOT / "examples/SO100/README.md"

DATASET_ROOT = REPO_ROOT / "examples/SO100/finish_sandwich_lerobot"
DATASET_PATH = DATASET_ROOT / "izuluaga/finish_sandwich"
MODALITY_SRC = REPO_ROOT / "examples/SO100/modality.json"
MODALITY_DST = DATASET_PATH / "meta/modality.json"
MODEL_CHECKPOINT = pathlib.Path(f"/tmp/so100_finetune/checkpoint-{TRAINING_STEPS}")

SHARED_DATASET_ROOT = TEST_CACHE_PATH / "datasets/so100_finish_sandwich"


def _dataset_ready(dataset_root: pathlib.Path) -> bool:
    """Return True when the converted SO100 dataset is present and non-empty."""
    inner = dataset_root / "izuluaga/finish_sandwich"
    info = inner / "meta/info.json"
    videos = inner / "videos"
    if not info.is_file() or not videos.is_dir():
        return False
    return next(videos.rglob("*.mp4"), None) is not None


def _point_to_shared() -> None:
    """Symlink DATASET_ROOT → SHARED_DATASET_ROOT."""
    if DATASET_ROOT.is_symlink():
        if DATASET_ROOT.resolve() == SHARED_DATASET_ROOT.resolve():
            return
        DATASET_ROOT.unlink()
    elif DATASET_ROOT.exists():
        return  # real local dataset — don't replace it
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.symlink_to(SHARED_DATASET_ROOT, target_is_directory=True)


def _prepare_so100_dataset(convert_block: str, convert_env: dict) -> None:
    """Download + convert the SO100 dataset, preferring shared cache when available."""
    if _dataset_ready(SHARED_DATASET_ROOT):
        _point_to_shared()
        return

    # Direct convert output to shared storage when the cache mount is present.
    convert_code = convert_block
    if TEST_CACHE_PATH.exists():
        convert_code = convert_code.replace(
            "examples/SO100/finish_sandwich_lerobot",
            str(SHARED_DATASET_ROOT),
        )

    run_bash_blocks([convert_code], cwd=REPO_ROOT, env=convert_env)

    if _dataset_ready(SHARED_DATASET_ROOT):
        _point_to_shared()
        return

    assert _dataset_ready(DATASET_ROOT), f"Expected SO100 dataset at {DATASET_ROOT}"


def _cleanup_dataset_path() -> None:
    """Remove the dataset directory created by the SO100 workflow."""
    try:
        if DATASET_ROOT.is_symlink():
            DATASET_ROOT.unlink()
        elif DATASET_ROOT.exists():
            shutil.rmtree(DATASET_ROOT)
    except OSError as exc:
        print(f"[so100] cleanup_warning path={DATASET_PATH} error={exc}", flush=True)


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_so100_readme_workflow_executes_via_subprocess() -> None:
    """Run the README's bash commands in order, with minor test-only substitutions."""

    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    print(f"[so100] uv_env={env.get('UV_PROJECT_ENVIRONMENT', '<unset>')}", flush=True)

    blocks = extract_code_blocks(README)

    try:
        # Step 1: Convert dataset (README: Handling the dataset)
        # The lerobot_conversion sub-project has its own dependencies (different
        # numpy/pyarrow versions).  Remove UV_PROJECT_ENVIRONMENT so uv creates
        # an isolated venv for it instead of contaminating the main one.
        convert_env = {k: v for k, v in env.items() if k != "UV_PROJECT_ENVIRONMENT"}
        with timed("step 1: dataset conversion"):
            _prepare_so100_dataset(
                find_block(blocks, "convert_v3_to_v2.py", language="bash").code,
                convert_env,
            )

        # Step 2: Copy modality.json (README cp command)
        MODALITY_DST.parent.mkdir(parents=True, exist_ok=True)
        with timed("step 2: modality.json copy"):
            run_bash_blocks(
                [find_block(blocks, "modality.json", language="bash")],
                cwd=REPO_ROOT,
                env=env,
            )
        assert MODALITY_DST.is_file(), f"Expected modality file after copy: {MODALITY_DST}"

        # Step 3: Finetune (README: Finetuning) — env overrides keep the run short
        finetune_code = (
            find_block(
                blocks,
                "--modality-config-path examples/SO100/so100_config.py",
                language="bash",
            ).code.rstrip()
            + " -- --skip_weight_loading"
        )
        with timed("step 3: finetune"):
            run_bash_blocks(
                [finetune_code],
                cwd=REPO_ROOT,
                env={
                    **env,
                    "SAVE_STEPS": str(TRAINING_STEPS),
                    "MAX_STEPS": str(TRAINING_STEPS),
                    "USE_WANDB": "0",
                    "DATALOADER_NUM_WORKERS": "0",
                    "GLOBAL_BATCH_SIZE": "2",
                    "SHARD_SIZE": "64",
                    "NUM_SHARDS_PER_EPOCH": "1",
                    "EPISODE_SAMPLING_RATE": "0.02",
                },
            )
        assert MODEL_CHECKPOINT.exists(), (
            f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
        )

        # Step 4: Open-loop eval — replace README defaults with test-specific values
        eval_cmd = replace_once(
            replace_once(
                find_block(blocks, "open_loop_eval.py", language="bash").code,
                "/tmp/so100_finetune/checkpoint-10000",
                str(MODEL_CHECKPOINT),
            ),
            "--steps 400",
            "--steps 5",
        )
        with timed("step 4: open-loop eval"):
            run_bash_blocks([eval_cmd], cwd=REPO_ROOT, env=env)
        assert pathlib.Path("/tmp/open_loop_eval/traj_0.jpeg").exists(), (
            "Expected eval plot at /tmp/open_loop_eval/traj_0.jpeg"
        )
    finally:
        _cleanup_dataset_path()
