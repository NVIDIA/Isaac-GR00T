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

import json
import os
import pathlib
import shutil
import subprocess
import zipfile

import pytest
from test_support.compass import (
    ISAACLAB_VENV,
    isaaclab_env,
    prepare_compass_repo,
    prepare_isaaclab,
    prepare_x_mobility,
)
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    TEST_CACHE_PATH,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    has_rt_core_gpu,
    hf_hub_download_cmd,
    run_subprocess_step,
    wait_for_server_ready,
)


REPO_ROOT = get_root()
TRAINING_STEPS = 2

README = REPO_ROOT / "examples/PointNav/README.md"

MODEL_CHECKPOINT = pathlib.Path(f"/tmp/pointnav_finetune/checkpoint-{TRAINING_STEPS}")
# HuggingFace source: nvidia/COMPASS model repo, file gr00t_post_training_g1.zip
_HF_REPO_ID = "nvidia/COMPASS"
_HF_FILENAME = "gr00t_post_training_g1.zip"
_DATASET_NAME = "lerobot_heading"

SHARED_POINTNAV_DIR = TEST_CACHE_PATH / "datasets/pointnav"
SHARED_POINTNAV_DATASET = SHARED_POINTNAV_DIR / _DATASET_NAME

# GR00T base model — downloaded once to shared storage and reused by the finetune step.
_GROOT_MODEL_REPO_ID = "nvidia/GR00T-N1.6-3B"
SHARED_GROOT_MODEL = TEST_CACHE_PATH / "models/GR00T-N1.6-3B"


def _dataset_ready(path: pathlib.Path) -> bool:
    """Return True when the PointNav dataset directory exists and is non-empty."""
    return path.is_dir() and any(path.iterdir())


def _groot_model_complete(path: pathlib.Path) -> bool:
    """Return True if the model directory has config.json and all shard files.

    Checks the shard index so that a partial previous download (e.g. missing
    model-00002-of-00002.safetensors) is detected and re-downloaded rather than
    causing a FileNotFoundError at load time.
    """
    if not (path / "config.json").is_file():
        return False
    index_file = path / "model.safetensors.index.json"
    if not index_file.is_file():
        return True  # single-shard model — config.json is sufficient
    shards = set(json.loads(index_file.read_text()).get("weight_map", {}).values())
    return all((path / shard).is_file() for shard in shards)


def _prepare_groot_model(env: dict[str, str]) -> pathlib.Path:
    """Return the GR00T-N1.6-3B model path, downloading to shared storage if needed.

    Using a pre-downloaded local copy avoids HF hub cache inconsistencies
    (e.g. stale index JSON with missing shard files) that cause
    ``AutoModel.from_pretrained`` to fail with "does not appear to have files named".

    Priority:
    1. GROOT_MODEL_PATH env var (user-supplied)
    2. Shared cache hit — all shards present
    3. snapshot_download into shared storage
    """
    env_path_str = os.environ.get("GROOT_MODEL_PATH", "")
    if env_path_str:
        env_path = pathlib.Path(env_path_str)
        if _groot_model_complete(env_path):
            return env_path

    if _groot_model_complete(SHARED_GROOT_MODEL):
        return SHARED_GROOT_MODEL

    token = os.environ.get("HF_TOKEN", "")
    assert token, "HF_TOKEN is required to download the gated nvidia/GR00T-N1.6-3B model"
    SHARED_GROOT_MODEL.mkdir(parents=True, exist_ok=True)
    run_subprocess_step(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={_GROOT_MODEL_REPO_ID!r}, "
            f"local_dir={str(SHARED_GROOT_MODEL)!r}, token={token!r})",
        ],
        step="groot_model_download",
        cwd=REPO_ROOT,
        env=env,
        log_prefix="pointnav",
        output_tail_chars=2000,
    )
    assert (SHARED_GROOT_MODEL / "config.json").is_file(), (
        f"GR00T model download succeeded but config.json not found at {SHARED_GROOT_MODEL}"
    )
    return SHARED_GROOT_MODEL


def _prepare_pointnav_dataset(env: dict[str, str]) -> pathlib.Path:
    """Return the PointNav dataset path, downloading and caching if needed.

    Priority:
    1. POINTNAV_DATASET_PATH env var (user-supplied, skip download entirely)
    2. Shared cache hit — reuse without re-downloading
    3. Download from HuggingFace, extract, and populate the shared cache
    """
    env_path_str = os.environ.get("POINTNAV_DATASET_PATH", "")
    if env_path_str:
        env_path = pathlib.Path(env_path_str)
        if env_path.exists():
            return env_path

    if _dataset_ready(SHARED_POINTNAV_DATASET):
        return SHARED_POINTNAV_DATASET

    SHARED_POINTNAV_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SHARED_POINTNAV_DIR / _HF_FILENAME

    run_subprocess_step(
        hf_hub_download_cmd(_HF_REPO_ID, _HF_FILENAME, str(SHARED_POINTNAV_DIR)),
        step="pointnav_dataset_download",
        cwd=REPO_ROOT,
        env=env,
        log_prefix="pointnav",
    )
    assert zip_path.exists(), (
        f"Expected zip at {zip_path} after download — hf_hub_download may have used a different path"
    )
    print(
        f"[pointnav] step=pointnav_dataset_extract extracting {zip_path} ({zip_path.stat().st_size} bytes)",
        flush=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SHARED_POINTNAV_DIR)
    zip_path.unlink(missing_ok=True)

    if _dataset_ready(SHARED_POINTNAV_DATASET):
        return SHARED_POINTNAV_DATASET

    # The zip may extract to a differently-named top-level directory.
    # Find the first non-empty subdirectory to help diagnose or recover.
    candidates = [p for p in SHARED_POINTNAV_DIR.iterdir() if p.is_dir() and any(p.iterdir())]
    if len(candidates) == 1:
        print(
            f"[pointnav] zip extracted to {candidates[0].name!r}, expected {_DATASET_NAME!r}; "
            "update _DATASET_NAME in the test to match.",
            flush=True,
        )
        return candidates[0]

    raise AssertionError(
        f"PointNav dataset not found at {SHARED_POINTNAV_DATASET} after download+extract.\n"
        f"Directories found in {SHARED_POINTNAV_DIR}: "
        f"{[p.name for p in SHARED_POINTNAV_DIR.iterdir() if p.is_dir()]}\n"
        f"Update _DATASET_NAME in the test to match the actual extracted directory name."
    )


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_pointnav_readme_finetune_executes_via_subprocess() -> None:
    """Run the PointNav README finetune script with a minimal step count."""

    env = build_shared_runtime_env(
        "pointnav",
        extra_env={
            "SAVE_STEPS": str(TRAINING_STEPS),
            "MAX_STEPS": str(TRAINING_STEPS),
            "USE_WANDB": "0",
            "DATALOADER_NUM_WORKERS": "0",
            # Limit to one GPU so HuggingFace Trainer uses plain single-device
            # mode instead of DataParallel, which breaks the model's device property.
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )

    dataset_path = _prepare_pointnav_dataset(env)
    groot_model_path = _prepare_groot_model(env)
    blocks = extract_code_blocks(README)

    # Wipe any leftover output dir so the trainer starts fresh rather than
    # trying to resume from a stale/incomplete checkpoint directory.
    if MODEL_CHECKPOINT.parent.exists():
        shutil.rmtree(MODEL_CHECKPOINT.parent)

    # Step 1: Finetune
    finetune_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    replace_once(
                        replace_once(
                            find_block(
                                blocks,
                                "--modality-config-path examples/PointNav/modality_config.py",
                                language="bash",
                            ).code,
                            "<dataset_path>",
                            str(dataset_path),
                        ),
                        "<output_dir>",
                        str(MODEL_CHECKPOINT.parent),
                    ),
                    "nvidia/GR00T-N1.6-3B",
                    str(groot_model_path),
                ),
                "MAX_STEPS=40000",
                f"MAX_STEPS={TRAINING_STEPS}",
            ),
            "SAVE_STEPS=2000",
            f"SAVE_STEPS={TRAINING_STEPS}",
        ),
        "GLOBAL_BATCH_SIZE=32",
        "GLOBAL_BATCH_SIZE=2",
    )
    run_bash_blocks([finetune_code], cwd=REPO_ROOT, env=env)

    assert MODEL_CHECKPOINT.exists(), (
        f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
    )

    # Step 2: Server — replace checkpoint placeholder and inject test-specific flags
    model_server_host = "127.0.0.1"
    model_server_port = 5553
    server_code = replace_once(
        find_block(blocks, "run_gr00t_server.py", language="bash").code,
        "<path/to/checkpoint>",
        str(MODEL_CHECKPOINT),
    )
    server_code += f" --host {model_server_host} --port {model_server_port}"

    # Step 3: Launch server, run COMPASS eval, then tear down.
    assert_port_available(model_server_host, model_server_port)
    model_server_proc = subprocess.Popen(
        ["bash", "-c", server_code],
        cwd=REPO_ROOT,
        env=env,
    )
    try:
        wait_for_server_ready(
            proc=model_server_proc,
            host=model_server_host,
            port=model_server_port,
            timeout_s=float(
                os.getenv("POINTNAV_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
            ),
        )

        # Step 4: COMPASS evaluation — runs against the live GR00T server.
        # Requires RT cores for Vulkan ray tracing; skipped on compute-only GPUs (e.g. B200).
        if has_rt_core_gpu():
            compass_repo = prepare_compass_repo(env)
            isaaclab_path = prepare_isaaclab(env)
            x_mobility_ckpt = prepare_x_mobility(compass_repo, env)

            run_subprocess_step(
                [
                    str(isaaclab_path / "isaaclab.sh"),
                    "-p",
                    "run.py",
                    "-c",
                    "configs/eval_config.gin",
                    "-o",
                    "/tmp/pointnav_compass_eval",
                    "-b",
                    str(x_mobility_ckpt),
                    "--enable_camera",
                    "--gr00t-policy",
                ],
                step="compass_eval",
                cwd=compass_repo,
                env=isaaclab_env(
                    env,
                    {
                        "ISAACLAB_PATH": str(isaaclab_path),
                        "VIRTUAL_ENV": str(ISAACLAB_VENV),
                        "PATH": f"{ISAACLAB_VENV / 'bin'}:{env.get('PATH', os.environ.get('PATH', ''))}",
                    },
                ),
                log_prefix="pointnav",
                failure_prefix="COMPASS evaluation failed",
                stream_output=True,
            )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)
