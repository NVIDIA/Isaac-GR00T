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

import os
import pathlib
import subprocess

import pytest
from test_support.readme import extract_code_blocks, find_block, replace_once, run_bash_blocks
from test_support.runtime import (
    DEFAULT_SERVER_STARTUP_SECONDS,
    assert_port_available,
    build_shared_runtime_env,
    get_root,
    start_server_process,
    wait_for_server_ready,
)


REPO_ROOT = get_root()

TRAINING_STEPS = 2

README = REPO_ROOT / "examples/DROID/README.md"

MODEL_CHECKPOINT = pathlib.Path(f"/tmp/droid_finetune/checkpoint-{TRAINING_STEPS}")


@pytest.mark.gpu
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "occurrence",
    [1, 2],
    ids=["base", "finetuned"],
)
def test_droid_readme_server_starts(occurrence: int) -> None:
    """Verify the DROID inference server starts and accepts connections."""

    env = build_shared_runtime_env("droid")
    blocks = extract_code_blocks(README)

    model_server_host = "127.0.0.1"
    model_server_port = 5557

    server_code = replace_once(
        find_block(blocks, "run_gr00t_server.py", language="bash", occurrence=occurrence).code,
        "uv run python gr00t/eval/run_gr00t_server.py",
        "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    assert_port_available(model_server_host, model_server_port)
    model_server_proc, server_log = start_server_process(server_code, cwd=REPO_ROOT, env=env)
    try:
        wait_for_server_ready(
            proc=model_server_proc,
            host=model_server_host,
            port=model_server_port,
            timeout_s=float(
                os.getenv("DROID_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
            ),
            server_log=server_log,
        )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_droid_finetune_and_finetuned_server() -> None:
    """Run a short DROID finetune, then verify server starts with the finetuned checkpoint."""

    env = build_shared_runtime_env("droid")
    blocks = extract_code_blocks(README)

    finetune_code = replace_once(
        replace_once(
            replace_once(
                replace_once(
                    find_block(blocks, "--output-dir /tmp/droid_finetune", language="bash").code,
                    "NUM_GPUS=8",
                    "NUM_GPUS=1",
                ),
                "MAX_STEPS=20000",
                f"MAX_STEPS={TRAINING_STEPS}",
            ),
            "SAVE_STEPS=1000",
            f"SAVE_STEPS={TRAINING_STEPS}",
        ),
        "GLOBAL_BATCH_SIZE=640",
        "GLOBAL_BATCH_SIZE=2",
    )
    run_bash_blocks(
        [finetune_code],
        cwd=REPO_ROOT,
        env={
            **env,
            "USE_WANDB": "0",
            "DATALOADER_NUM_WORKERS": "0",
            "SHARD_SIZE": "64",
            "NUM_SHARDS_PER_EPOCH": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    assert MODEL_CHECKPOINT.exists(), (
        f"Expected model checkpoint after finetune: {MODEL_CHECKPOINT}"
    )

    model_server_host = "127.0.0.1"
    model_server_port = 5558

    server_code = replace_once(
        replace_once(
            find_block(blocks, "nvidia/GR00T-N1.7-DROID", language="bash").code,
            "uv run python gr00t/eval/run_gr00t_server.py",
            "uv run --extra=dev python gr00t/eval/run_gr00t_server.py",
        ),
        "nvidia/GR00T-N1.7-DROID",
        str(MODEL_CHECKPOINT),
    )
    server_code += f" --device cuda:0 --host {model_server_host} --port {model_server_port}"

    assert_port_available(model_server_host, model_server_port)
    model_server_proc, server_log = start_server_process(server_code, cwd=REPO_ROOT, env=env)
    try:
        wait_for_server_ready(
            proc=model_server_proc,
            host=model_server_host,
            port=model_server_port,
            timeout_s=float(
                os.getenv("DROID_SERVER_STARTUP_SECONDS", str(DEFAULT_SERVER_STARTUP_SECONDS))
            ),
            server_log=server_log,
        )
    finally:
        if model_server_proc.poll() is None:
            model_server_proc.terminate()
            try:
                model_server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                model_server_proc.kill()
                model_server_proc.wait(timeout=15)
