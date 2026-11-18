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

"""Hermetic argument-plumbing tests for examples/finetune.sh.

The script is executed with a stub ``python`` on PATH that captures the argv it
would have launched, so these tests verify the exact command construction
(single-GPU path) without importing torch or starting a training run.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
from test_support.runtime import get_root


REPO_ROOT = get_root()
FINETUNE_SH = REPO_ROOT / "examples" / "finetune.sh"

_STUB_PYTHON = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$CAPTURE_FILE"
"""


def _run_finetune_sh(tmp_path: Path, env_overrides: dict[str, str]) -> list[str]:
    """Run finetune.sh with a stubbed ``python`` and return the captured argv."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    capture_file = tmp_path / "captured_argv.txt"

    env = os.environ.copy()
    env.update(env_overrides)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["CAPTURE_FILE"] = str(capture_file)
    env.setdefault("NUM_GPUS", "1")  # single-GPU path uses `exec python`
    env.setdefault("USE_WANDB", "0")

    result = subprocess.run(
        [
            "bash",
            str(FINETUNE_SH),
            "--base-model-path",
            "/fake/model",
            "--dataset-path",
            "/fake/dataset",
            "--embodiment-tag",
            "new_embodiment",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"finetune.sh failed:\n{result.stderr}"
    assert capture_file.exists(), "stub python was never invoked"
    return capture_file.read_text().splitlines()


def _flag_value(argv: list[str], flag: str) -> str:
    """Return the value following ``flag`` in the captured argv."""
    assert flag in argv, f"{flag} not found in argv: {argv}"
    return argv[argv.index(flag) + 1]


class TestGradientAccumulationPlumbing:
    def test_default_is_one(self, tmp_path):
        """Without the env var the launcher receives accumulation steps of 1
        (identical behavior to before the flag was exposed)."""
        argv = _run_finetune_sh(tmp_path, {})
        assert _flag_value(argv, "--gradient_accumulation_steps") == "1"

    def test_env_var_is_forwarded(self, tmp_path):
        argv = _run_finetune_sh(tmp_path, {"GRADIENT_ACCUMULATION_STEPS": "4"})
        assert _flag_value(argv, "--gradient_accumulation_steps") == "4"

    def test_low_vram_recipe_preserves_effective_batch(self, tmp_path):
        """The documented low-VRAM recipe: micro-batch 8 x 4 accumulation steps
        must reach the launcher exactly as given (effective batch 32, the
        single-GPU default)."""
        argv = _run_finetune_sh(
            tmp_path,
            {"GLOBAL_BATCH_SIZE": "8", "GRADIENT_ACCUMULATION_STEPS": "4"},
        )
        global_batch = int(_flag_value(argv, "--global_batch_size"))
        accum_steps = int(_flag_value(argv, "--gradient_accumulation_steps"))
        assert global_batch == 8
        assert accum_steps == 4
        assert global_batch * accum_steps == 32

    def test_existing_defaults_unchanged(self, tmp_path):
        """Guard: exposing the new env var must not disturb neighboring args."""
        argv = _run_finetune_sh(tmp_path, {})
        assert _flag_value(argv, "--global_batch_size") == "32"
        assert _flag_value(argv, "--dataloader_num_workers") == "4"
        assert _flag_value(argv, "--embodiment_tag") == "new_embodiment"


@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_launcher_rejects_invalid_accumulation(bad_value):
    """FinetuneConfig validates gradient_accumulation_steps >= 1, so a bad env
    var fails fast at config time rather than corrupting the batch math."""
    from gr00t.configs.finetune_config import FinetuneConfig

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        FinetuneConfig(
            base_model_path="/fake/model",
            dataset_path="/fake/dataset",
            embodiment_tag="new_embodiment",
            output_dir="/fake/out",
            gradient_accumulation_steps=int(bad_value),
        )
