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

"""
Standalone inference smoke test.

Runs standalone_inference_script.py in PyTorch mode on the bundled
libero_demo dataset for a few steps.  Verifies that the full inference
pipeline (model loading, data processing, action prediction, plot saving)
completes without errors.

Environment variables (optional):
  INFERENCE_TEST_MODEL_PATH   – checkpoint path (default: shared cache + HF download)
  INFERENCE_TEST_DATASET_PATH – dataset path    (default: :func:`resolve_libero_demo_dataset_path`)
"""

from __future__ import annotations

import subprocess

import pytest
from test_support.runtime import (
    build_uv_runtime_env,
    get_root,
    resolve_libero_demo_dataset_path,
    resolve_libero_n17_libero10_checkpoint_path,
    run_subprocess_step,
)


ROOT = get_root()


def _model_path() -> str:
    return str(
        resolve_libero_n17_libero10_checkpoint_path(
            ROOT, path_override_env="INFERENCE_TEST_MODEL_PATH"
        )
    )


def _dataset_path() -> str:
    return str(
        resolve_libero_demo_dataset_path(ROOT, path_override_env="INFERENCE_TEST_DATASET_PATH")
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_standalone_inference_pytorch() -> None:
    """Run standalone inference in PyTorch mode for 5 steps on 1 trajectory."""

    model_path = _model_path()
    dataset_path = _dataset_path()

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "scripts/deployment/standalone_inference_script.py",
            "--model-path",
            model_path,
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--traj-ids",
            "0",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "20",
        ],
        step="standalone_inference_pytorch",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix="Standalone inference (PyTorch) failed",
        output_tail_chars=8000,
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_standalone_inference_invalid_traj_id() -> None:
    """Passing an out-of-range --traj-ids should raise ValueError, not UnboundLocalError."""

    model_path = _model_path()
    dataset_path = _dataset_path()

    env = build_uv_runtime_env()

    result = subprocess.run(
        [
            "python",
            "scripts/deployment/standalone_inference_script.py",
            "--model-path",
            model_path,
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--traj-ids",
            "999",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "5",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Expected non-zero exit for out-of-range traj_id"
    combined = result.stdout + result.stderr
    assert "out of range" in combined.lower(), (
        f"Expected 'out of range' in output, got:\n{combined[-2000:]}"
    )
    assert "UnboundLocalError" not in combined, "Should raise ValueError, not UnboundLocalError"


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_open_loop_eval_with_checkpoint() -> None:
    """Run open_loop_eval.py directly with a model checkpoint."""

    model_path = _model_path()
    dataset_path = _dataset_path()

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "gr00t/eval/open_loop_eval.py",
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            "LIBERO_PANDA",
            "--model-path",
            model_path,
            "--traj-ids",
            "0",
            "--action-horizon",
            "8",
            "--steps",
            "5",
        ],
        step="open_loop_eval",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix="Open-loop eval failed",
        output_tail_chars=8000,
    )
