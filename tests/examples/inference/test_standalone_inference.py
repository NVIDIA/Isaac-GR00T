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
Standalone inference smoke tests.

Runs standalone_inference_script.py and open_loop_eval.py in PyTorch mode on
bundled demo datasets for a few steps.  Verifies that the full inference
pipeline (model loading, data processing, action prediction, plot saving)
completes without errors.

The tests are parameterized over multiple embodiments (LIBERO, DROID) so each
configuration is exercised independently.

Environment variables (optional, per-embodiment):
  INFERENCE_TEST_LIBERO_MODEL_PATH   – LIBERO checkpoint path override
  INFERENCE_TEST_LIBERO_DATASET_PATH – LIBERO dataset path override
  INFERENCE_TEST_DROID_MODEL_PATH    – DROID checkpoint path override
  INFERENCE_TEST_DROID_DATASET_PATH  – DROID dataset path override
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

import pytest
from test_support.runtime import (
    build_uv_runtime_env,
    get_root,
    resolve_demo_dataset,
    resolve_model_checkpoint_path,
    run_subprocess_step,
)


ROOT = get_root()


@dataclass(frozen=True)
class InferenceVariant:
    """Configuration for one embodiment variant of the inference smoke tests."""

    id: str
    embodiment_tag: str
    hf_repo_id: str
    hf_subdir: str | None
    dataset_name: str
    model_env_var: str = ""
    dataset_env_var: str = ""

    def __str__(self) -> str:
        return self.id


LIBERO = InferenceVariant(
    id="libero",
    embodiment_tag="LIBERO_PANDA",
    hf_repo_id="nvidia/GR00T-N1.7-LIBERO",
    hf_subdir="libero_10",
    dataset_name="libero_demo",
    model_env_var="INFERENCE_TEST_LIBERO_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_LIBERO_DATASET_PATH",
)

DROID = InferenceVariant(
    id="droid",
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    hf_repo_id="nvidia/GR00T-N1.7-DROID",
    hf_subdir=None,
    dataset_name="droid_sample",
    model_env_var="INFERENCE_TEST_DROID_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_DROID_DATASET_PATH",
)

VARIANTS = [LIBERO, DROID]


def _model_path(variant: InferenceVariant) -> str:
    return str(
        resolve_model_checkpoint_path(
            hf_repo_id=variant.hf_repo_id,
            hf_subdir=variant.hf_subdir,
            path_override_env=variant.model_env_var,
            repo_root=ROOT,
        )
    )


def _dataset_path(variant: InferenceVariant) -> str:
    return str(
        resolve_demo_dataset(
            dataset_name=variant.dataset_name,
            path_override_env=variant.dataset_env_var,
            repo_root=ROOT,
        )
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_standalone_inference_pytorch(variant: InferenceVariant) -> None:
    """Run standalone inference in PyTorch mode for a few steps on 1 trajectory."""

    model_path = _model_path(variant)
    dataset_path = _dataset_path(variant)

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
            variant.embodiment_tag,
            "--traj-ids",
            "0",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "20",
        ],
        step=f"standalone_inference_pytorch_{variant.id}",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix=f"Standalone inference (PyTorch, {variant.id}) failed",
        output_tail_chars=8000,
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_standalone_inference_invalid_traj_id(variant: InferenceVariant) -> None:
    """Passing an out-of-range --traj-ids should raise ValueError, not UnboundLocalError."""

    model_path = _model_path(variant)
    dataset_path = _dataset_path(variant)

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
            variant.embodiment_tag,
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
@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_open_loop_eval_with_checkpoint(variant: InferenceVariant) -> None:
    """Run open_loop_eval.py directly with a model checkpoint."""

    model_path = _model_path(variant)
    dataset_path = _dataset_path(variant)

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "gr00t/eval/open_loop_eval.py",
            "--dataset-path",
            dataset_path,
            "--embodiment-tag",
            variant.embodiment_tag,
            "--model-path",
            model_path,
            "--traj-ids",
            "0",
            "--action-horizon",
            "8",
            "--steps",
            "5",
        ],
        step=f"open_loop_eval_{variant.id}",
        cwd=ROOT,
        env=env,
        log_prefix="inference",
        failure_prefix=f"Open-loop eval ({variant.id}) failed",
        output_tail_chars=8000,
    )
