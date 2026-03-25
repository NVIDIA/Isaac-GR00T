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
DROID example smoke test.

Tests zero-shot inference with the base model on the bundled droid_sample
demo data, then runs open-loop eval on the same data.  This catches
model-loading, data-pipeline, and embodiment-tag regressions for the
OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT pretrain tag.

Unlike LIBERO, DROID is a pretrain embodiment — no finetuning or checkpoint
download is needed.  The base model (nvidia/GR00T-N1.7-3B) is used directly.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.examples.utils import build_uv_runtime_env, run_subprocess_step


ROOT = pathlib.Path(__file__).resolve().parents[3]

DATASET_PATH = ROOT / "demo_data/droid_sample"
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_droid_standalone_inference() -> None:
    """Run standalone inference on DROID demo data with the base model."""

    if not DATASET_PATH.exists():
        pytest.skip(f"DROID demo data not found at {DATASET_PATH}")

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "scripts/deployment/standalone_inference_script.py",
            "--model-path",
            BASE_MODEL,
            "--dataset-path",
            str(DATASET_PATH),
            "--embodiment-tag",
            EMBODIMENT_TAG,
            "--traj-ids",
            "1",
            "2",
            "--inference-mode",
            "pytorch",
            "--action-horizon",
            "8",
            "--steps",
            "20",
        ],
        step="standalone_inference_droid",
        cwd=ROOT,
        env=env,
        log_prefix="droid",
        failure_prefix="DROID standalone inference failed",
        output_tail_chars=8000,
    )


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_droid_open_loop_eval() -> None:
    """Run open-loop eval on DROID demo data with the base model."""

    if not DATASET_PATH.exists():
        pytest.skip(f"DROID demo data not found at {DATASET_PATH}")

    env = build_uv_runtime_env()

    run_subprocess_step(
        [
            "python",
            "gr00t/eval/open_loop_eval.py",
            "--dataset-path",
            str(DATASET_PATH),
            "--embodiment-tag",
            EMBODIMENT_TAG,
            "--model-path",
            BASE_MODEL,
            "--traj-ids",
            "1",
            "--action-horizon",
            "8",
            "--steps",
            "20",
        ],
        step="open_loop_eval_droid",
        cwd=ROOT,
        env=env,
        log_prefix="droid",
        failure_prefix="DROID open-loop eval failed",
        output_tail_chars=8000,
    )
