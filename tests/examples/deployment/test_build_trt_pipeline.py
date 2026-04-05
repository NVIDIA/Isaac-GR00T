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
End-to-end test for the unified build_trt_pipeline.py script.

Runs the full pipeline (export, build, verify) via a single subprocess call
and asserts that the printed cosine similarity is >= COSINE_THRESHOLD.

Environment variables (all optional):
  TRT_TEST_MODEL_PATH   – path to a finetuned checkpoint
                          (default: checkpoints/GR00T-N1.7-LIBERO/libero_10)
  TRT_TEST_DATASET_PATH – path to a LeRobot dataset
                          (default: demo_data/libero_demo)
  TRT_TEST_EMBODIMENT   – embodiment tag
                          (default: LIBERO_PANDA)
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile

import pytest
from test_support.runtime import build_uv_runtime_env, run_subprocess_step


logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = os.getenv(
    "TRT_TEST_MODEL_PATH",
    str(ROOT / "checkpoints/GR00T-N1.7-LIBERO/libero_10"),
)
DEFAULT_DATASET_PATH = os.getenv(
    "TRT_TEST_DATASET_PATH",
    str(ROOT / "demo_data/libero_demo"),
)
DEFAULT_EMBODIMENT = os.getenv("TRT_TEST_EMBODIMENT", "LIBERO_PANDA")

COSINE_THRESHOLD = 0.99

# The pipeline prints: "[Step 3/3] Verify complete -- cosine=0.999123 PASS (42s)"
_COSINE_RE = re.compile(r"cosine=([\d.]+)")


def _parse_cosine(output: str) -> float:
    """Extract the cosine similarity value from pipeline output."""
    matches = _COSINE_RE.findall(output)
    if not matches:
        raise ValueError(
            f"Could not find 'cosine=<value>' in pipeline output.\nOutput tail:\n{output[-2000:]}"
        )
    return float(matches[-1])


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_build_trt_pipeline() -> None:
    """Run build_trt_pipeline.py (export, build, verify) and check cosine >= threshold."""

    model_path = DEFAULT_MODEL_PATH
    dataset_path = DEFAULT_DATASET_PATH

    if not pathlib.Path(model_path).exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    if not pathlib.Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    env = build_uv_runtime_env()
    tmpdir = tempfile.mkdtemp(prefix="trt_pipeline_test_")

    try:
        result, _ = run_subprocess_step(
            [
                sys.executable,
                "scripts/deployment/build_trt_pipeline.py",
                "--model-path",
                model_path,
                "--dataset-path",
                dataset_path,
                "--output-dir",
                tmpdir,
                "--embodiment-tag",
                DEFAULT_EMBODIMENT,
                "--steps",
                "export,build,verify",
            ],
            step="build_trt_pipeline",
            cwd=ROOT,
            env=env,
            log_prefix="trt_pipeline_unified",
            failure_prefix="build_trt_pipeline.py failed",
            output_tail_chars=8000,
        )

        combined_output = (result.stdout or "") + (result.stderr or "")
        cosine = _parse_cosine(combined_output)
        logger.info("build_trt_pipeline cosine similarity: %.6f", cosine)

        assert cosine >= COSINE_THRESHOLD, (
            f"TRT vs PyTorch cosine similarity {cosine:.6f} is below threshold {COSINE_THRESHOLD}."
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
