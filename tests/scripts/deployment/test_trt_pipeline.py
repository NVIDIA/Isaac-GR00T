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
End-to-end test for the unified TRT deployment pipeline (build_trt_pipeline.py).

Runs the full export → build → verify flow and asserts that the final cosine
similarity between PyTorch and TRT outputs is >= COSINE_THRESHOLD (0.99).

Environment variables (all optional):
  TRT_TEST_MODEL_PATH   – path to a finetuned checkpoint
                          (default: shared cache + HF download of libero_10)
  TRT_TEST_DATASET_PATH – path to a LeRobot dataset
                          (default: :func:`resolve_libero_demo_dataset_path`)
  TRT_TEST_EMBODIMENT   – embodiment tag
                          (default: LIBERO_PANDA)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile

import pytest
from test_support.runtime import (
    build_uv_runtime_env,
    get_root,
    resolve_libero_demo_dataset_path,
    resolve_libero_n17_libero10_checkpoint_path,
    run_subprocess_step,
)


logger = logging.getLogger(__name__)


ROOT = get_root()
DEFAULT_EMBODIMENT = os.getenv("TRT_TEST_EMBODIMENT", "LIBERO_PANDA")

COSINE_THRESHOLD = 0.99

# Regex to capture cosine similarity from build_trt_pipeline.py output.
# It prints lines like:  "[Step 3/3] Verify complete -- cosine=0.999123 PASS (1m 23s)"
_COSINE_RE = re.compile(r"cosine[=:]\s*([\d.]+)")


def _parse_final_cosine(output: str) -> float:
    """Extract the last cosine similarity value from pipeline output."""
    matches = _COSINE_RE.findall(output)
    if not matches:
        raise ValueError(
            "Could not find cosine similarity value in pipeline output.\n"
            f"Output tail:\n{output[-2000:]}"
        )
    return float(matches[-1])


@pytest.mark.gpu
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_trt_full_pipeline(batch_size: int) -> None:
    """Export ONNX, build TRT engines, and verify cosine similarity >= threshold."""

    model_path = str(
        resolve_libero_n17_libero10_checkpoint_path(ROOT, path_override_env="TRT_TEST_MODEL_PATH")
    )
    dataset_path = str(
        resolve_libero_demo_dataset_path(ROOT, path_override_env="TRT_TEST_DATASET_PATH")
    )

    env = build_uv_runtime_env()
    tmpdir = tempfile.mkdtemp(prefix=f"trt_test_bs{batch_size}_")

    try:
        result, _ = run_subprocess_step(
            [
                "python",
                "scripts/deployment/build_trt_pipeline.py",
                "--model-path",
                model_path,
                "--dataset-path",
                dataset_path,
                "--embodiment-tag",
                DEFAULT_EMBODIMENT,
                "--output-dir",
                tmpdir,
                "--export-mode",
                "full_pipeline",
                "--batch-size",
                str(batch_size),
                "--steps",
                "export,build,verify",
            ],
            step=f"trt_pipeline_bs{batch_size}",
            cwd=ROOT,
            env=env,
            log_prefix=f"trt_pipeline_bs{batch_size}",
            failure_prefix=f"TRT pipeline (bs={batch_size}) failed",
            output_tail_chars=8000,
        )

        combined_output = (result.stdout or "") + (result.stderr or "")
        cosine = _parse_final_cosine(combined_output)
        logger.info("final cosine similarity (bs=%d): %.6f", batch_size, cosine)

        assert cosine >= COSINE_THRESHOLD, (
            f"TRT vs PyTorch cosine similarity {cosine:.6f} (batch_size={batch_size}) "
            f"is below threshold {COSINE_THRESHOLD}. "
            f"This indicates a significant accuracy regression in the "
            f"ONNX export or TRT engine build."
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
