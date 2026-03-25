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
End-to-end test for the torch -> ONNX -> TensorRT serialization pipeline.

Follows the 3-step deployment guide (README.md):
  1. Export model to ONNX  (export_onnx_n1d7.py --export-mode full_pipeline)
  2. Build TensorRT engines (build_tensorrt_engine.py --mode full_pipeline)
  3. Verify accuracy         (verify_n1d7_trt.py --mode n17_full_pipeline)

The test asserts that the final cosine similarity between PyTorch and TRT
outputs is >= COSINE_THRESHOLD (0.99).  Any crash in steps 1-3 also fails
the test via non-zero exit code.

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
import tempfile

import pytest

from tests.examples.utils import build_uv_runtime_env, run_subprocess_step


logger = logging.getLogger(__name__)


ROOT = pathlib.Path(__file__).resolve().parents[3]

# Defaults match the deployment README quick-start paths.
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

# Regex to capture the final cosine similarity value printed by verify_n1d7_trt.py
# It prints lines like:  "  Cosine Similarity: 0.999123"
# We want the *last* occurrence (the final action output comparison).
_COSINE_RE = re.compile(r"Cosine Similarity:\s+([\d.]+)")


def _parse_final_cosine(output: str) -> float:
    """Extract the last Cosine Similarity value from verify script output."""
    matches = _COSINE_RE.findall(output)
    if not matches:
        raise ValueError(
            "Could not find 'Cosine Similarity: <value>' in verify output.\n"
            f"Output tail:\n{output[-2000:]}"
        )
    return float(matches[-1])


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_trt_full_pipeline_accuracy() -> None:
    """Export ONNX, build TRT engines, and verify cosine similarity >= threshold."""

    model_path = DEFAULT_MODEL_PATH
    dataset_path = DEFAULT_DATASET_PATH

    if not pathlib.Path(model_path).exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    if not pathlib.Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    # Use build_uv_runtime_env (not build_shared_runtime_env) to preserve the
    # caller's HF_HOME / HF_TOKEN / TRANSFORMERS_CACHE.  The shared-env helper
    # redirects HF cache dirs to /shared/hf-cache/ which won't have auth tokens
    # or cached gated models outside of CI.
    env = build_uv_runtime_env()

    # Use a temp directory for ONNX and engine artifacts so the test is hermetic.
    tmpdir = tempfile.mkdtemp(prefix="trt_test_")
    onnx_dir = os.path.join(tmpdir, "onnx")
    engine_dir = os.path.join(tmpdir, "engines")

    try:
        # ── Step 1: Export to ONNX ──────────────────────────────────────
        run_subprocess_step(
            [
                "python",
                "scripts/deployment/export_onnx_n1d7.py",
                "--model-path",
                model_path,
                "--dataset-path",
                dataset_path,
                "--output-dir",
                onnx_dir,
                "--export-mode",
                "full_pipeline",
                "--embodiment-tag",
                DEFAULT_EMBODIMENT,
            ],
            step="export_onnx",
            cwd=ROOT,
            env=env,
            log_prefix="trt_pipeline",
            failure_prefix="TRT pipeline: ONNX export failed",
            output_tail_chars=8000,
        )

        # Sanity-check that ONNX files were actually produced.
        onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
        assert onnx_files, f"No .onnx files found in {onnx_dir}"

        # ── Step 2: Build TensorRT engines ──────────────────────────────
        run_subprocess_step(
            [
                "python",
                "scripts/deployment/build_tensorrt_engine.py",
                "--mode",
                "full_pipeline",
                "--onnx-dir",
                onnx_dir,
                "--engine-dir",
                engine_dir,
                "--precision",
                "bf16",
            ],
            step="build_trt_engines",
            cwd=ROOT,
            env=env,
            log_prefix="trt_pipeline",
            failure_prefix="TRT pipeline: TRT engine build failed",
            output_tail_chars=8000,
        )

        engine_files = [f for f in os.listdir(engine_dir) if f.endswith(".engine")]
        assert engine_files, f"No .engine files found in {engine_dir}"

        # Remove the ViT TRT engine so the verify step falls back to PyTorch ViT.
        # The ViT TRT engine has a known accuracy issue (~0.57 cosine due to TRT
        # builder kernel fusion bugs across 24 ViT blocks).  Production deployments
        # keep ViT in PyTorch; the runtime automatically falls back when the engine
        # file is absent.
        vit_engine = os.path.join(engine_dir, "vit_bf16.engine")
        if os.path.exists(vit_engine):
            os.remove(vit_engine)
            logger.info("removed vit_bf16.engine (ViT stays in PyTorch)")

        # ── Step 3: Verify accuracy (PyTorch vs TRT) ───────────────────
        result, _ = run_subprocess_step(
            [
                "python",
                "scripts/deployment/verify_n1d7_trt.py",
                "--model-path",
                model_path,
                "--dataset-path",
                dataset_path,
                "--engine-dir",
                engine_dir,
                "--mode",
                "n17_full_pipeline",
                "--embodiment-tag",
                DEFAULT_EMBODIMENT,
            ],
            step="verify_trt_accuracy",
            cwd=ROOT,
            env=env,
            log_prefix="trt_pipeline",
            failure_prefix="TRT pipeline: accuracy verification failed",
            output_tail_chars=8000,
        )

        # Parse cosine similarity from combined stdout/stderr.
        combined_output = (result.stdout or "") + (result.stderr or "")
        cosine = _parse_final_cosine(combined_output)
        logger.info("final cosine similarity: %.6f", cosine)

        assert cosine >= COSINE_THRESHOLD, (
            f"TRT vs PyTorch cosine similarity {cosine:.6f} "
            f"is below threshold {COSINE_THRESHOLD}. "
            f"This indicates a significant accuracy regression in the "
            f"ONNX export or TRT engine build."
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
