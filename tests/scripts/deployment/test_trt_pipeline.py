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

Runs the full export → build → verify flow in-process and asserts that the final
cosine similarity between PyTorch and TRT outputs is >= COSINE_THRESHOLD (0.99).

To keep CI fast the test loads the real checkpoint but immediately truncates the DiT
action head to _TRUNCATED_DIT_BLOCKS transformer blocks. Export, TRT build, and
verify all see the same truncated model, so the cosine comparison remains meaningful.

Environment variables (all optional):
  TRT_TEST_MODEL_PATH   – path to a finetuned checkpoint
                          (default: shared cache + HF download of libero_10)
  TRT_TEST_DATASET_PATH – path to a LeRobot dataset
                          (default: :func:`resolve_libero_demo_dataset_path`)
  TRT_TEST_EMBODIMENT   – embodiment tag
                          (default: LIBERO_PANDA)
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
from unittest.mock import patch


# scripts/deployment/ is not a package; add it to sys.path so its modules are importable.
# build_trt_pipeline.py does the same for its own sibling imports when it runs.
_DEPLOY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
)
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from build_trt_pipeline import (  # noqa: E402
    PipelineConfig,
    _resolve_embodiment,
    _run_build,
    _run_export,
    _run_verify,
)
import pytest  # noqa: E402
import tensorrt as trt  # noqa: E402
from test_support.runtime import (  # noqa: E402
    get_root,
    resolve_libero_demo_dataset_path,
    resolve_libero_n17_libero10_checkpoint_path,
)


logger = logging.getLogger(__name__)


ROOT = get_root()
DEFAULT_EMBODIMENT = os.getenv("TRT_TEST_EMBODIMENT", "LIBERO_PANDA")

COSINE_THRESHOLD = 0.99

# Keep only this many DiT transformer blocks so that ONNX export and TRT build
# finish quickly. Both PyTorch and TRT see the identical truncated graph, so the
# cosine comparison remains a valid accuracy check.
_TRUNCATED_DIT_BLOCKS = 2


@contextlib.contextmanager
def _truncated_policy():
    """Patch Gr00tPolicy to drop DiT blocks down to _TRUNCATED_DIT_BLOCKS after init."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    _real_init = Gr00tPolicy.__init__

    def _fast_init(self, *args, **kwargs):
        _real_init(self, *args, **kwargs)
        blocks = self.model.action_head.model.transformer_blocks
        if len(blocks) > _TRUNCATED_DIT_BLOCKS:
            self.model.action_head.model.transformer_blocks = blocks[:_TRUNCATED_DIT_BLOCKS]

    with patch.object(Gr00tPolicy, "__init__", _fast_init):
        yield


@pytest.mark.gpu
@pytest.mark.timeout(600)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_trt_full_pipeline(batch_size: int, tmp_path) -> None:
    """Export ONNX, build TRT engines, and verify cosine similarity >= threshold."""

    model_path = str(
        resolve_libero_n17_libero10_checkpoint_path(ROOT, path_override_env="TRT_TEST_MODEL_PATH")
    )
    dataset_path = str(
        resolve_libero_demo_dataset_path(ROOT, path_override_env="TRT_TEST_DATASET_PATH")
    )

    cfg = PipelineConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        embodiment_tag=DEFAULT_EMBODIMENT,
        output_dir=str(tmp_path),
        export_mode="full_pipeline",
        batch_size=batch_size,
        steps="export,build,verify",
    )

    onnx_dir = str(tmp_path / "onnx")
    engine_dir = str(tmp_path / "engines")
    embodiment_tag = _resolve_embodiment(cfg.model_path, cfg.embodiment_tag)

    with open(tmp_path / "pipeline.log", "w") as log_fp, _truncated_policy():
        _run_export(cfg, onnx_dir, embodiment_tag, log_fp)
        _run_build(cfg, onnx_dir, engine_dir, log_fp, trt_severity=trt.Logger.WARNING)
        cosine = _run_verify(cfg, engine_dir, embodiment_tag, log_fp)

    logger.info("final cosine similarity (bs=%d): %.6f", batch_size, cosine)
    assert cosine >= COSINE_THRESHOLD, (
        f"TRT vs PyTorch cosine similarity {cosine:.6f} (batch_size={batch_size}) "
        f"is below threshold {COSINE_THRESHOLD}. "
        "This indicates a significant accuracy regression in the ONNX export or TRT engine build."
    )


# ---------------------------------------------------------------------------
# __main__ entrypoint tests
#
# These tests exercise the script's CLI surface (tyro argument registration,
# argument parsing, and error handling) by invoking it via subprocess — the
# same way a user or deployment script would call it.  They are CPU-only and
# complete in under a second.
# ---------------------------------------------------------------------------

_PIPELINE_SCRIPT = os.path.join(_DEPLOY_DIR, "build_trt_pipeline.py")


def test_build_trt_pipeline_help() -> None:
    """--help exits 0 and surfaces the expected CLI options."""
    result = subprocess.run(
        [sys.executable, _PIPELINE_SCRIPT, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}:\n{result.stderr}"
    for flag in ["--model-path", "--dataset-path", "--steps", "--export-mode", "--batch-size"]:
        assert flag in result.stdout, f"Expected '{flag}' in --help output:\n{result.stdout}"


def test_build_trt_pipeline_missing_model_path() -> None:
    """Invoking the script without --model-path exits non-zero with a clear error."""
    result = subprocess.run(
        [sys.executable, _PIPELINE_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Expected non-zero exit when --model-path is omitted"
    combined = result.stdout + result.stderr
    # main() raises ValueError("Please provide --model-path") when model_path is empty;
    # the traceback appears on stderr.
    assert "Please provide --model-path" in combined, (
        f"Expected error 'Please provide --model-path' in output:\n{combined}"
    )
