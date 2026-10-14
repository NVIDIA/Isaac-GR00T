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

Covers every export mode and runtime mode, including retained PyTorch components.
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
import gc
import io
import logging
import os
from pathlib import Path
import re
import runpy
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

import pytest  # noqa: E402
from test_support.runtime import (  # noqa: E402
    get_root,
    resolve_libero_demo_dataset_path,
    resolve_libero_n17_libero10_checkpoint_path,
)
import torch  # noqa: E402


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


@pytest.fixture
def _release_cuda_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.edge_device
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "export_mode,execution_profile,quantization,batch_size",
    [
        pytest.param("dit_only", "baseline", "none", 1, id="dit-only"),
        pytest.param("action_head", "baseline", "none", 1, id="action-head-baseline"),
        pytest.param("action_head", "optimized", "none", 1, id="action-head-optimized"),
        pytest.param("action_head", "optimized", "mixed_nvfp4", 1, id="action-head-nvfp4"),
        pytest.param("full_pipeline", "baseline", "none", 1, id="full-baseline-bs1"),
        pytest.param("full_pipeline", "baseline", "none", 2, id="full-baseline-bs2"),
        pytest.param("full_pipeline", "optimized", "none", 1, id="full-optimized-bs1"),
        pytest.param("full_pipeline", "optimized", "none", 2, id="full-optimized-bs2"),
        pytest.param("full_pipeline", "optimized", "mixed_nvfp4", 1, id="full-nvfp4"),
    ],
)
def test_trt_pipeline(
    export_mode: str,
    execution_profile: str,
    quantization: str,
    batch_size: int,
    tmp_path,
    load_hf_model_weights,
    _release_cuda_memory,
    monkeypatch,
    record_property,
) -> None:
    """Export ONNX, build TRT engines, and verify cosine similarity >= threshold."""

    model_path = str(
        resolve_libero_n17_libero10_checkpoint_path(ROOT, path_override_env="TRT_TEST_MODEL_PATH")
    )
    dataset_path = str(
        resolve_libero_demo_dataset_path(ROOT, path_override_env="TRT_TEST_DATASET_PATH")
    )

    if quantization != "none":
        if "thor" not in torch.cuda.get_device_name(0).lower() or batch_size != 1:
            pytest.skip("NVFP4 smoke is scoped to Thor batch 1")
        # On supported hardware a missing dependency must fail, not silently skip.
        __import__("modelopt.torch.quantization")
    command = [
        "scripts/deployment/build_trt_pipeline.py",
        "--model-path",
        model_path,
        "--dataset-path",
        dataset_path,
        "--embodiment-tag",
        DEFAULT_EMBODIMENT,
        "--output-dir",
        str(tmp_path),
        "--export-mode",
        export_mode,
        "--execution-profile",
        execution_profile,
        "--quantization",
        quantization,
        "--batch-size",
        str(batch_size),
        "--calib-size",
        "2",
        "--steps",
        "export,build,verify",
    ]
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record_property("tested_source_sha", source_sha)
    record_property("smoke_dit_blocks", _TRUNCATED_DIT_BLOCKS)
    record_property("public_command", "python " + " ".join(command))
    logger.info("Current source %s; public command: python %s", source_sha, " ".join(command))
    with load_hf_model_weights(), _truncated_policy():
        monkeypatch.setattr(sys, "argv", command)
        runpy.run_path(str(ROOT / command[0]), run_name="__main__")
    matches = re.findall(r"Cosine Similarity:\s*([0-9.]+)", (tmp_path / "pipeline.log").read_text())
    assert matches, "Public verification command must report final-action cosine"
    _assert_action_cosine(float(matches[-1]), "pipeline", record_property)

    with load_hf_model_weights():
        if export_mode == "full_pipeline" and batch_size == 1:
            # Reuse this build to exercise TRT backbone -> retained PyTorch action head,
            # including the FP16 quantized LLM -> BF16 head boundary on Thor.
            _verify_runtime_mode(
                model_path,
                dataset_path,
                tmp_path / "engines",
                "vit_llm_only",
                monkeypatch,
                record_property,
            )
            if execution_profile == "baseline":
                # The full loader allows either backbone engine to be absent. Exercise
                # both fallbacks with real engines; no additional export/build needed.
                for missing in ("llm_bf16.engine", "vit.engine"):
                    subset = tmp_path / f"without-{missing}"
                    subset.mkdir()
                    for artifact in (tmp_path / "engines").iterdir():
                        if artifact.is_file() and artifact.name != missing:
                            (subset / artifact.name).symlink_to(artifact.resolve())
                    _verify_runtime_mode(
                        model_path,
                        dataset_path,
                        subset,
                        "n17_full_pipeline",
                        monkeypatch,
                        record_property,
                        label=f"without-{missing}",
                    )


def _assert_action_cosine(cosine, label, record_property):
    record_property(f"action_cosine_{label}", cosine)
    assert cosine >= COSINE_THRESHOLD, (
        f"TRT vs PyTorch final-action cosine {cosine:.6f} ({label}) "
        f"is below threshold {COSINE_THRESHOLD}"
    )


def _verify_runtime_mode(
    model_path,
    dataset_path,
    engine_dir,
    mode,
    monkeypatch,
    record_property,
    label=None,
):
    """Run the public verifier with the same two-block policy and real engines."""
    label = label or mode
    command = [
        "scripts.deployment.trt.verify_n1d7_trt",
        "--model-path",
        model_path,
        "--dataset-path",
        dataset_path,
        "--engine-dir",
        str(engine_dir),
        "--embodiment-tag",
        DEFAULT_EMBODIMENT,
        "--mode",
        mode,
        "--batch-size",
        "1",
    ]
    record_property(f"public_command_{label}", "python -m " + " ".join(command))
    output = io.StringIO()
    # Use module execution to preserve the verifier's relative imports.
    try:
        with contextlib.redirect_stdout(output), _truncated_policy():
            monkeypatch.setattr(sys, "argv", command)
            runpy.run_module("scripts.deployment.trt.verify_n1d7_trt", run_name="__main__")
    finally:
        # Preserve diagnostics even when a dtype/shape boundary raises.
        Path(engine_dir).parent.joinpath(f"verify-{label}.log").write_text(output.getvalue())
    log = output.getvalue()
    matches = re.findall(r"Cosine Similarity:\s*([0-9.]+)", log)
    assert matches, f"{label}: verifier must report final-action cosine"
    _assert_action_cosine(float(matches[-1]), label, record_property)


# ---------------------------------------------------------------------------
# __main__ entrypoint tests
#
# These tests exercise the script's CLI surface (tyro argument registration,
# argument parsing, and error handling) by invoking it via subprocess — the
# same way a user or deployment script would call it.  They are CPU-only and
# complete in under a second.
# ---------------------------------------------------------------------------

_PIPELINE_SCRIPT = os.path.join(_DEPLOY_DIR, "build_trt_pipeline.py")


@pytest.mark.serial
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


@pytest.mark.serial
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
