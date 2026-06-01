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

"""CPU-only regression tests for build_tensorrt_engine.build_full_pipeline.

The full TRT build path is exercised by tests/scripts/deployment/test_trt_pipeline.py
under @pytest.mark.gpu. These tests cover the orchestration layer only: shape
inference, engine compilation, and the tensorrt / onnx imports themselves are
stubbed so the assertions run on any CPU host.

Keep the stubs and the build_tensorrt_engine import inside the
build_full_pipeline fixture. Installing them at module top-level replaces
sys.modules["onnx"] for every pytest-xdist worker that collects this file,
including GPU workers, where the empty stub then crashes torch.onnx.export
inside the unrelated test_trt_full_pipeline.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch

import pytest


_PIPELINE_ONNX_FILES = [
    "vit_bf16.onnx",
    "llm_bf16.onnx",
    "vl_self_attention.onnx",
    "state_encoder.onnx",
    "action_encoder.onnx",
    "dit_bf16.onnx",
    "action_decoder.onnx",
]


@pytest.fixture
def build_full_pipeline(monkeypatch):
    """Yield build_full_pipeline with tensorrt/onnx stubbed in sys.modules.

    Every side effect goes through monkeypatch so it is reverted at teardown
    and never leaks across tests collected by the same pytest-xdist worker.
    """
    if "tensorrt" not in sys.modules:
        trt_stub = types.ModuleType("tensorrt")
        trt_stub.Logger = types.SimpleNamespace(WARNING=0, ERROR=1, INFO=2, VERBOSE=3)
        monkeypatch.setitem(sys.modules, "tensorrt", trt_stub)
    if "onnx" not in sys.modules:
        monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))

    # scripts/deployment/ is not a package; mirror the pattern used by
    # test_trt_pipeline.py so build_tensorrt_engine is importable.
    deploy_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
    )
    monkeypatch.syspath_prepend(deploy_dir)

    from build_tensorrt_engine import build_full_pipeline as fn

    yield fn


def _seed_dummy_onnx_dir(onnx_dir):
    """Touch every ONNX file build_full_pipeline iterates over."""
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for fname in _PIPELINE_ONNX_FILES:
        (onnx_dir / fname).touch()


def _fake_build_engine_success(onnx_path, engine_path, **kwargs):
    with open(engine_path, "wb"):
        pass


def test_build_full_pipeline_raises_when_any_engine_fails(tmp_path, build_full_pipeline):
    """Regression: a single sub-engine failure must not silently exit 0.

    Before the fix, build_full_pipeline caught all build_engine exceptions, logged
    them into a results list, and returned without raising. main() therefore
    exited 0 even though the engine directory was incomplete, and downstream
    verify/benchmark steps were the first to notice.
    """
    onnx_dir = tmp_path / "onnx"
    engine_dir = tmp_path / "engines"
    _seed_dummy_onnx_dir(onnx_dir)

    def fake_build_engine(onnx_path, engine_path, **kwargs):
        if "llm" in os.path.basename(onnx_path):
            raise RuntimeError("simulated TRT failure")
        _fake_build_engine_success(onnx_path, engine_path)

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=fake_build_engine),
        pytest.raises(RuntimeError, match="Pipeline build failed for 1/"),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )


def test_build_full_pipeline_returns_normally_when_all_engines_build(tmp_path, build_full_pipeline):
    """Happy path: every engine builds → no exception."""
    onnx_dir = tmp_path / "onnx"
    engine_dir = tmp_path / "engines"
    _seed_dummy_onnx_dir(onnx_dir)

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=_fake_build_engine_success),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )
