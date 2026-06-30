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
    # The full_pipeline exporter writes vit_fp32.onnx (ViT is FP32 for accuracy);
    # build_full_pipeline prefers it over vit_bf16.onnx when both exist. Mirror
    # that here so the per-component precision override is exercised.
    "vit_fp32.onnx",
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
        pytest.raises(RuntimeError, match=r"1/\d+ engine\(s\) failed"),
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


def test_build_full_pipeline_raises_when_all_onnx_inputs_missing(tmp_path, build_full_pipeline):
    """Empty ONNX dir must raise instead of producing zero engines and exiting 0."""
    onnx_dir = tmp_path / "onnx_missing"
    onnx_dir.mkdir()
    engine_dir = tmp_path / "engines"

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=_fake_build_engine_success),
        pytest.raises(
            RuntimeError,
            match=rf"{len(_PIPELINE_ONNX_FILES)}/{len(_PIPELINE_ONNX_FILES)} component\(s\) had no ONNX input",
        ),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )

    assert not engine_dir.exists() or not list(engine_dir.iterdir())


def test_build_full_pipeline_raises_when_some_onnx_inputs_missing(tmp_path, build_full_pipeline):
    """Partially-populated ONNX dir must raise; "full pipeline" means full."""
    onnx_dir = tmp_path / "onnx_partial"
    onnx_dir.mkdir()
    seeded = _PIPELINE_ONNX_FILES[:5]
    missing = _PIPELINE_ONNX_FILES[5:]
    for fname in seeded:
        (onnx_dir / fname).touch()
    engine_dir = tmp_path / "engines"

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=_fake_build_engine_success),
        pytest.raises(
            RuntimeError,
            match=rf"{len(missing)}/{len(_PIPELINE_ONNX_FILES)} component\(s\) had no ONNX input",
        ),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )


def test_build_full_pipeline_error_mentions_both_skips_and_failures(tmp_path, build_full_pipeline):
    """When components both skip AND fail, the exception message lists both reasons."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    skipped_file = "llm_bf16.onnx"
    failing_file = "dit_bf16.onnx"
    for fname in _PIPELINE_ONNX_FILES:
        if fname == skipped_file:
            continue
        (onnx_dir / fname).touch()
    engine_dir = tmp_path / "engines"

    def fake_build(onnx_path, engine_path, **kwargs):
        if os.path.basename(onnx_path) == failing_file:
            raise RuntimeError("simulated TRT failure")
        _fake_build_engine_success(onnx_path, engine_path)

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=fake_build),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            build_full_pipeline(
                onnx_dir=str(onnx_dir),
                engine_dir=str(engine_dir),
                precision="bf16",
            )

    message = str(exc_info.value)
    assert "engine(s) failed" in message, message
    assert "had no ONNX input" in message, message


# ---------------------------------------------------------------------------
# STRONGLY_TYPED precision sanity check
# ---------------------------------------------------------------------------
#
# Under STRONGLY_TYPED (TRT 10+), precision is read from the ONNX tensor
# types and --precision builder flags are ignored. These tests verify that
# _check_strongly_typed_precision_match fails fast when --precision cannot be
# honored by the network instead of silently building a mismatched engine.


@pytest.fixture
def check_precision_match(build_full_pipeline):  # noqa: ARG001 — share stub setup
    """Return the _check_strongly_typed_precision_match helper from the module."""
    from build_tensorrt_engine import _check_strongly_typed_precision_match

    return _check_strongly_typed_precision_match


def test_strongly_typed_precision_match_passes_for_pure_match(check_precision_match):
    """bf16 request against a pure-BF16 network is fine."""
    check_precision_match({"BF16"}, "bf16")


def test_strongly_typed_precision_match_passes_for_mixed_network(check_precision_match):
    """The real exporter produces a mixed graph (ViT FP32, rest BF16); bf16 still matches."""
    check_precision_match({"BF16", "FLOAT"}, "bf16")


@pytest.mark.parametrize(
    ("network_dtypes", "requested"),
    [
        ({"BF16"}, "fp16"),
        ({"BF16", "FLOAT"}, "fp16"),
        ({"FLOAT"}, "bf16"),
        ({"BF16"}, "fp8"),
        # fp32 against the real-pipeline mixed graph (ViT FP32, rest BF16):
        # FLOAT is present so the basic "expected dtype must be in network"
        # check passes, but STRONGLY_TYPED won't promote BF16 to FLOAT, so
        # the engine would silently mix precisions. The fp32 branch in
        # _check_strongly_typed_precision_match catches this explicitly.
        ({"BF16", "FLOAT"}, "fp32"),
        ({"HALF", "FLOAT"}, "fp32"),
        ({"FP8", "FLOAT"}, "fp32"),
    ],
)
def test_strongly_typed_precision_match_raises_on_mismatch(
    check_precision_match, network_dtypes, requested
):
    """A --precision the STRONGLY_TYPED network cannot honor must raise, not pass silently."""
    with pytest.raises(ValueError, match="cannot be honored"):
        check_precision_match(network_dtypes, requested)


def test_strongly_typed_precision_match_rejects_unknown_token(check_precision_match):
    """A typo like --precision=int8 should fail loudly, not be treated as 'no match'."""
    with pytest.raises(ValueError, match="Unknown precision"):
        check_precision_match({"BF16"}, "int8")


# ---------------------------------------------------------------------------
# Per-component precision override
# ---------------------------------------------------------------------------
#
# build_full_pipeline must mirror the export's mixed-precision layout
# (ViT FP32, every other component BF16) instead of forwarding the
# pipeline-wide --precision to all engines. Without this, building the
# ViT engine from vit_fp32.onnx with --precision=bf16 trips the
# STRONGLY_TYPED sanity check above and the whole pipeline fails.


@pytest.fixture
def precision_from_onnx_path(build_full_pipeline):  # noqa: ARG001 — share stub setup
    """Return the _precision_from_onnx_path helper from the module."""
    from build_tensorrt_engine import _precision_from_onnx_path

    return _precision_from_onnx_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/some/dir/vit_fp32.onnx", "fp32"),
        ("vit_bf16.onnx", "bf16"),
        ("dit_bf16.onnx", "bf16"),
        ("llm_bf16.onnx", "bf16"),
        ("model_fp16.onnx", "fp16"),
        ("model_fp8.onnx", "fp8"),
    ],
)
def test_precision_from_onnx_path_reads_filename_suffix(precision_from_onnx_path, path, expected):
    """Recognized suffix overrides the pipeline default."""
    assert precision_from_onnx_path(path, default="bf16") == expected


@pytest.mark.parametrize(
    "path",
    [
        "state_encoder.onnx",
        "action_encoder.onnx",
        "vl_self_attention.onnx",
        "action_decoder.onnx",
        "/abs/path/with_no_precision_suffix.onnx",
    ],
)
def test_precision_from_onnx_path_falls_back_to_default(precision_from_onnx_path, path):
    """No recognized suffix → return the pipeline default unchanged."""
    assert precision_from_onnx_path(path, default="bf16") == "bf16"
    assert precision_from_onnx_path(path, default="fp32") == "fp32"


def test_build_full_pipeline_passes_per_component_precision(tmp_path, build_full_pipeline):
    """ViT is built FP32, every other component inherits the pipeline default.

    The real exporter writes vit_fp32.onnx (FLOAT IO), so the ViT engine must
    be built with precision=fp32 to satisfy the STRONGLY_TYPED sanity check,
    while every other component uses the pipeline-wide default (bf16).
    """
    onnx_dir = tmp_path / "onnx"
    engine_dir = tmp_path / "engines"
    _seed_dummy_onnx_dir(onnx_dir)

    seen_precisions: dict[str, str] = {}

    def fake_build_engine(onnx_path, engine_path, precision, **kwargs):
        seen_precisions[os.path.basename(onnx_path)] = precision
        _fake_build_engine_success(onnx_path, engine_path)

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=fake_build_engine),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )

    assert seen_precisions["vit_fp32.onnx"] == "fp32", (
        f"ViT must be built as fp32 to match its STRONGLY_TYPED FLOAT IO "
        f"(saw precision={seen_precisions['vit_fp32.onnx']!r})"
    )
    for fname in (
        "llm_bf16.onnx",
        "dit_bf16.onnx",
        "vl_self_attention.onnx",
        "state_encoder.onnx",
        "action_encoder.onnx",
        "action_decoder.onnx",
    ):
        assert seen_precisions[fname] == "bf16", (
            f"{fname} should inherit the pipeline default precision=bf16 "
            f"(saw {seen_precisions[fname]!r})"
        )


def test_build_full_pipeline_falls_back_to_vit_bf16_when_fp32_missing(
    tmp_path, build_full_pipeline
):
    """If only vit_bf16.onnx is on disk, ViT picks up the pipeline default (bf16)."""
    onnx_dir = tmp_path / "onnx"
    engine_dir = tmp_path / "engines"
    onnx_dir.mkdir(parents=True)
    # No vit_fp32.onnx — exercise the build_full_pipeline fallback branch.
    fallback_files = [
        "vit_bf16.onnx",
        "llm_bf16.onnx",
        "vl_self_attention.onnx",
        "state_encoder.onnx",
        "action_encoder.onnx",
        "dit_bf16.onnx",
        "action_decoder.onnx",
    ]
    for fname in fallback_files:
        (onnx_dir / fname).touch()

    seen_precisions: dict[str, str] = {}

    def fake_build_engine(onnx_path, engine_path, precision, **kwargs):
        seen_precisions[os.path.basename(onnx_path)] = precision
        _fake_build_engine_success(onnx_path, engine_path)

    with (
        patch("build_tensorrt_engine.derive_shapes_with_hint", return_value=({}, {}, {})),
        patch("build_tensorrt_engine.build_engine", side_effect=fake_build_engine),
    ):
        build_full_pipeline(
            onnx_dir=str(onnx_dir),
            engine_dir=str(engine_dir),
            precision="bf16",
        )

    assert seen_precisions["vit_bf16.onnx"] == "bf16"
