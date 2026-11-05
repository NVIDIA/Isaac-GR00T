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

"""CPU checks for quantized artifact contracts and export graph transformations."""

import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


_DEPLOY_DIR = str(Path(__file__).resolve().parents[3] / "scripts" / "deployment")
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from trt.n1d7_optimization_config import (  # noqa: E402
    COMPONENT_PRECISIONS,
    OptimizationArtifacts,
    component_precisions,
    trt_boundary_precision,
)


_PIPELINE_SCRIPT = os.path.join(_DEPLOY_DIR, "build_trt_pipeline.py")

# ---------------------------------------------------------------------------
# Quantization-policy artifact naming tests (CPU-only)
#
# Regression class: `--quantization fp8 --export-mode dit_only` once exported
# dit_fp8.onnx while the pipeline build step looked for dit_bf16.onnx. Export,
# build, benchmark, and runtime must all derive artifact names from the shared
# COMPONENT_PRECISIONS table / export metadata.
# ---------------------------------------------------------------------------


def _metadata_for(quantization: str) -> dict:
    precisions = component_precisions(quantization)
    return {
        "vit_precision": precisions["vit"],
        "llm_precision": precisions["llm"],
        "dit_precision": precisions["dit"],
        "aux_precision": precisions["aux"],
        "quantization": quantization,
    }


def test_component_precisions_none_matches_baseline_artifacts() -> None:
    """The 'none' policy must reproduce the baseline export exactly."""
    assert component_precisions("none") == {
        "vit": "fp32",
        "llm": "bf16",
        "dit": "bf16",
        "aux": "bf16",
    }


@pytest.mark.parametrize("quantization", ["fp8", "mixed_nvfp4"])
def test_quantized_recipes_build_with_fp16_boundary(quantization: str) -> None:
    """FP8/NVFP4 artifacts carry Q/DQ nodes internally; TRT I/O stays FP16."""
    precisions = component_precisions(quantization)
    for component in ("vit", "llm", "dit"):
        assert trt_boundary_precision(precisions[component]) == "fp16"
    assert precisions["aux"] == "fp16"


@pytest.mark.parametrize("quantization", sorted(COMPONENT_PRECISIONS))
def test_runtime_engine_path_matches_export_naming(quantization: str, tmp_path) -> None:
    """The runtime resolves exactly the engine name the export/build produce."""
    import json

    pytest.importorskip("tensorrt", reason="TensorRT runtime is unavailable on this host")
    from trt.trt_model_forward import _engine_path

    dit_precision = component_precisions(quantization)["dit"]
    engine_name = f"dit_{dit_precision}.engine"
    (tmp_path / "export_metadata.json").write_text(json.dumps(_metadata_for(quantization)))
    (tmp_path / engine_name).touch()

    assert _engine_path(str(tmp_path), "dit") == str(tmp_path / engine_name)


def test_runtime_engine_path_reports_missing_quantized_engine(tmp_path) -> None:
    """A quantized bundle with a missing engine must resolve to the expected
    quantized filename (so the caller errors with the right path), not silently
    fall back to the legacy BF16 name."""
    import json

    pytest.importorskip("tensorrt", reason="TensorRT runtime is unavailable on this host")
    from trt.trt_model_forward import _engine_path

    (tmp_path / "export_metadata.json").write_text(json.dumps(_metadata_for("mixed_nvfp4")))

    assert _engine_path(str(tmp_path), "dit").endswith("dit_nvfp4.engine")


def test_optimization_artifacts_resolved_from_flat_metadata() -> None:
    metadata = {
        "optimized_action_head": True,
        "dit_precision": "nvfp4",
        "action_head_format_version": 1,
    }
    artifacts = OptimizationArtifacts.from_metadata("/bundle", metadata, artifact_kind="engine")
    assert artifacts.cross_kv == "/bundle/dit_cross_kv_nvfp4.engine"
    assert artifacts.constants == "/bundle/action_head_constants.pt"
    assert artifacts.format_version == 1

    assert (
        OptimizationArtifacts.from_metadata(
            "/bundle", {"optimized_action_head": False}, artifact_kind="onnx"
        )
        is None
    )
    assert OptimizationArtifacts.from_metadata("/bundle", None, artifact_kind="onnx") is None


@pytest.mark.serial
def test_cli_rejects_optimized_profile_with_dit_only() -> None:
    """Invalid mode combinations fail fast, before any model loading."""
    result = subprocess.run(
        [
            sys.executable,
            _PIPELINE_SCRIPT,
            "--model-path",
            "/unused",
            "--execution-profile",
            "optimized",
            "--export-mode",
            "dit_only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--execution-profile optimized requires" in combined


@pytest.mark.parametrize("bias", [True, False])
def test_fused_gated_mlp_matches_unfused_outputs_and_gradients(bias):
    """Fusing gate/up projections preserves the Qwen MLP numerical contract."""
    torch.manual_seed(17)
    source = SimpleNamespace(
        config=None,
        hidden_size=8,
        intermediate_size=12,
        gate_proj=nn.Linear(8, 12, bias=bias),
        up_proj=nn.Linear(8, 12, bias=bias),
        down_proj=nn.Linear(12, 8, bias=bias),
        act_fn=nn.functional.silu,
    )
    FusedQwenMLP = pytest.importorskip("trt.export_onnx_n1d7", exc_type=ImportError).FusedQwenMLP
    fused = FusedQwenMLP(source)
    reference_input = torch.randn(2, 5, 8, requires_grad=True)
    fused_input = reference_input.detach().clone().requires_grad_()
    expected = source.down_proj(
        source.act_fn(source.gate_proj(reference_input)) * source.up_proj(reference_input)
    )
    actual = fused(fused_input)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(fused_input.grad, reference_input.grad)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_cached_dit_matches_masked_alternating_attention(batch_size, monkeypatch):
    """Cached K/V and timestep modulations preserve the reference DiT output."""
    pytest.importorskip("tree", reason="Full model dependencies are unavailable on this host")
    from gr00t.model.modules.dit import AlternateVLDiT
    from trt.export_onnx_n1d7 import CachedCrossKVDiT, DiTCrossKV

    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    torch.manual_seed(17)
    dit = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=6,
        num_layers=4,
        dropout=0,
        final_dropout=False,
        interleave_self_attention=True,
        cross_attention_dim=12,
    ).eval()
    actions = torch.randn(batch_size, 5, 16)
    context = torch.randn(batch_size, 7, 12)
    image_mask = torch.tensor([[True, True, True, False, False, False, False]]).expand(
        batch_size, -1
    )
    context_mask = torch.tensor([[True, True, True, True, True, False, False]]).expand(
        batch_size, -1
    )
    timestep = torch.tensor([0.0]).expand(batch_size)
    with torch.inference_mode():
        expected = dit(
            actions, context, timestep, image_mask=image_mask, backbone_attention_mask=context_mask
        )
        temb = dit.timestep_encoder(timestep)
        modulations = torch.stack(
            [b.norm1.linear(b.norm1.silu(temb)) for b in dit.transformer_blocks]
        )
        output_modulation = dit.proj_out_1(nn.functional.silu(temb))
        keys, values = DiTCrossKV(dit)(context)
        actual = CachedCrossKVDiT(dit)(
            actions, keys, values, modulations, output_modulation, image_mask, context_mask
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_dit_export_attention_masks_use_a_finite_floor(monkeypatch):
    """Both exported DiT attention paths mask padding with finfo.min, never -inf.

    CPU math attention softmaxes a -inf row safely, so the masks are checked directly.
    """
    pytest.importorskip("tree", reason="Full model dependencies are unavailable on this host")
    from gr00t.model.modules.dit import AlternateVLDiT
    from trt.export_onnx_n1d7 import (
        CachedCrossKVDiT,
        DiTCrossKV,
        _patch_dit_attn_for_export,
        _unpatch_dit_attn,
    )

    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    torch.manual_seed(17)
    dit = AlternateVLDiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=6,
        num_layers=4,
        dropout=0,
        final_dropout=False,
        interleave_self_attention=True,
        cross_attention_dim=12,
    ).eval()
    actions = torch.randn(2, 5, 16)
    context = torch.randn(2, 7, 12)
    context_mask = torch.tensor([[True] * 5 + [False] * 2, [False] * 7])
    image_mask = context_mask & torch.tensor([True] * 3 + [False] * 4)
    timestep = torch.zeros(2)
    with torch.inference_mode():
        expected = dit(
            actions[:1],
            context[:1],
            timestep[:1],
            image_mask=image_mask[:1],
            backbone_attention_mask=context_mask[:1],
        )
        masks = []
        attention = nn.functional.scaled_dot_product_attention

        def recording_attention(*args, attn_mask=None, **kwargs):
            if attn_mask is not None:
                masks.append(attn_mask)
            return attention(*args, attn_mask=attn_mask, **kwargs)

        monkeypatch.setattr(nn.functional, "scaled_dot_product_attention", recording_attention)
        handles = _patch_dit_attn_for_export(dit)
        try:
            patched = dit(
                actions,
                context,
                timestep,
                image_mask=image_mask,
                backbone_attention_mask=context_mask,
            )
        finally:
            _unpatch_dit_attn(handles)
        temb = dit.timestep_encoder(timestep)
        modulations = torch.stack(
            [b.norm1.linear(b.norm1.silu(temb)) for b in dit.transformer_blocks]
        )
        output_modulation = dit.proj_out_1(nn.functional.silu(temb))
        keys, values = DiTCrossKV(dit)(context)
        patched_masks = len(masks)
        cached = CachedCrossKVDiT(dit)(
            actions, keys, values, modulations, output_modulation, image_mask, context_mask
        )
    assert 0 < patched_masks < len(masks)
    for mask in masks:
        assert torch.isfinite(mask).all()
        assert mask.min() == torch.finfo(mask.dtype).min
    for actual in (patched, cached):
        torch.testing.assert_close(actual[:1], expected, rtol=1e-4, atol=1e-5)


def _strict_head(strict):
    return SimpleNamespace(
        mask_vl_padding=strict, config=SimpleNamespace(strict_vl_padding_mask=strict)
    )


def _construct_optimized_runner(
    tmp_path,
    export_mode,
    engine_present,
    action_head,
    vl_self_attention_exported=None,
    engine_has_mask=None,
    closed=None,
):
    """Run the real runner constructor with engine loading stubbed.

    This keeps the regressions in CPU CI, without importing CUDA-only TensorRT
    runtime dependencies. Returns the runner, the loaded paths, and the VL engine path.
    """
    source = Path(_DEPLOY_DIR) / "trt" / "trt_model_forward.py"
    tree = ast.parse(source.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OptimizedActionHeadRunner"
    )
    constructor = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    engine_path = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "vl_self_attention_engine_path"
    )
    metadata = {
        "optimized_action_head": True,
        "dit_precision": "bf16",
        "action_head_format_version": 1,
        "export_mode": export_mode,
    }
    if vl_self_attention_exported is not None:
        metadata["vl_self_attention_exported"] = vl_self_attention_exported
    artifacts = OptimizationArtifacts.from_metadata(str(tmp_path), metadata, artifact_kind="engine")
    Path(artifacts.cross_kv).touch()
    Path(artifacts.constants).touch()
    vl_engine = tmp_path / "vl_self_attention.engine"
    if engine_present:
        vl_engine.touch()
    loaded = []
    closed = [] if closed is None else closed

    def load_engine(path):
        loaded.append(path)
        strict = action_head.config.strict_vl_padding_mask
        has_mask = strict if engine_has_mask is None else engine_has_mask
        in_meta = [("hidden_states", (), torch.bfloat16)]
        if has_mask:
            in_meta.append(("attention_mask", (), torch.bool))
        return SimpleNamespace(path=path, in_meta=in_meta, close=lambda: closed.append(path))

    mask_validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_vl_attention_engine"
    )
    scope = {
        "os": os,
        "logger": logging.getLogger(__name__),
        "OptimizationArtifacts": OptimizationArtifacts,
        "ACTION_HEAD_FORMAT_VERSION": 1,
        "GraphSafeEngine": load_engine,
    }
    exec(
        compile(
            ast.Module(body=[engine_path, mask_validator, constructor], type_ignores=[]),
            str(source),
            "exec",
        ),
        scope,
    )
    runtime = SimpleNamespace(
        _validate_interfaces=lambda: None,
        _load_constants=lambda path: {},
        _make_position_embeddings=lambda: None,
    )
    scope["__init__"](
        runtime, action_head, str(tmp_path), metadata, str(tmp_path / "dit_bf16.engine")
    )
    return runtime, loaded, vl_engine


@pytest.mark.parametrize(
    "export_mode,engine_present,expect_loaded",
    [("action_head", True, False), ("full_pipeline", True, True), ("full_pipeline", False, False)],
)
def test_optimized_action_head_ignores_stale_vl_engine(
    export_mode,
    engine_present,
    expect_loaded,
    tmp_path,
):
    """A reused bundle must not replace PyTorch VL attention in action-head mode."""
    runtime, loaded, vl_engine = _construct_optimized_runner(
        tmp_path, export_mode, engine_present, _strict_head(False)
    )
    assert (str(vl_engine) in loaded) == expect_loaded
    assert (runtime.vl_sa_engine is not None) == expect_loaded


def test_optimized_action_head_loads_mask_aware_vl_engine_with_strict_vl_padding_mask(tmp_path):
    head = _strict_head(True)
    runtime, loaded, vl_engine = _construct_optimized_runner(tmp_path, "full_pipeline", True, head)
    assert str(vl_engine) in loaded
    assert runtime.vl_sa_engine is not None


@pytest.mark.parametrize("strict", [True, False])
def test_optimized_action_head_refuses_vl_engine_with_mismatched_mask_input(strict, tmp_path):
    """An engine exported with the other strict_vl_padding_mask setting must be rebuilt."""
    head = _strict_head(strict)
    closed = []
    with pytest.raises(ValueError, match="attention_mask binding"):
        _construct_optimized_runner(
            tmp_path, "full_pipeline", True, head, engine_has_mask=not strict, closed=closed
        )
    assert closed == [str(tmp_path / "vl_self_attention.engine")]


def test_optimized_action_head_ignores_vl_engine_left_from_an_earlier_export(tmp_path):
    """An export that kept VL attention in PyTorch makes an engine beside it stale."""
    head = _strict_head(False)
    runtime, loaded, vl_engine = _construct_optimized_runner(
        tmp_path, "full_pipeline", True, head, vl_self_attention_exported=False
    )
    assert str(vl_engine) not in loaded
    assert runtime.vl_sa_engine is None
