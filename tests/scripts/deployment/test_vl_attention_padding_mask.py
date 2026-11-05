# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check padding isolation across VL ONNX export and TRT runtime boundaries."""

import ast
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.model.modules.dit import SelfAttentionTransformer
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/deployment"))


@pytest.fixture
def runtime():
    # Exercise the actual runtime helpers even on CPU-only CI hosts.
    source = Path(__file__).resolve().parents[3] / "scripts/deployment/trt/trt_model_forward.py"
    names = {"_refine_vl_features", "_validate_vl_attention_engine"}
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = {"torch": torch, "BatchFeature": BatchFeature}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.fixture
def export_on_cpu(monkeypatch):
    """Run the actual VL ONNX export on CPU; the exporter always targets CUDA."""
    export = pytest.importorskip("trt.export_onnx_n1d7")
    onnx = pytest.importorskip("onnx")

    def on_cpu(factory):
        return lambda *args, device=None, **kwargs: factory(*args, **kwargs)

    monkeypatch.setattr(nn.Module, "cuda", lambda self, device=None: self)
    monkeypatch.setattr(torch, "randn", on_cpu(torch.randn))
    monkeypatch.setattr(torch, "ones", on_cpu(torch.ones))

    def run(output_dir, fusion, strict, **kwargs):
        head = SimpleNamespace(
            vl_self_attention=fusion,
            mask_vl_padding=strict,
            config=SimpleNamespace(backbone_embedding_dim=32, strict_vl_padding_mask=strict),
        )
        policy = SimpleNamespace(model=SimpleNamespace(action_head=head))
        path = export.export_vl_self_attention_to_onnx(
            policy, str(output_dir), vl_seq_len=7, batch_size=2, **kwargs
        )
        return onnx.load(path)

    return run


def _fusion(dtype=torch.float32, *, attention_head_dim=16):
    torch.manual_seed(41)
    fusion = SelfAttentionTransformer(
        num_attention_heads=2,
        attention_head_dim=attention_head_dim,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
    )
    return fusion.to(dtype).eval()


def _backbone():
    features = torch.randn(2, 7, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    return BatchFeature(data={"backbone_features": features, "backbone_attention_mask": mask})


def _attention_precision(model):
    """Summarize the ops that decide which dtype TensorRT computes attention in."""
    from onnx import TensorProto, shape_inference

    graph = shape_inference.infer_shapes(model).graph
    dtypes = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    floats = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
    casts = Counter(
        attribute.i
        for node in graph.node
        if node.op_type == "Cast"
        for attribute in node.attribute
        if attribute.name == "to" and attribute.i in floats
    )
    softmax = {dtypes[node.input[0]] for node in graph.node if node.op_type == "Softmax"}
    ops = Counter(node.op_type for node in graph.node)
    return SimpleNamespace(
        inputs=[value.name for value in graph.input], ops=ops, casts=casts, softmax=softmax
    )


def test_strict_vl_export_keeps_legacy_attention_precision(tmp_path, export_on_cpu):
    """A bool SDPA mask would make the exporter compute attention in FP32 with NaN guards."""
    from onnx import TensorProto

    fusion = _fusion(torch.bfloat16)
    legacy = _attention_precision(export_on_cpu(tmp_path / "legacy", fusion, strict=False))
    strict = _attention_precision(export_on_cpu(tmp_path / "strict", fusion, strict=True))
    assert legacy.inputs == ["hidden_states"]
    assert strict.inputs == ["hidden_states", "attention_mask"]
    assert strict.ops["IsNaN"] == 0
    assert strict.casts == legacy.casts
    assert strict.softmax == legacy.softmax == {TensorProto.BFLOAT16}
    # The export-only mask hooks must not stay on the PyTorch module.
    assert not any(block.attn1._forward_pre_hooks for block in fusion.transformer_blocks)


def test_strict_vl_export_removes_hooks_when_export_fails(tmp_path, export_on_cpu, monkeypatch):
    fusion = _fusion(torch.bfloat16)

    def fail_export(*args, **kwargs):
        assert all(block.attn1._forward_pre_hooks for block in fusion.transformer_blocks)
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    with pytest.raises(RuntimeError, match="injected export failure"):
        export_on_cpu(tmp_path, fusion, strict=True)
    assert not any(block.attn1._forward_pre_hooks for block in fusion.transformer_blocks)


def test_finite_mask_floor_is_opt_in_for_vl_export():
    """VL avoids TRT NaNs without changing the existing DiT export mask floor."""
    export = pytest.importorskip("trt.export_onnx_n1d7")
    features = torch.ones(1, 2, 32, dtype=torch.bfloat16)
    mask = torch.tensor([[False, True]])
    for finite in (False, True):
        _, kwargs = export._dit_attn_bool_mask_to_additive_pre_hook(
            None, (features,), {"attention_mask": mask}, finite_mask=finite
        )
        bias = kwargs["attention_mask"]
        assert bias.dtype == features.dtype
        assert bias[0, 1] == 0
        if finite:
            assert bias[0, 0] == torch.finfo(features.dtype).min
        else:
            assert torch.isneginf(bias[0, 0])


def test_strict_vl_export_matches_pytorch_on_valid_tokens(tmp_path, export_on_cpu):
    reference = pytest.importorskip("onnx.reference")
    fusion = _fusion()
    # ONNX's reference evaluator cannot run the BF16 graph, so check the FP32 export.
    evaluator = reference.ReferenceEvaluator(
        export_on_cpu(tmp_path, fusion, strict=True, use_bf16=False)
    )
    backbone = _backbone()
    features = backbone.backbone_features
    mask = backbone.backbone_attention_mask.bool()
    changed = features.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 20

    def run(values):
        inputs = {"hidden_states": values.numpy(), "attention_mask": mask.numpy()}
        return torch.from_numpy(evaluator.run(None, inputs)[0])

    with torch.inference_mode():
        expected = fusion(features, attention_mask=mask)
    actual = run(features)
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(run(changed)[mask], actual[mask], rtol=0, atol=0)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_pytorch_fallback_leaves_backbone_output_unchanged(runtime, strict, input_dtype):
    fusion = _fusion()
    head = SimpleNamespace(
        vlln=nn.LayerNorm(32),
        vl_self_attention=fusion,
        mask_vl_padding=strict,
        config=SimpleNamespace(strict_vl_padding_mask=strict),
    )
    backbone = _backbone()
    backbone["backbone_features"] = backbone.backbone_features.to(input_dtype)
    features = backbone.backbone_features
    mask = backbone.backbone_attention_mask
    original = features.clone()
    with torch.inference_mode():
        actual = runtime._refine_vl_features(head, backbone)
        expected = fusion(head.vlln(features.float()), attention_mask=mask if strict else None)
        repeated = runtime._refine_vl_features(head, backbone)
    assert backbone.backbone_features is features
    assert backbone.backbone_attention_mask is mask
    torch.testing.assert_close(features, original, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


def test_runtime_delivers_mask_and_shapes_to_engine(runtime):
    class Engine:
        in_meta = [("hidden_states", None, torch.float32), ("attention_mask", None, torch.bool)]

        def __init__(self):
            self.shapes = {}
            self.inputs = {}

        def dtype_of(self, name):
            return dict((n, d) for n, _, d in self.in_meta)[name]

        def set_runtime_tensor_shape(self, name, shape):
            self.shapes[name] = shape

        def __call__(self, **inputs):
            self.inputs = inputs
            return {"output": inputs["hidden_states"]}

    engine = Engine()
    backbone = _backbone()
    head = SimpleNamespace(vlln=nn.LayerNorm(32), mask_vl_padding=True)
    output = runtime._refine_vl_features(head, backbone, engine, cache_outputs=False)
    torch.testing.assert_close(output, head.vlln(backbone.backbone_features))
    torch.testing.assert_close(
        engine.inputs["attention_mask"], backbone.backbone_attention_mask.bool()
    )
    assert engine.shapes["attention_mask"] == (2, 7)
    assert engine.shapes["hidden_states"] == (2, 7, 32)
    assert engine.inputs["cache_outputs"] is False


def test_runtime_accepts_legacy_maskless_engine_only_when_strict_is_off(runtime):
    engine = SimpleNamespace(in_meta=[("hidden_states", None, torch.float32)])
    runtime._validate_vl_attention_engine(engine)
    with pytest.raises(ValueError, match="Re-export.*rebuild"):
        runtime._validate_vl_attention_engine(engine, strict_vl_padding_mask=True)
    masked = SimpleNamespace(in_meta=[("attention_mask", None, torch.bool)])
    with pytest.raises(ValueError, match="strict_vl_padding_mask=False"):
        runtime._validate_vl_attention_engine(masked)


def test_runtime_preserves_legacy_engine_inputs(runtime):
    class Engine:
        in_meta = [("hidden_states", None, torch.float32)]

        def dtype_of(self, name):
            assert name == "hidden_states"
            return torch.float32

        def set_runtime_tensor_shape(self, name, shape):
            assert name == "hidden_states"
            assert shape == (2, 7, 32)

        def __call__(self, **inputs):
            assert set(inputs) == {"hidden_states"}
            return {"output": inputs["hidden_states"]}

    head = SimpleNamespace(vlln=nn.Identity(), mask_vl_padding=False)
    backbone = _backbone()
    actual = runtime._refine_vl_features(head, backbone, Engine())
    torch.testing.assert_close(actual, backbone.backbone_features, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strict", [False, True])
def test_exported_vl_engine_respects_strict_mode_and_is_graph_safe(tmp_path, strict):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")
    from trt import trt_model_forward as runtime
    from trt.build_tensorrt_engine import build_engine
    from trt.export_onnx_n1d7 import export_vl_self_attention_to_onnx

    # Head dim 64 and a longer sequence exercise TRT's fused mha_v2 tactic.
    fusion = _fusion(torch.bfloat16, attention_head_dim=64).cuda()
    embedding_dim = 128
    head = SimpleNamespace(
        vl_self_attention=fusion,
        mask_vl_padding=strict,
        config=SimpleNamespace(backbone_embedding_dim=embedding_dim, strict_vl_padding_mask=strict),
        vlln=nn.Identity(),
    )
    policy = SimpleNamespace(model=SimpleNamespace(action_head=head))
    path = export_vl_self_attention_to_onnx(policy, str(tmp_path), vl_seq_len=7, batch_size=2)
    expected_inputs = ["hidden_states", "attention_mask"] if strict else ["hidden_states"]
    assert [value.name for value in onnx.load(path).graph.input] == expected_inputs
    engine_path = str(tmp_path / "vl_self_attention.engine")
    shapes = {"hidden_states": (2, 7, embedding_dim)}
    min_shapes = {"hidden_states": (2, 2, embedding_dim)}
    max_shapes = {"hidden_states": (2, 287, embedding_dim)}
    if strict:
        shapes["attention_mask"] = (2, 7)
        min_shapes["attention_mask"] = (2, 2)
        max_shapes["attention_mask"] = (2, 287)
    build_engine(
        path,
        engine_path,
        precision="bf16",
        workspace_mb=256,
        min_shapes=min_shapes,
        opt_shapes=shapes,
        max_shapes=max_shapes,
    )
    engine = runtime.GraphSafeEngine(engine_path)
    try:
        for seq_len in (7, 10, 287):
            features = torch.randn(2, seq_len, embedding_dim, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
            mask[0, -3:] = False
            mask[1, :2] = False
            if seq_len == 287:
                mask[0, :128] = False
                mask[1, :64] = False
            changed = features.clone()
            changed[~mask] = torch.randn_like(changed[~mask]) * 20

            def backbone(values):
                return BatchFeature(
                    data={"backbone_features": values, "backbone_attention_mask": mask}
                )

            with torch.inference_mode():
                reference = fusion(features, attention_mask=mask if strict else None)
                actual = runtime._refine_vl_features(head, backbone(features), engine).clone()
                perturbed = runtime._refine_vl_features(head, backbone(changed), engine).clone()
            torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.03)
            assert torch.isfinite(actual).all()
            if strict:
                torch.testing.assert_close(actual[mask], perturbed[mask], rtol=0, atol=0)
            else:
                with torch.inference_mode():
                    reference = fusion(changed)
                torch.testing.assert_close(perturbed, reference, rtol=0.03, atol=0.03)

        # CUDA graphs must read changing mask values from the same input buffer.
        with torch.inference_mode():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = runtime._refine_vl_features(head, backbone(features), engine)
            mask[0, -1] = True
            graph.replay()
            torch.cuda.synchronize()
            reference = fusion(features, attention_mask=mask if strict else None)
            torch.testing.assert_close(captured, reference, rtol=0.03, atol=0.03)
    finally:
        engine.close()


@pytest.mark.parametrize("strict", [False, True])
def test_calibration_respects_strict_mode(monkeypatch, strict):
    from trt import n1d7_calibration as calibration

    backbone = _backbone()
    expected = backbone.backbone_attention_mask.bool()
    model = nn.Linear(1, 1)
    model.action_head = SimpleNamespace(vlln=nn.Identity(), mask_vl_padding=strict)
    model.backbone = lambda inputs: backbone
    policy = SimpleNamespace(model=model, modality_configs={})
    monkeypatch.setattr(calibration, "LeRobotEpisodeLoader", lambda **kwargs: [[None]])
    monkeypatch.setattr(calibration, "_prepare_step_observation", lambda *args: {})
    monkeypatch.setattr(calibration, "_collate_inputs", lambda *args: {})
    batches = calibration.capture_calibration_batches(policy, "unused", 1, {"vl_sa"})
    if strict:
        torch.testing.assert_close(batches["vl_sa"][0]["attention_mask"], expected)
    else:
        assert set(batches["vl_sa"][0]) == {"hidden_states"}
    torch.testing.assert_close(batches["vl_sa"][0]["hidden_states"], backbone.backbone_features)
