# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check padding isolation across VL ONNX export and TRT runtime boundaries."""

import ast
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
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def _backbone():
    features = torch.randn(2, 7, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]])
    return BatchFeature(data={"backbone_features": features, "backbone_attention_mask": mask})


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
    head = SimpleNamespace(
        vlln=nn.LayerNorm(32), config=SimpleNamespace(strict_vl_padding_mask=True)
    )
    output = runtime._refine_vl_features(head, backbone, engine, cache_outputs=False)
    torch.testing.assert_close(output, head.vlln(backbone.backbone_features))
    torch.testing.assert_close(
        engine.inputs["attention_mask"], backbone.backbone_attention_mask.bool()
    )
    assert engine.shapes["attention_mask"] == (2, 7)
    assert engine.shapes["hidden_states"] == (2, 7, 32)
    assert engine.inputs["cache_outputs"] is False
    del backbone["backbone_attention_mask"]
    runtime._refine_vl_features(head, backbone, engine)
    assert engine.inputs["attention_mask"].all()


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

    head = SimpleNamespace(vlln=nn.Identity(), config=SimpleNamespace(strict_vl_padding_mask=False))
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

    torch.manual_seed(41)
    fusion = (
        SelfAttentionTransformer(
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=4,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    head = SimpleNamespace(
        vl_self_attention=fusion,
        config=SimpleNamespace(backbone_embedding_dim=32, strict_vl_padding_mask=strict),
        vlln=nn.Identity(),
    )
    policy = SimpleNamespace(model=SimpleNamespace(action_head=head))
    path = export_vl_self_attention_to_onnx(policy, str(tmp_path), vl_seq_len=7, batch_size=2)
    expected_inputs = ["hidden_states", "attention_mask"] if strict else ["hidden_states"]
    assert [value.name for value in onnx.load(path).graph.input] == expected_inputs
    engine_path = str(tmp_path / "vl_self_attention.engine")
    shapes = {"hidden_states": (2, 7, 32)}
    min_shapes = {"hidden_states": (2, 2, 32)}
    max_shapes = {"hidden_states": (2, 12, 32)}
    if strict:
        shapes["attention_mask"] = (2, 7)
        min_shapes["attention_mask"] = (2, 2)
        max_shapes["attention_mask"] = (2, 12)
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
        for seq_len in (7, 10):
            features = torch.randn(2, seq_len, 32, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
            mask[0, -3:] = False
            mask[1, :2] = False
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


@pytest.mark.parametrize("mask_present", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_calibration_respects_strict_mode(monkeypatch, mask_present, strict):
    from trt import n1d7_calibration as calibration

    backbone = _backbone()
    expected = backbone.backbone_attention_mask.bool()
    if not mask_present:
        del backbone["backbone_attention_mask"]
        expected = torch.ones(2, 7, dtype=torch.bool)
    model = nn.Linear(1, 1)
    model.action_head = SimpleNamespace(
        vlln=nn.Identity(), config=SimpleNamespace(strict_vl_padding_mask=strict)
    )
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
