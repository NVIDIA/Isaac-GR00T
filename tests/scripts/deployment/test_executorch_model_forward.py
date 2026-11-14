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

"""``ExecuTorchEngine`` must stand in for ``trt_torch.Engine`` in the TensorRT forward
functions, and ``setup_executorch_programs`` must refuse the same mismatched bundles
as the TensorRT loader."""

import importlib.machinery
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import get_args

import pytest
import torch


_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "scripts" / "deployment"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))


_SIX = ("vit", "llm", "state_encoder", "action_encoder", "dit", "action_decoder")


def _forward_module(monkeypatch):
    """Import the loader module. Nothing under test calls TensorRT, so where the CPU job
    has no ``tensorrt`` a bare stub satisfies the import chain through ``trt_torch``."""
    try:
        import tensorrt  # noqa: F401
    except (ImportError, OSError):
        stub = types.ModuleType("tensorrt")
        stub.__spec__ = importlib.machinery.ModuleSpec("tensorrt", None)
        monkeypatch.setitem(sys.modules, "tensorrt", stub)
    from trt import executorch_model_forward

    return executorch_model_forward


@pytest.fixture(autouse=True)
def _forget_modules_imported_over_the_stub():
    """Modules imported over the ``tensorrt`` stub must not outlive it, or a later test
    that skips without TensorRT would run against the stub."""
    before = set(sys.modules)
    yield
    if "tensorrt" not in sys.modules:
        for name in set(sys.modules) - before:
            if name == "trt" or name.startswith("trt."):
                del sys.modules[name]


class _FakeEngine:
    def __init__(self, program_dir, io):
        self.io = io
        self.in_meta = [[name, (1,), torch.float32] for name in io.get("inputs", [])]
        self.out_meta = [["out", None, None]]

    def dtype_of(self, name):
        return torch.bfloat16


def _stub_policy():
    language_model = SimpleNamespace(
        get_input_embeddings=lambda: "embeddings", layers="layers", norm="norm"
    )
    qwen = SimpleNamespace(
        model=SimpleNamespace(visual="visual", language_model=language_model),
        config=SimpleNamespace(image_token_id=7),
    )
    action_head = SimpleNamespace(
        mask_vl_padding=True,
        vl_self_attention="vl_self_attention",
        model="dit",
        state_encoder="state_encoder",
        action_encoder="action_encoder",
        action_decoder="action_decoder",
        action_dim=32,
        config=SimpleNamespace(max_action_dim=32),
    )
    return SimpleNamespace(
        model=SimpleNamespace(backbone=SimpleNamespace(model=qwen), action_head=action_head)
    )


def _io(names, program_dir=None):
    io = {name: {"file": f"{name}.pte"} for name in names}
    if "dit" in io:
        io["dit"]["inputs"] = ["sa_embs", "vl_embs"]
    if program_dir is not None:
        for entry in io.values():
            (program_dir / entry["file"]).touch()
    return io


@pytest.fixture
def loader(monkeypatch, tmp_path):
    module = _forward_module(monkeypatch)

    calls = {}
    monkeypatch.setattr(module, "ExecuTorchEngine", _FakeEngine)
    monkeypatch.setattr(
        module,
        "_validate_vl_attention_engine",
        lambda engine, mask_vl_padding: calls.setdefault("vl", (engine.io, mask_vl_padding)),
    )
    monkeypatch.setattr(
        module,
        "assert_dit_engine_matches_action_padding",
        lambda config, names, source: calls.setdefault("dit", (config, names)),
    )

    def setup(metadata, mask_vl_padding=True):
        def checked(policy, program_dir, source):
            calls["policy"] = (policy, program_dir)
            return metadata

        monkeypatch.setattr(module, "assert_engine_matches_policy", checked)
        policy = _stub_policy()
        policy.model.action_head.mask_vl_padding = mask_vl_padding
        module.setup_executorch_programs(policy, str(tmp_path))
        return policy

    return setup, calls


@pytest.mark.parametrize("mask_vl_padding", [True, False])
def test_setup_runs_the_tensorrt_loader_checks(loader, tmp_path, mask_vl_padding):
    setup, calls = loader
    io = _io(_SIX + ("vl_self_attention",), tmp_path)
    grid = [[1, 16, 16]]
    policy = setup({"executorch_io": io, "vit_grid_thw": grid}, mask_vl_padding)
    backbone, action_head = policy.model.backbone, policy.model.action_head

    assert calls["policy"] == (policy, str(tmp_path))
    assert backbone._vit_baked_grid_thw == grid
    assert calls["vl"] == (io["vl_self_attention"], mask_vl_padding)
    assert calls["dit"] == (action_head.config, {"sa_embs", "vl_embs"})
    assert backbone.vit_engine.io is io["vit"] and backbone.llm_engine.io is io["llm"]
    for name in ("state_encoder", "action_encoder", "dit", "action_decoder"):
        assert getattr(action_head, f"{name}_engine").io is io[name]
    assert action_head.vl_sa_engine.io is io["vl_self_attention"]
    for name in ("vl_self_attention", "model", "state_encoder", "action_encoder", "action_decoder"):
        assert not hasattr(action_head, name)
    assert backbone.forward.func.__name__ == "qwen3_backbone_full_trt_forward"
    assert action_head.get_action.func.__name__ == "action_head_tensorrt_forward"
    assert not hasattr(backbone.model.model, "visual")
    assert not hasattr(backbone.model.model.language_model, "layers")


def test_setup_keeps_a_vl_self_attention_that_was_not_exported(loader, tmp_path):
    setup, calls = loader
    policy = setup({"executorch_io": _io(_SIX, tmp_path), "vl_self_attention_exported": False})
    action_head = policy.model.action_head

    assert action_head.vl_sa_engine is None
    assert action_head.vl_self_attention == "vl_self_attention"
    assert "vl" not in calls
    assert not hasattr(action_head, "model")


@pytest.mark.parametrize("lost", ["vl_self_attention", "action_decoder"])
def test_setup_refuses_a_partial_export_before_changing_the_policy(
    loader, monkeypatch, tmp_path, lost
):
    setup, _calls = loader
    from trt import executorch_model_forward as module

    policy = _stub_policy()
    io = _io(_SIX + ("vl_self_attention",), tmp_path)
    if lost == "vl_self_attention":
        del io[lost]
    else:
        (tmp_path / io[lost]["file"]).unlink()
    monkeypatch.setattr(
        module, "assert_engine_matches_policy", lambda *a, **k: {"executorch_io": io}
    )
    with pytest.raises(ValueError, match=lost):
        module.setup_executorch_programs(policy, str(tmp_path))
    assert policy.model.backbone.model.model.visual == "visual"
    assert policy.model.action_head.model == "dit"


@pytest.mark.parametrize(
    ("exported", "vl_self_attention_exported", "missing"),
    [
        (_SIX + ("vl_self_attention",), True, None),
        (_SIX, False, None),
        (_SIX, True, "vl_self_attention"),
        (_SIX[:-1] + ("vl_self_attention",), True, "action_decoder"),
        (_SIX, None, "vl_self_attention"),
        (None, True, "no ExecuTorch export"),
    ],
)
def test_partial_exports_are_refused(
    monkeypatch, tmp_path, exported, vl_self_attention_exported, missing
):
    executorch_components = _forward_module(monkeypatch).executorch_components

    metadata = {}
    if vl_self_attention_exported is not None:
        metadata["vl_self_attention_exported"] = vl_self_attention_exported
    if exported is not None:
        metadata["executorch_io"] = _io(exported, tmp_path)
    if missing is None:
        assert executorch_components(str(tmp_path), metadata).keys() == set(exported)
    else:
        with pytest.raises(ValueError, match=missing):
            executorch_components(str(tmp_path), metadata)


def test_a_missing_data_file_is_refused(monkeypatch, tmp_path):
    executorch_components = _forward_module(monkeypatch).executorch_components

    io = _io(_SIX + ("vl_self_attention",), tmp_path)
    io["dit"]["data_file"] = "dit.ptd"
    with pytest.raises(ValueError, match="dit"):
        executorch_components(str(tmp_path), {"executorch_io": io})
    (tmp_path / "dit.ptd").touch()
    assert executorch_components(str(tmp_path), {"executorch_io": io}) is io


@pytest.mark.parametrize("absent", ["executorch.runtime", "torch_tensorrt_executorch_runtime"])
def test_check_executorch_export_fails_before_slow_steps(monkeypatch, tmp_path, absent):
    module = _forward_module(monkeypatch)

    def import_module(name):
        if name == absent:
            raise ImportError(name)

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    with pytest.raises(ImportError, match=absent):
        module.check_executorch_export(str(tmp_path))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: None)
    with pytest.raises(FileNotFoundError, match="No ExecuTorch export"):
        module.check_executorch_export(str(tmp_path))
    io = _io(_SIX + ("vl_self_attention",), tmp_path)
    (tmp_path / "export_metadata.json").write_text(json.dumps({"executorch_io": io}))
    assert module.check_executorch_export(str(tmp_path)) == io


class _RecordingMethod:
    def __init__(self):
        self.inputs = None

    def execute(self, inputs):
        self.inputs = inputs
        return [inputs[0] + 1, inputs[1] * 2]


class _FakeRuntime:
    def load_program(self, path, data_path=None):
        self.loaded = (path, data_path)
        return self

    def load_method(self, name):
        assert name == "forward"
        return _RecordingMethod()


def _engine(monkeypatch, program_dir, with_data_file=True):
    executorch_model_forward = _forward_module(monkeypatch)

    runtime = _FakeRuntime()
    monkeypatch.setattr(executorch_model_forward, "_runtime", runtime)
    data_path = program_dir / "component.ptd"
    if with_data_file:
        data_path.touch()
    io = {
        "file": "component.pte",
        "inputs": [
            {"name": "a", "shape": [2], "dtype": "float32"},
            {"name": "b", "shape": [2], "dtype": "int64"},
        ],
        "outputs": [{"name": "x", "dtype": "float32"}, {"name": "y", "dtype": "int64"}],
        "data_file": "component.ptd" if with_data_file else None,
    }
    engine = executorch_model_forward.ExecuTorchEngine(str(program_dir), io)
    expected_data = str(data_path) if with_data_file else None
    assert runtime.loaded == (str(program_dir / "component.pte"), expected_data)
    return engine


@pytest.mark.parametrize("with_data_file", [True, False])
def test_engine_loads_the_recorded_program(monkeypatch, tmp_path, with_data_file):
    engine = _engine(monkeypatch, tmp_path, with_data_file)
    assert [dtype for _name, _shape, dtype in engine.in_meta] == [torch.float32, torch.int64]
    assert engine.dtype_of("b") == torch.int64
    assert [(item[0], item[2]) for item in engine.out_meta] == [
        ("x", torch.float32),
        ("y", torch.int64),
    ]
    with pytest.raises(AssertionError, match="expected torch.float32"):
        engine(torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(AssertionError, match="expected cuda"):
        engine(torch.zeros(2), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="3 inputs given"):
        engine(torch.zeros(2), torch.zeros(2), torch.zeros(2))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="inputs must be CUDA tensors")
def test_engine_binds_inputs_in_program_order(monkeypatch, tmp_path):
    engine = _engine(monkeypatch, tmp_path)
    a = torch.zeros(2, device="cuda")
    b = torch.ones(2, dtype=torch.int64, device="cuda")

    out = engine(b=b, a=a, cache_outputs=True)
    assert [t.data_ptr() for t in engine._method.inputs] == [a.data_ptr(), b.data_ptr()]
    assert out["x"].tolist() == [1, 1] and out["y"].tolist() == [2, 2]
    assert engine(a, b, return_list=True)[1].tolist() == [2, 2]
    with pytest.raises(ValueError, match="missing inputs"):
        engine(a)


def test_output_dir_guards(tmp_path):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ImportError)
    everything = list(get_args(export.Component))

    fresh = tmp_path / "fresh"
    assert export._check_output_dir(str(fresh), everything) == str(fresh / "export_metadata.json")

    for built in ("llm_bf16.engine", "llm_bf16.onnx"):
        folder = tmp_path / built
        folder.mkdir()
        (folder / built).touch()
        with pytest.raises(ValueError, match="TensorRT engines or ONNX"):
            export._check_output_dir(str(folder), everything)

    onnx = tmp_path / "onnx"
    onnx.mkdir()
    (onnx / "export_metadata.json").write_text(json.dumps({"export_mode": "full_pipeline"}))
    with pytest.raises(ValueError, match="another export"):
        export._check_output_dir(str(onnx), everything)

    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "export_metadata.json").write_text(json.dumps({"executorch_io": {}}))
    export._check_output_dir(str(previous), everything)
    with pytest.raises(ValueError, match="a subset would replace"):
        export._check_output_dir(str(previous), ["dit"])


class _MaskedAdd(torch.nn.Module):
    def forward(self, x, y, *masks):
        out = x + y
        for mask in masks:
            out = out * mask
        return out


@pytest.mark.parametrize("num_masks", [0, 2])
def test_dynamic_shapes_follow_the_bound_masks(num_masks):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ImportError)

    seq = torch.export.Dim("seq", min=2, max=64)
    inputs = (torch.ones(1, 8), torch.ones(1, 8)) + (torch.ones(1, 8),) * num_masks
    flat = [{1: seq}] * len(inputs)
    module = _MaskedAdd()
    program = torch.export.export(
        module, inputs, dynamic_shapes=export._nest_varargs_shapes(module, flat), strict=False
    )
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    assert len(placeholders) == len(inputs)
    assert all(isinstance(node.meta["val"].shape[1], torch.SymInt) for node in placeholders)
