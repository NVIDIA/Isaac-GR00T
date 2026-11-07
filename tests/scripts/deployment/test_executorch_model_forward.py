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

import copy
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
                # The parent package keeps its submodules as attributes, and a later
                # ``from trt import ...`` would return the forgotten copy.
                parent, _, child = name.rpartition(".")
                if parent in sys.modules:
                    vars(sys.modules[parent]).pop(child, None)


class _FakeEngine:
    def __init__(self, program_dir, io):
        self.io = io
        self.in_meta = [[name, (1,), torch.float32] for name in io.get("inputs", [])]


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


def _io(names, program_dir):
    io = {name: {"file": f"{name}.pte"} for name in names}
    if "dit" in io:
        io["dit"]["inputs"] = ["sa_embs", "vl_embs"]
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
    assert backbone._embedding_layer == "embeddings" and backbone._image_token_id == 7
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
    from trt import executorch_model_forward as module

    policy = _stub_policy()
    untouched = copy.deepcopy(policy)
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
    assert policy == untouched


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


@pytest.mark.parametrize("file", [None, ""])
def test_a_program_without_a_file_name_is_refused(monkeypatch, tmp_path, file):
    executorch_components = _forward_module(monkeypatch).executorch_components

    io = _io(_SIX + ("vl_self_attention",), tmp_path)
    io["llm"]["file"] = file
    with pytest.raises(ValueError, match="llm"):
        executorch_components(str(tmp_path), {"executorch_io": io})


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
            {"name": "state", "shape": [2], "dtype": "float32"},
            {"name": "embodiment_id", "shape": [2], "dtype": "int64"},
        ],
        "outputs": [{"name": "y", "dtype": "float32"}, {"name": "x", "dtype": "int64"}],
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
    assert engine.dtype_of("embodiment_id") == torch.int64
    assert [(item[0], item[2]) for item in engine.out_meta] == [
        ("y", torch.float32),
        ("x", torch.int64),
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

    out = engine(embodiment_id=b, state=a, cache_outputs=True)
    assert [t.data_ptr() for t in engine._method.inputs] == [a.data_ptr(), b.data_ptr()]
    assert out["y"].tolist() == [1, 1] and out["x"].tolist() == [2, 2]
    assert engine(a, b, return_list=True)[1].tolist() == [2, 2]
    with pytest.raises(ValueError, match="missing inputs"):
        engine(a)


def test_close_releases_every_program(monkeypatch, tmp_path):
    module = _forward_module(monkeypatch)
    model = torch.nn.Module()
    model.action_head = torch.nn.Module()
    model.vit_engine = _engine(monkeypatch, tmp_path)
    model.action_head.dit_engine = _engine(monkeypatch, tmp_path)

    module.close_executorch_programs(SimpleNamespace(model=model))

    assert model.vit_engine._method is None
    assert model.action_head.dit_engine._method is None


class _DoublingRuntime:
    def load_program(self, path, data_path=None):
        return self

    def load_method(self, name):
        return self

    def execute(self, inputs):
        self.inputs = inputs
        return [inputs[0] * 2]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="inputs must be CUDA tensors")
@pytest.mark.parametrize("mask_vl_padding", [True, False])
def test_a_tensorrt_forward_function_runs_an_executorch_program(
    monkeypatch, tmp_path, mask_vl_padding
):
    module = _forward_module(monkeypatch)
    from trt.trt_model_forward import _refine_vl_features

    runtime = _DoublingRuntime()
    monkeypatch.setattr(module, "_runtime", runtime)
    inputs = [{"name": "hidden_states", "shape": [1, 3, 4], "dtype": "float16"}]
    if mask_vl_padding:
        inputs.append({"name": "attention_mask", "shape": [1, 3], "dtype": "bool"})
    io = {"file": "vl.pte", "inputs": inputs, "outputs": [{"name": "output", "dtype": "float16"}]}
    engine = module.ExecuTorchEngine(str(tmp_path), io)
    action_head = SimpleNamespace(vlln=torch.nn.Identity(), mask_vl_padding=mask_vl_padding)
    mask = torch.tensor([[1, 1, 0]], device="cuda")
    backbone_output = {
        "backbone_features": torch.ones(1, 3, 4, device="cuda"),
        "backbone_attention_mask": mask,
    }

    out = _refine_vl_features(action_head, backbone_output, engine, cache_outputs=True)

    assert out.dtype == torch.float16 and torch.equal(out, torch.full_like(out, 2))
    assert [t.dtype for t in runtime.inputs] == [torch.float16] + [torch.bool] * mask_vl_padding
    if mask_vl_padding:
        assert torch.equal(runtime.inputs[1], mask.bool())


def test_output_dir_guards(tmp_path):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
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

    # A checkpoint without VL self-attention exports these 6, and may export them again.
    previous = tmp_path / "previous"
    previous.mkdir()
    io = {name: {} for name in _SIX}
    (previous / "export_metadata.json").write_text(json.dumps({"executorch_io": io}))
    export._check_output_dir(str(previous), everything)
    export._check_output_dir(str(previous), list(_SIX))
    with pytest.raises(ValueError, match=r"\['action_decoder', 'action_encoder', 'llm', 'state_"):
        export._check_output_dir(str(previous), ["dit"])


def _stub_main(monkeypatch, export, policy):
    """Stub the runtime packages and the policy-loading steps around ``main``."""
    for name in (
        "executorch",
        "executorch.backends",
        "executorch.backends.cuda",
        "executorch.backends.cuda.cuda_partitioner",
        "executorch.exir",
        "torch_tensorrt",
        "torch_tensorrt_executorch_runtime",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    loaded = []
    monkeypatch.setattr(export, "Gr00tPolicy", lambda **kwargs: loaded.append(1) or policy)
    monkeypatch.setattr(export, "LeRobotEpisodeLoader", lambda **kwargs: None)
    monkeypatch.setattr(export, "capture_component_inputs", lambda *args: ("dit", "vit", "llm"))
    monkeypatch.setattr(
        export,
        "build_export_metadata",
        lambda *args, **kwargs: {"vl_seq_len": 8, "llm_seq_len": 8},
    )
    monkeypatch.setattr(export, "build_vit_for_export", lambda *args: (None, None, None, None))
    monkeypatch.setattr(export, "_restore_vision_attention", lambda *args: None)
    return loaded


def _export_args(export, output_dir, **kwargs):
    return export.ExecuTorchExportConfig(
        model_path="model",
        dataset_path="data",
        embodiment_tag=export.EmbodimentTag.LIBERO_PANDA,
        output_dir=str(output_dir),
        **kwargs,
    )


def test_main_removes_the_old_metadata_before_any_component_exports(monkeypatch, tmp_path):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    action_head = SimpleNamespace(config=None, vl_self_attention=torch.nn.Linear(4, 4))
    policy = SimpleNamespace(
        get_modality_config=dict, model=SimpleNamespace(action_head=action_head)
    )
    _stub_main(monkeypatch, export, policy)
    metadata = tmp_path / "export_metadata.json"
    metadata.write_text(json.dumps({"executorch_io": {}}))
    seen = []

    def failing_export(*args, **kwargs):
        seen.append(metadata.exists())
        raise RuntimeError("export failed")

    monkeypatch.setattr(export, "export_component", failing_export)
    with pytest.raises(RuntimeError, match="export failed"):
        export.main(_export_args(export, tmp_path))
    assert seen == [False] and not metadata.exists()


@pytest.mark.parametrize(
    "missing",
    [
        "executorch.backends.cuda.cuda_partitioner",
        "executorch.exir",
        "torch_tensorrt",
        "torch_tensorrt_executorch_runtime",
    ],
)
def test_main_fails_in_the_wrong_environment_before_loading_the_policy(
    monkeypatch, tmp_path, missing
):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    loaded = _stub_main(monkeypatch, export, None)
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError):
        export.main(_export_args(export, tmp_path))
    assert not loaded


def test_main_refuses_to_export_nothing(monkeypatch, tmp_path):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    action_head = SimpleNamespace(config=None, vl_self_attention=torch.nn.Identity())
    policy = SimpleNamespace(
        get_modality_config=dict, model=SimpleNamespace(action_head=action_head)
    )
    _stub_main(monkeypatch, export, policy)
    with pytest.raises(ValueError, match="applies to this checkpoint"):
        export.main(_export_args(export, tmp_path, components=("vl_self_attention",)))
    assert not (tmp_path / "export_metadata.json").exists()


def test_main_refuses_a_folder_whose_vl_self_attention_it_would_not_replace(monkeypatch, tmp_path):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    action_head = SimpleNamespace(config=None, vl_self_attention=torch.nn.Identity())
    policy = SimpleNamespace(
        get_modality_config=dict, model=SimpleNamespace(action_head=action_head)
    )
    _stub_main(monkeypatch, export, policy)
    metadata = tmp_path / "export_metadata.json"
    io = {name: {} for name in _SIX + ("vl_self_attention",)}
    metadata.write_text(json.dumps({"executorch_io": io}))
    with pytest.raises(ValueError, match=r"\['vl_self_attention'\] from an export"):
        export.main(_export_args(export, tmp_path))
    assert metadata.exists()


@pytest.mark.parametrize(
    "everything",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="main builds CUDA example inputs"
                ),
            ],
        ),
    ],
)
def test_main_writes_metadata_the_loader_reads(monkeypatch, tmp_path, everything):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    module = _forward_module(monkeypatch)
    config = SimpleNamespace(
        backbone_embedding_dim=4,
        max_state_dim=2,
        state_history_length=1,
        action_horizon=3,
        max_action_dim=2,
        hidden_size=4,
    )
    action_head = SimpleNamespace(
        config=config,
        mask_vl_padding=False,
        vl_self_attention=torch.nn.Linear(4, 4),
        state_encoder=torch.nn.Module(),
        action_encoder=torch.nn.Module(),
        model=torch.nn.Module(),
        action_decoder=torch.nn.Module(),
    )
    policy = SimpleNamespace(
        get_modality_config=dict, model=SimpleNamespace(action_head=action_head)
    )
    loaded = _stub_main(monkeypatch, export, policy)
    monkeypatch.setattr(export, "build_llm_for_export", lambda *args: None)
    monkeypatch.setattr(
        export,
        "llm_export_inputs",
        lambda *args: (["inputs_embeds"], [torch.ones(1, 8)], {"inputs_embeds": {1: "seq"}}),
    )
    monkeypatch.setattr(
        export,
        "dit_export_inputs",
        lambda *args: (["sa_embs", "vl_embs", "timestep"], [torch.ones(1)] * 3),
    )
    monkeypatch.setattr(export, "_patch_dit_attn_for_export", lambda model: [])
    monkeypatch.setattr(export, "_unpatch_dit_attn", lambda handles: None)
    exported = {}

    def recording_export(name, module, inputs, input_names, output_names, output_dir, **kwargs):
        Path(output_dir, f"{name}.pte").touch()
        exported[name] = {"file": f"{name}.pte", "data_file": None, "outputs": output_names}
        return exported[name]

    monkeypatch.setattr(export, "export_component", recording_export)

    onnx = tmp_path / "onnx"
    onnx.mkdir()
    (onnx / "vit.onnx").touch()
    with pytest.raises(ValueError, match="TensorRT engines or ONNX"):
        export.main(_export_args(export, onnx))
    assert not loaded

    components = get_args(export.Component) if everything else ("vit", "llm")
    out = tmp_path / "out"
    export.main(_export_args(export, out, components=components))

    metadata = json.loads((out / "export_metadata.json").read_text())
    assert sorted(metadata["executorch_io"].values(), key=lambda entry: entry["file"]) == sorted(
        exported.values(), key=lambda entry: entry["file"]
    )
    if everything:
        assert set(module.executorch_components(str(out), metadata)) == set(components)
    else:
        with pytest.raises(ValueError) as refused:
            module.executorch_components(str(out), metadata)
        assert "lacks ['state_encoder', 'action_encoder', 'dit', 'action_decoder', 'vl" in str(
            refused.value
        )


def test_programs_keep_their_tensors_on_the_gpu(monkeypatch):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    stub("executorch.exir", ExecutorchBackendConfig=SimpleNamespace)
    stub("executorch.exir.passes.memory_planning_pass", MemoryPlanningPass=SimpleNamespace)
    stub("executorch.exir.passes.propagate_device_pass", PropagateDeviceConfig=SimpleNamespace)
    backend = SimpleNamespace(
        generate_method_name_compile_spec=lambda method: ("method", method),
        generate_data_name_compile_spec=lambda data: ("data", data),
    )
    stub("executorch.backends.cuda.cuda_backend", CudaBackend=backend)
    stub("executorch.backends.cuda.cuda_partitioner", CudaPartitioner=list)

    config = export._gpu_memory_config()
    devices = config.propagate_device_config
    assert devices.skip_h2d_for_method_inputs and devices.skip_d2h_for_method_outputs
    assert config.enable_non_cpu_memory_planning
    planning = config.memory_planning_pass
    assert planning.alloc_graph_input is False and planning.alloc_graph_output is False
    assert export._cuda_partitioner("dit_bf16") == [("method", "forward"), ("data", "dit_bf16")]


class _MaskedAdd(torch.nn.Module):
    def forward(self, x, y, *masks):
        out = x + y
        for mask in masks:
            out = out * mask
        return out


@pytest.mark.parametrize("num_masks", [0, 2])
def test_dynamic_shapes_follow_the_bound_masks(num_masks):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)

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


class _FakeTorchTensorRT(types.ModuleType):
    """Records what the exporter hands Torch-TensorRT, and writes the files ``save`` would."""

    def __init__(self, written):
        super().__init__("torch_tensorrt")
        self.Input = lambda **spec: spec
        self.dynamo = SimpleNamespace(compile=self._compile)
        self.written = written

    def _compile(self, program, **options):
        self.compiled = options
        return "compiled"

    def save(self, module, path, **kwargs):
        self.saved = kwargs
        Path(path).touch()
        for name in self.written:
            (Path(path).parent / name).touch()


class _TwoOutputs(torch.nn.Module):
    def forward(self, x, ids):
        return x * 2, ids + 1


def _export_component(
    monkeypatch, output_dir, written=(), output_names=("doubled", "shifted"), **options
):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    trt = _FakeTorchTensorRT(written)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", trt)
    monkeypatch.setattr(export, "_cuda_partitioner", lambda name: f"partitioner for {name}")
    monkeypatch.setattr(export, "_gpu_memory_config", lambda: "gpu memory config")
    inputs = (torch.ones(1, 8), torch.ones(1, 8, dtype=torch.int64))
    io = export.export_component(
        "part", _TwoOutputs(), inputs, ("x", "ids"), output_names, str(output_dir), **options
    )
    return io, trt


@pytest.mark.parametrize("written", [(), ("part.ptd",)])
def test_export_component_records_what_the_runtime_binds(monkeypatch, tmp_path, written):
    for stale in ("part.pte", "part.ptd", "part.old.ptd", "part_other.pte", "other.ptd"):
        (tmp_path / stale).touch()

    io, _ = _export_component(monkeypatch, tmp_path, written)

    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        ("part.pte", "part_other.pte", "other.ptd") + written
    )
    assert io["file"] == "part.pte"
    assert io["data_file"] == (written[0] if written else None)
    assert io["inputs"] == [
        {"name": "x", "shape": [1, 8], "dtype": "float32"},
        {"name": "ids", "shape": [1, 8], "dtype": "int64"},
    ]
    assert io["outputs"] == [
        {"name": "doubled", "dtype": "float32"},
        {"name": "shifted", "dtype": "int64"},
    ]


def test_export_component_refuses_what_the_runtime_could_not_bind(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="more than one data file"):
        _export_component(monkeypatch, tmp_path, ("part.ptd", "part.extra.ptd"))
    with pytest.raises(RuntimeError, match="returns 2 tensors"):
        _export_component(monkeypatch, tmp_path, output_names=("a", "b", "c"))


@pytest.mark.parametrize("dynamic", [False, True])
def test_export_component_hands_torch_tensorrt_the_runtime_settings(monkeypatch, tmp_path, dynamic):
    axes = {"dynamic_axes": {"x": {1: "seq"}, "ids": {1: "seq"}}, "opt_len": 8}

    _, trt = _export_component(monkeypatch, tmp_path, **(axes if dynamic else {}))

    assert trt.compiled["fallback_pytorch_plugin_ops"] and trt.compiled["min_block_size"] == 1
    first = trt.compiled["arg_inputs"][0]
    assert first["max_shape"] == [1, 72] if dynamic else torch.equal(first, torch.ones(1, 8))
    assert trt.saved["output_format"] == "executorch"
    assert trt.saved["use_cuda_graphs"] is not dynamic
    assert (trt.saved["dynamic_shapes"] is None) is not dynamic
    assert trt.saved["partitioners"] == ["partitioner for part"]
    assert trt.saved["backend_config"] == "gpu memory config"


@pytest.mark.parametrize(("opt_len", "max_len"), [(8, 72), (100, 200)])
def test_dynamic_dims_use_the_engine_builder_profile(monkeypatch, opt_len, max_len):
    export = pytest.importorskip("trt.export_executorch_n1d7", exc_type=ModuleNotFoundError)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", _FakeTorchTensorRT(()))
    inputs = (torch.ones(1, opt_len, 4), torch.ones(1, opt_len), torch.ones(3))
    axes = {"embeds": {1: "seq"}, "mask": {1: "seq"}}

    shapes, trt_inputs = export._dynamic_dims(("embeds", "mask", "step"), inputs, axes, opt_len)

    assert shapes[0][1] is shapes[1][1] and shapes[2] is None
    assert trt_inputs[0] == {
        "min_shape": [1, 2, 4],
        "opt_shape": [1, opt_len, 4],
        "max_shape": [1, max_len, 4],
        "dtype": torch.float32,
    }
    assert trt_inputs[1]["min_shape"] == [1, 2] and trt_inputs[1]["max_shape"] == [1, max_len]
    assert trt_inputs[2] == {"shape": [3], "dtype": torch.float32}
