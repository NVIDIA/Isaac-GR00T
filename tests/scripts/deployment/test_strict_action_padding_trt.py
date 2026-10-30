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

"""TensorRT action-head loops must serve a strict_action_padding_mask checkpoint
exactly like the PyTorch action head.

The TensorRT runtime modules import ``tensorrt`` at module level, so the loops
under test are compiled out of their source and run against stand-in engines
that call the PyTorch modules they replace. The stand-in DiT masks padding only
when the loop passes ``hidden_attention_mask``, as the exported engine does.
"""

import ast
from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.deployment.action_padding import (
    HIDDEN_ATTENTION_MASK,
    assert_dit_engine_matches_action_padding,
    strict_action_masks,
)
from gr00t.deployment.vl_padding import assert_plain_dit_sees_no_vl_padding
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "scripts" / "deployment"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

BATCH, HORIZON, ACTION_DIM, REAL_HORIZON, REAL_DIM, VL_LEN = 2, 4, 7, 2, 3, 6

# Keywords Gr00tN1d7ActionHead passes to the DiT it may have swapped for TRT.
DIT_FORWARD_KWARGS = (
    "hidden_states",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "timestep",
    "return_all_hidden_states",
    "image_mask",
    "backbone_attention_mask",
    "hidden_attention_mask",
)


def _head(strict=True, use_alternate_vl_dit=True):
    config = Gr00tN1d7Config(
        backbone_embedding_dim=16,
        hidden_size=16,
        input_embedding_dim=16,
        max_state_dim=5,
        max_action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        strict_action_padding_mask=strict,
        state_history_length=1,
        num_inference_timesteps=3,
        max_num_embodiments=2,
        add_pos_embed=True,
        max_seq_len=8,
        use_alternate_vl_dit=use_alternate_vl_dit,
        attend_text_every_n_blocks=2,
        vl_self_attention_cfg=None,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 16,
            "interleave_self_attention": True,
        },
    )
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(config).eval()
    # The default init is small enough to hide a missing mask within tolerance.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0, 0.3)
    return head


def _inputs():
    torch.manual_seed(3)
    image_mask = torch.zeros(BATCH, VL_LEN, dtype=torch.bool)
    image_mask[:, :2] = True
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(BATCH, VL_LEN, 16),
            "backbone_attention_mask": torch.ones(BATCH, VL_LEN, dtype=torch.bool),
            "image_mask": image_mask,
        }
    )
    action_mask = torch.zeros(BATCH, HORIZON, ACTION_DIM)
    action_mask[:, :REAL_HORIZON, :REAL_DIM] = 1
    action_input = BatchFeature(
        data={
            "state": torch.randn(BATCH, 1, 5),
            "embodiment_id": torch.zeros(BATCH, dtype=torch.long),
            "action_mask": action_mask,
        }
    )
    return backbone_output, action_input


def _pytorch_action(head):
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        return head.get_action(backbone_output, action_input)["action_pred"]


class _Engine:
    """Stand-in TensorRT engine calling the PyTorch module it replaces."""

    def __init__(self, fn, inputs, exact_inputs=False):
        self.fn = fn
        self.in_meta = [[name, None, torch.float32] for name in inputs]
        self.exact_inputs = exact_inputs

    def set_runtime_tensor_shape(self, name, shape):
        # TensorRT fails to set the shape of an input the engine does not have.
        if self.exact_inputs and name not in {item[0] for item in self.in_meta}:
            raise RuntimeError(f"Failed to set runtime shape for {name}")

    def dtype_of(self, name):
        return torch.float32

    def __call__(self, *args, cache_outputs=None, **kwargs):
        if self.exact_inputs:
            kwargs = {k: v for k, v in kwargs.items() if k in {i[0] for i in self.in_meta}}
        return self.fn(*args, **kwargs)


def _dit_engine(head):
    """Takes the inputs export_dit_to_onnx gives this DiT, as the built engine does."""

    def dit(sa_embs, vl_embs, timestep, **masks):
        return {"output": head.model(sa_embs, vl_embs, timestep, **masks)}

    vl_masks = ["image_mask", "backbone_attention_mask"] if head.config.use_alternate_vl_dit else []
    inputs = ["sa_embs", "vl_embs", "timestep", *vl_masks, HIDDEN_ATTENTION_MASK]
    if not head.config.strict_action_padding_mask:
        inputs.remove(HIDDEN_ATTENTION_MASK)
    return _Engine(dit, inputs, exact_inputs=True)


def _compile(script, name, scope, parent=None):
    path = _DEPLOY_DIR / script
    tree = ast.parse(path.read_text())
    root = tree
    if parent is not None:
        root = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == parent
        )
    function = next(n for n in ast.walk(root) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {
        "torch": torch,
        "BatchFeature": BatchFeature,
        "logger": logging.getLogger(__name__),
        "assert_plain_dit_sees_no_vl_padding": assert_plain_dit_sees_no_vl_padding,
        "strict_action_masks": strict_action_masks,
        "HIDDEN_ATTENTION_MASK": HIDDEN_ATTENTION_MASK,
        **scope,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def _action_head_path(head):
    head.state_encoder_engine = _Engine(
        lambda state, ids: {"output": head.state_encoder(state, ids)}, ["state"]
    )
    head.action_encoder_engine = _Engine(
        lambda actions, t, ids: {"output": head.action_encoder(actions, t, ids)}, ["actions"]
    )
    head.dit_engine = _dit_engine(head)
    head.action_decoder_engine = _Engine(
        lambda output, ids: {"output": head.action_decoder(output, ids)}, ["model_output"]
    )
    forward = _compile("trt/trt_model_forward.py", "action_head_tensorrt_forward", {})
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        return forward(head, backbone_output, action_input)["action_pred"]


def _dit_only_path(head):
    head.dit_engine = _dit_engine(head)
    get_action_with_features = _compile(
        "trt/trt_model_forward.py",
        "dit_only_get_action_with_features",
        {"action_head": head},
        parent="_setup_dit_only",
    )
    backbone_output, action_input = _inputs()
    with torch.no_grad():
        features = head._encode_features(backbone_output, action_input)
        torch.manual_seed(11)
        return get_action_with_features(
            features.backbone_features,
            features.state_features,
            action_input.embodiment_id,
            backbone_output,
            action_input,
        )["action_pred"]


def _optimized_runner(head, tmp_path):
    export = pytest.importorskip("trt.export_onnx_n1d7", exc_type=ImportError)
    dit = head.model
    encoder = head.action_encoder
    constants = export.save_action_head_constants(
        SimpleNamespace(model=SimpleNamespace(action_head=head)), dit, str(tmp_path), BATCH
    )

    def encode(actions, time_embedding, ids):
        hidden = torch.cat([encoder.W1(actions, ids), time_embedding], dim=-1)
        hidden = encoder.W2(hidden, ids)
        return {"output": encoder.W3(hidden * torch.sigmoid(hidden), ids)}

    def cross_kv(vl_embs):
        keys, values = export.DiTCrossKV(dit)(vl_embs)
        return {"cross_keys": keys, "cross_values": values}

    cached_dit = export.CachedCrossKVDiT(dit)
    runner = _runner_class().__new__(_runner_class())
    runner.action_head = head
    runner.vl_sa_engine = None
    runner.cross_kv_engine = _Engine(cross_kv, ["vl_embs"])
    runner.state_encoder_engine = _Engine(
        lambda state, ids, **_: {"output": head.state_encoder(state, ids)}, ["state"]
    )
    runner.action_encoder_engine = _Engine(encode, ["actions"])
    runner.dit_engine = _Engine(
        lambda **inputs: {"output": cached_dit(**inputs)}, ["sa_embs", HIDDEN_ATTENTION_MASK]
    )
    runner.action_decoder_engine = _Engine(
        lambda output, ids, **_: {"output": head.action_decoder(output, ids)}, ["model_output"]
    )
    runner.constants = constants
    runner.position_embeddings = head.position_embedding(torch.arange(HORIZON)).unsqueeze(0)
    return runner


def _runner_class():
    """The optimized runner, built without __init__, which loads TensorRT engines."""
    path = _DEPLOY_DIR / "trt" / "trt_model_forward.py"
    tree = ast.parse(path.read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("ActionHeadGraphState", "OptimizedActionHeadRunner")
    ]
    scope = {
        "torch": torch,
        "BatchFeature": BatchFeature,
        "dataclass": dataclass,
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "strict_action_masks": strict_action_masks,
        "HIDDEN_ATTENTION_MASK": HIDDEN_ATTENTION_MASK,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)
    return scope["OptimizedActionHeadRunner"]


def _optimized_action(runner):
    backbone_output, action_input = _inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        state = runner._make_graph_state(backbone_output, action_input)
        return runner._execute(state)


@pytest.mark.parametrize(
    "path,use_alternate_vl_dit",
    [
        ("action_head", True),
        ("dit_only", True),
        ("optimized", True),
        ("action_head", False),
        ("dit_only", False),
    ],
)
def test_tensorrt_loops_match_the_strict_pytorch_action_head(
    path, use_alternate_vl_dit, tmp_path, monkeypatch
):
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    expected = _pytorch_action(_head(use_alternate_vl_dit=use_alternate_vl_dit))
    head = _head(use_alternate_vl_dit=use_alternate_vl_dit)
    if path == "action_head":
        actual = _action_head_path(head)
    elif path == "dit_only":
        actual = _dit_only_path(head)
    else:
        actual = _optimized_action(_optimized_runner(head, tmp_path))

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert not actual[:, REAL_HORIZON:].any() and not actual[..., REAL_DIM:].any()


def test_cuda_graph_mask_buffers_follow_each_call(tmp_path):
    """The captured graph is replayed for every embodiment with the same batch shape,
    so its mask buffers must be refreshed from each call's action_mask."""
    runner = _optimized_runner(_head(), tmp_path)
    backbone_output, action_input = _inputs()
    state = runner._make_graph_state(backbone_output, action_input)

    other = action_input["action_mask"].clone()
    other[:, :3, :5] = 1
    for action_mask in (action_input["action_mask"], other):
        action_input["action_mask"] = action_mask
        runner._copy_inputs(state, backbone_output, action_input)
        masks = strict_action_masks(runner.action_head.config, action_input, state.actions, 1)
        assert torch.equal(state.action_mask, masks.action)
        assert torch.equal(state.action_token_mask, masks.tokens)
        assert torch.equal(state.hidden_attention_mask, masks.hidden)


@pytest.mark.parametrize("use_alternate_vl_dit", [True, False])
def test_exported_dit_takes_the_padding_mask(use_alternate_vl_dit, tmp_path, monkeypatch):
    """Which DiT masks exist depends on the DiT, so the export binds them by name."""
    export = pytest.importorskip("trt.export_onnx_n1d7", exc_type=ImportError)
    onnx = pytest.importorskip("onnx")
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    head = _head(use_alternate_vl_dit=use_alternate_vl_dit)
    capture = export.DiTInputCapture()
    hook = head.model.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)
    backbone_output, action_input = _inputs()
    with torch.no_grad():
        head.get_action(backbone_output, action_input)
    hook.remove()

    names, inputs = export.dit_export_inputs(capture, BATCH, torch.float32, "cpu")
    assert HIDDEN_ATTENTION_MASK in names
    inputs[names.index(HIDDEN_ATTENTION_MASK)] = capture.hidden_attention_mask
    masks = dict(zip(names[3:], inputs[3:]))
    wrapper = export.DiTExportWrapper(head.model, names[3:]).eval()
    with torch.no_grad():
        expected = head.model(*inputs[:3], **masks)
        torch.testing.assert_close(wrapper(*inputs), expected)
        torch.testing.assert_close(wrapper(*inputs[:3], **masks), expected)

    path = tmp_path / "dit.onnx"
    handles = export._patch_dit_attn_for_export(head.model)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                tuple(inputs),
                str(path),
                input_names=names,
                output_names=["output"],
                opset_version=19,
                dynamo=False,
            )
    finally:
        export._unpatch_dit_attn(handles)
    graph = onnx.load(str(path), load_external_data=False).graph
    assert [i.name for i in graph.input] == names


def test_flag_off_tensorrt_loop_still_matches_pytorch(monkeypatch):
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    expected = _pytorch_action(_head(strict=False))
    actual = _action_head_path(_head(strict=False))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_masks_follow_the_embodiment_extent():
    actions = torch.zeros(BATCH, HORIZON, ACTION_DIM)
    _, action_input = _inputs()
    masks = strict_action_masks(
        SimpleNamespace(strict_action_padding_mask=True), action_input, actions, 1
    )
    assert torch.equal(masks.action, action_input["action_mask"])
    assert masks.tokens.squeeze(-1).tolist() == [[True, True, False, False]] * BATCH
    assert masks.hidden.tolist() == [[True, True, True, False, False]] * BATCH

    assert (
        strict_action_masks(SimpleNamespace(strict_action_padding_mask=False), {}, actions, 1)
        is None
    )
    with pytest.raises(ValueError, match="action_mask is required"):
        strict_action_masks(SimpleNamespace(strict_action_padding_mask=True), {}, actions, 1)


@pytest.mark.parametrize("strict", [True, False])
def test_dit_engine_must_match_the_checkpoint_setting(strict):
    config = SimpleNamespace(strict_action_padding_mask=strict)
    matching = {"sa_embs", HIDDEN_ATTENTION_MASK} if strict else {"sa_embs"}
    other = {"sa_embs"} if strict else {"sa_embs", HIDDEN_ATTENTION_MASK}
    assert_dit_engine_matches_action_padding(config, matching, "test path")
    with pytest.raises(ValueError, match="Re-export"):
        assert_dit_engine_matches_action_padding(config, other, "test path")


@pytest.mark.parametrize(
    "function",
    ["_setup_n17_full_pipeline", "_setup_action_head", "_setup_dit_only", "_validate_interfaces"],
)
def test_every_dit_engine_load_is_checked(function):
    """These run only with tensorrt, so check that they reach the tested check."""
    tree = ast.parse((_DEPLOY_DIR / "trt" / "trt_model_forward.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    called = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "assert_dit_engine_matches_action_padding" in called


def test_standalone_trt_dit_binds_every_action_head_keyword(monkeypatch):
    try:
        import standalone_inference_script as mod
    except (ImportError, OSError) as e:  # torch/tensorrt absent on CPU CI
        pytest.skip(f"standalone_inference_script not importable in this env: {e}")

    monkeypatch.setattr(
        mod, "TensorRTDiTWrapper", lambda *a, **k: SimpleNamespace(input_names={"sa_embs"})
    )
    action_head = SimpleNamespace(
        model=SimpleNamespace(),
        config=SimpleNamespace(strict_action_padding_mask=False),
    )
    policy = SimpleNamespace(model=SimpleNamespace(action_head=action_head))

    mod.replace_dit_with_tensorrt(policy, "/does-not-exist")

    sig = inspect.signature(action_head.model.forward)
    sig.bind(**{name: None for name in DIT_FORWARD_KWARGS})


def test_standalone_trt_dit_passes_the_padding_mask_to_the_engine(monkeypatch):
    try:
        import standalone_inference_script as mod
    except (ImportError, OSError) as e:  # torch/tensorrt absent on CPU CI
        pytest.skip(f"standalone_inference_script not importable in this env: {e}")

    received = {}

    class RecordingDiT:
        input_names = {"sa_embs", HIDDEN_ATTENTION_MASK}

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, **kwargs):
            received.update(kwargs)
            return "output"

    monkeypatch.setattr(mod, "TensorRTDiTWrapper", RecordingDiT)
    action_head = SimpleNamespace(
        model=SimpleNamespace(),
        config=SimpleNamespace(strict_action_padding_mask=True, use_alternate_vl_dit=True),
    )
    policy = SimpleNamespace(model=SimpleNamespace(action_head=action_head))
    mod.replace_dit_with_tensorrt(policy, "/does-not-exist")

    mask = torch.tensor([[True, True, False]])
    action_head.model.forward(None, None, None, hidden_attention_mask=mask)
    assert received[HIDDEN_ATTENTION_MASK] is mask
