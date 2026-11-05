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

"""CPU checks that TensorRT paths keep VL padding handled like the PyTorch model.

The TensorRT runtime modules import ``tensorrt`` at module level, so the call sites
under test are compiled out of their source and run against stub engines.
"""

import ast
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.deployment.modes import EXPORT_MODE_COMPONENTS, ExportMode, components_to_build
from gr00t.deployment.vl_padding import assert_plain_dit_sees_no_vl_padding
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "scripts" / "deployment"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

PADDED = torch.tensor([[False, True, True], [True, True, True]])
UNPADDED = torch.ones(2, 3, dtype=torch.bool)
PLAIN_DIT = SimpleNamespace(use_alternate_vl_dit=False, strict_vl_padding_mask=True)
LEGACY_PLAIN_DIT = SimpleNamespace(use_alternate_vl_dit=False, strict_vl_padding_mask=False)
ALTERNATE_VL_DIT = SimpleNamespace(use_alternate_vl_dit=True, strict_vl_padding_mask=True)


class _Reached(Exception):
    """Raised by the first engine touched after the VL self-attention step."""


class _StopEngine:
    @property
    def in_meta(self):
        raise _Reached

    def dtype_of(self, name):
        raise _Reached


def _compile(script, name, scope, parent=None):
    """Compile one function from ``script``, optionally nested in ``parent``."""
    path = _DEPLOY_DIR / script
    tree = ast.parse(path.read_text())
    root = tree
    if parent is not None:
        root = next(
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == parent
        )
    function = next(
        node for node in ast.walk(root) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_refine_vl_features", "_validate_vl_attention_engine"}
    ]
    scope = {
        "torch": torch,
        "BatchFeature": BatchFeature,
        "assert_plain_dit_sees_no_vl_padding": assert_plain_dit_sees_no_vl_padding,
        **scope,
    }
    exec(compile(ast.Module(body=[*helpers, function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def _head(config, mask_vl_padding, seen_masks):
    def vl_self_attention(hidden_states, attention_mask=None):
        seen_masks.append(attention_mask)
        return hidden_states

    def process_backbone_output(backbone_output):
        features = backbone_output["backbone_features"]
        if mask_vl_padding:
            features = vl_self_attention(
                features, attention_mask=backbone_output["backbone_attention_mask"]
            )
        else:
            features = vl_self_attention(features)
        backbone_output["backbone_features"] = features
        return backbone_output

    return SimpleNamespace(
        config=config,
        vlln=nn.Identity(),
        vl_self_attention=vl_self_attention,
        process_backbone_output=process_backbone_output,
        mask_vl_padding=mask_vl_padding,
        vl_sa_engine=None,
        dit_engine=_StopEngine(),
    )


def _backbone_output(mask):
    return BatchFeature(
        data={"backbone_features": torch.randn(*mask.shape, 4), "backbone_attention_mask": mask}
    )


def _run_action_head(head, backbone_output):
    forward = _compile("trt/trt_model_forward.py", "action_head_tensorrt_forward", {})
    with pytest.raises(_Reached):
        forward(head, backbone_output, SimpleNamespace(embodiment_id=None))


def _run_optimized_runner(head, backbone_output):
    execute = _compile(
        "trt/trt_model_forward.py", "_execute", {}, parent="OptimizedActionHeadRunner"
    )
    runner = SimpleNamespace(action_head=head, vl_sa_engine=None, cross_kv_engine=_StopEngine())
    state = SimpleNamespace(
        backbone_features=backbone_output.backbone_features,
        backbone_attention_mask=backbone_output.backbone_attention_mask,
    )
    with pytest.raises(_Reached):
        execute(runner, state)


@pytest.mark.parametrize("run", [_run_action_head, _run_optimized_runner])
@pytest.mark.parametrize("mask_vl_padding", [True, False])
def test_pytorch_vl_self_attention_fallback_gets_the_mask(run, mask_vl_padding):
    """Without a VL engine, TensorRT paths run the head's own masking PyTorch path."""
    seen_masks = []
    run(_head(ALTERNATE_VL_DIT, mask_vl_padding, seen_masks), _backbone_output(PADDED))
    assert seen_masks == [PADDED if mask_vl_padding else None]


@pytest.mark.parametrize("strict", [True, False])
def test_setup_loads_a_vl_engine_only_if_the_export_built_it(tmp_path, strict, caplog):
    """An export that kept VL attention in PyTorch makes an engine beside it stale."""
    engine_path = _compile(
        "trt/trt_model_forward.py",
        "vl_self_attention_engine_path",
        {"os": os, "logger": logging.getLogger(__name__)},
    )
    head = SimpleNamespace(mask_vl_padding=strict)
    engine_dir = str(tmp_path)
    assert engine_path(head, engine_dir, None) is None

    engine = tmp_path / "vl_self_attention.engine"
    engine.touch()
    assert engine_path(head, engine_dir, {"vl_self_attention_exported": False}) is None
    assert str(engine) in caplog.text
    for metadata in (None, {}, {"vl_self_attention_exported": True}):
        assert engine_path(head, engine_dir, metadata) == str(engine)


def _call_action_head(head, backbone_output):
    _compile("trt/trt_model_forward.py", "action_head_tensorrt_forward", {})(
        head, backbone_output, None
    )


def _call_dit_only(head, backbone_output):
    dit_only = _compile(
        "trt/trt_model_forward.py",
        "dit_only_get_action_with_features",
        {"action_head": head},
        parent="_setup_dit_only",
    )
    dit_only(backbone_output.backbone_features, None, None, backbone_output)


def _call_standalone_dit(head, backbone_output):
    policy = SimpleNamespace(model=SimpleNamespace(action_head=head))
    trt_forward = _compile(
        "standalone_inference_script.py",
        "trt_forward",
        {"policy": policy},
        parent="replace_dit_with_tensorrt",
    )
    trt_forward(
        None,
        backbone_output.backbone_features,
        None,
        encoder_attention_mask=backbone_output.backbone_attention_mask,
    )


@pytest.mark.parametrize("call", [_call_action_head, _call_dit_only, _call_standalone_dit])
def test_plain_dit_tensorrt_paths_refuse_padded_batches(call):
    with pytest.raises(NotImplementedError, match="no VL mask input"):
        call(_head(PLAIN_DIT, False, []), _backbone_output(PADDED))


def test_plain_dit_guard_refuses_only_a_padded_batch():
    with pytest.raises(NotImplementedError, match="no VL mask input"):
        assert_plain_dit_sees_no_vl_padding(PLAIN_DIT, PADDED, "test path")
    assert_plain_dit_sees_no_vl_padding(PLAIN_DIT, UNPADDED, "test path")
    assert_plain_dit_sees_no_vl_padding(ALTERNATE_VL_DIT, PADDED, "test path")
    assert_plain_dit_sees_no_vl_padding(LEGACY_PLAIN_DIT, PADDED, "test path")


def test_only_an_identity_vl_self_attention_is_not_exported(tmp_path):
    """A masking vl_self_attention is exported with a mask input; Identity has no graph."""
    export = pytest.importorskip("trt.export_onnx_n1d7", exc_type=ImportError)
    for strict in (True, False):
        assert export.exports_vl_self_attention(
            SimpleNamespace(vl_self_attention=nn.Linear(2, 2), mask_vl_padding=strict)
        )
    identity = SimpleNamespace(vl_self_attention=nn.Identity(), mask_vl_padding=False)
    assert not export.exports_vl_self_attention(identity)
    policy = SimpleNamespace(model=SimpleNamespace(action_head=identity))
    assert export.export_vl_self_attention_to_onnx(policy, str(tmp_path), vl_seq_len=4) is None
    assert not any(tmp_path.iterdir())


def test_build_skips_vl_self_attention_the_export_kept_in_pytorch():
    full = EXPORT_MODE_COMPONENTS[ExportMode.full_pipeline]
    kept = {"vl_self_attention_exported": False}
    assert "VL Self-Attention" in full
    assert components_to_build(None, kept) == full - {"VL Self-Attention"}
    assert components_to_build(None, {"vl_self_attention_exported": True}) is None
    assert components_to_build(None, {}) is None
    assert components_to_build(None, None) is None


@pytest.mark.parametrize(
    "script,function,helper",
    [
        ("trt/trt_model_forward.py", "_setup_n17_full_pipeline", "vl_self_attention_engine_path"),
        ("trt/build_tensorrt_engine.py", "build_full_pipeline", "components_to_build"),
    ],
)
def test_tensorrt_call_sites_use_the_tested_helpers(script, function, helper):
    """These call sites import tensorrt, so check that they reach the helpers tested above."""
    tree = ast.parse((_DEPLOY_DIR / script).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    called = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert helper in called
