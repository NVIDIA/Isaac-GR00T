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

"""CPU-only checks for the N1.7 TRT deployment scripts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
EXPORT_SCRIPT = ROOT / "scripts" / "deployment" / "export_onnx_n1d7.py"
BUILD_PIPELINE_SCRIPT = ROOT / "scripts" / "deployment" / "build_trt_pipeline.py"
MODES_SCRIPT = ROOT / "gr00t" / "deployment" / "modes.py"


def _is_torch_onnx_export(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "export"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "onnx"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "torch"
    )


def test_all_onnx_exports_use_legacy_exporter_explicitly() -> None:
    """N1.7 TRT export relies on dynamic_axes; keep legacy export explicit."""

    tree = ast.parse(EXPORT_SCRIPT.read_text())
    missing = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_torch_onnx_export(node):
            continue
        dynamo_kw = next((kw for kw in node.keywords if kw.arg == "dynamo"), None)
        has_legacy_exporter = (
            dynamo_kw is not None
            and isinstance(dynamo_kw.value, ast.Constant)
            and dynamo_kw.value.value is False
        )
        if not has_legacy_exporter:
            missing.append(node.lineno)

    assert not missing, f"torch.onnx.export calls must pass dynamo=False: lines {missing}"


def test_dit_only_pipeline_uses_dit_only_verifier() -> None:
    """A single DiT engine cannot be verified through the four-engine action-head path."""

    from gr00t.deployment.modes import PIPELINE_STAGE_MODES, ExportMode, VerifyMode

    # build_trt_pipeline imports this table as _MODE_MAP from the SoT module.
    _build, verify_mode, _bench = PIPELINE_STAGE_MODES[ExportMode.dit_only]
    assert verify_mode == VerifyMode.dit_only


def test_verify_mode_accepts_dit_only_mode() -> None:
    """The unified pipeline can route dit_only export into verify_n1d7_trt.py."""

    tree = ast.parse(MODES_SCRIPT.read_text())
    verify_modes = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "VerifyMode":
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                verify_modes.add(stmt.targets[0].id)

    assert "dit_only" in verify_modes


def test_dit_export_wrapper_routes_hidden_mask_without_other_optional_masks(tmp_path) -> None:
    """A regular DiT capture may contain only the action-token mask."""

    try:
        from scripts.deployment.export_onnx_n1d7 import _DiTExportWrapper
    except (ImportError, OSError) as exc:
        pytest.skip(f"export_onnx_n1d7 not importable in this env: {exc}")

    class DummyDiT(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            image_mask=None,
            backbone_attention_mask=None,
            hidden_attention_mask=None,
        ):
            assert image_mask is None
            assert backbone_attention_mask is None
            assert hidden_attention_mask is not None
            return hidden_states * hidden_attention_mask.unsqueeze(-1)

    wrapper = _DiTExportWrapper(DummyDiT(), ["hidden_attention_mask"])
    hidden_states = torch.ones(1, 3, 2)
    hidden_attention_mask = torch.tensor([[True, False, True]])
    output = wrapper(
        hidden_states,
        torch.ones(1, 2, 2),
        torch.ones(1, dtype=torch.long),
        hidden_attention_mask,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]]),
    )

    onnx = pytest.importorskip("onnx")
    output_path = tmp_path / "dit_hidden_mask.onnx"
    torch.onnx.export(
        wrapper,
        (
            hidden_states,
            torch.ones(1, 2, 2),
            torch.ones(1, dtype=torch.long),
            hidden_attention_mask,
        ),
        output_path,
        input_names=["sa_embs", "vl_embs", "timestep", "hidden_attention_mask"],
        output_names=["output"],
        opset_version=19,
        dynamo=False,
    )
    graph_inputs = [value.name for value in onnx.load(output_path).graph.input]
    assert graph_inputs == ["sa_embs", "hidden_attention_mask"]
