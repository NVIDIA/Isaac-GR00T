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

"""CPU regressions for dtype boundaries when PyTorch modules are retained."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SOURCE = Path(__file__).resolve().parents[3] / "scripts/deployment/trt/trt_model_forward.py"


@pytest.fixture
def forward_functions():
    # Load the actual forward definitions without importing the CUDA/TRT loader.
    names = {
        "_qwen3_vit_and_scatter",
        "qwen3_backbone_tensorrt_forward",
        "qwen3_backbone_full_trt_forward",
    }
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {
        "torch": torch,
        "BatchFeature": lambda data: data,
        "assert_grid_thw_matches": lambda *args, **kwargs: None,
        "_assert_supports_trt_padding_strip": lambda mask: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class ViTEngine:
    out_meta = [("image_embeds", (), torch.float32)]

    def dtype_of(self, name):
        return torch.float32

    def set_runtime_tensor_shape(self, *args):
        pass

    def __call__(self, pixels):
        return {
            "image_embeds": torch.ones(2, 4, dtype=torch.float32),
            "deepstack_features": torch.ones(1, 2, 4, dtype=torch.float32),
        }


class LLMEngine:
    out_meta = [("embeddings", (), torch.float16)]
    in_meta = [("inputs_embeds", (), torch.float16)]

    def dtype_of(self, name):
        return torch.float16

    def set_runtime_tensor_shape(self, *args):
        pass

    def __call__(self, embeds, *args, **kwargs):
        assert embeds.dtype == torch.float16
        return {"embeddings": embeds}


def make_backbone():
    projection = nn.Linear(4, 4).to(torch.bfloat16)

    def language_model(**kwargs):
        features = projection(kwargs["inputs_embeds"])
        for deep in kwargs["deepstack_visual_embeds"] or []:
            assert deep.dtype == torch.bfloat16
            features = features + deep
        return SimpleNamespace(last_hidden_state=features)

    inner = SimpleNamespace(
        language_model=language_model,
        get_placeholder_mask=lambda ids, **kwargs: (torch.ones(1, 2, 4, dtype=torch.bool), None),
        get_rope_index=lambda *args, **kwargs: (torch.zeros(3, 1, 2, dtype=torch.long), None),
    )
    return SimpleNamespace(
        model=SimpleNamespace(model=inner),
        vit_engine=ViTEngine(),
        _embedding_layer=nn.Embedding(10, 4).to(torch.bfloat16),
        _image_token_id=1,
        set_frozen_modules_to_eval_mode=lambda: None,
    )


def inputs():
    return {
        "pixel_values": torch.ones(2, 4),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
        "input_ids": torch.tensor([[1, 1]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }


def test_fp32_vit_preserves_retained_bf16_llm(forward_functions):
    result = forward_functions.qwen3_backbone_tensorrt_forward(make_backbone(), inputs())
    assert result["backbone_features"].dtype == torch.bfloat16


@pytest.mark.parametrize("retained_action_head", [True, False])
def test_quantized_llm_respects_downstream_boundary(forward_functions, retained_action_head):
    backbone = make_backbone()
    backbone.llm_engine = LLMEngine()
    if retained_action_head:
        backbone._trt_action_head_dtype = torch.bfloat16
    result = forward_functions.qwen3_backbone_full_trt_forward(backbone, inputs())
    features = result["backbone_features"]
    if retained_action_head:
        assert features.dtype == torch.bfloat16
        nn.LayerNorm(4).to(torch.bfloat16)(features)
    else:
        assert features.dtype == torch.float16
