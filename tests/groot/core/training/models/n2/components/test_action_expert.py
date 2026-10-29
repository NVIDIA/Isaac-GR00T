# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("albumentations")

from groot.core.training.models.n2.components import transformer_cosmos
from groot.core.training.models.n2.components.transformer_cosmos import (
    CosmosTransformer3DModel,
    CosmosTransformerConfig,
)


def _tiny_transformer(*, action_expert: bool) -> CosmosTransformer3DModel:
    config = CosmosTransformerConfig(
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        mlp_ratio=2,
        text_embed_dim=6,
        adaln_lora_dim=4,
        max_n_cameras=1,
        extra_pos_embed_type=None,
        action_expert=action_expert,
    )
    return CosmosTransformer3DModel(config).eval()


def _sdpa(q, k, v, *, flatten=False):
    del flatten
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def test_action_expert_starts_dense_and_preserves_video_path(monkeypatch):
    monkeypatch.setattr(transformer_cosmos, "flash_attn_func_v23", _sdpa)
    torch.manual_seed(7)
    dense = _tiny_transformer(action_expert=False)
    torch.manual_seed(7)
    expert = _tiny_transformer(action_expert=True)

    latents = torch.randn(1, 5, 8)
    timesteps = torch.rand(1, 5)
    rope = (torch.ones(5, 4), torch.zeros(5, 4))
    context = torch.randn(1, 3, 6)
    modality_ids = torch.tensor([[0, 1, 2, 3, 3]])
    args = (latents, timesteps, rope, context, modality_ids)

    with torch.no_grad():
        dense_out = dense(*args)
        initial_expert_out = expert(*args)
    torch.testing.assert_close(initial_expert_out, dense_out)

    # Change only the action FFN. With one block, the video rows never consume
    # the changed action rows and therefore remain byte-for-byte on the dense path.
    with torch.no_grad():
        expert.transformer_blocks[0].action_ff.net[2].weight.add_(0.1)
        tuned_out = expert(*args)
    torch.testing.assert_close(tuned_out[:, :2], dense_out[:, :2])
    assert not torch.equal(tuned_out[:, 2:], dense_out[:, 2:])


def test_freeze_base_for_action_expert_leaves_only_expert_trainable():
    model = _tiny_transformer(action_expert=True)
    model.freeze_base_for_action_expert()

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("action_" in name for name in trainable)
    assert any(name.startswith("action_norm_out.") for name in trainable)
    assert any(".action_attn1." in name for name in trainable)
