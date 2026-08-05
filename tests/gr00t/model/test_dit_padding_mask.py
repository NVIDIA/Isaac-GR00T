# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.model.modules.dit import AlternateVLDiT, DiT
import pytest
import torch


def _make_model(model_cls):
    kwargs = dict(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        norm_type="ada_norm",
        max_num_positional_embeddings=16,
        final_dropout=False,
        positional_embeddings="sinusoidal",
        interleave_self_attention=True,
        cross_attention_dim=8,
    )
    if model_cls is AlternateVLDiT:
        kwargs["attend_text_every_n_blocks"] = 2
    model = model_cls(**kwargs)
    model.eval()
    return model


def _forward(model, hidden_states, hidden_attention_mask, encoder_hidden_states):
    kwargs = dict(
        hidden_states=hidden_states,
        hidden_attention_mask=hidden_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        timestep=torch.tensor([3]),
    )
    if isinstance(model, AlternateVLDiT):
        kwargs.update(
            image_mask=torch.tensor([[True, True, False, False, False]]),
            backbone_attention_mask=torch.ones(1, 5, dtype=torch.bool),
        )
    return model(**kwargs)


@pytest.mark.parametrize("model_cls", [DiT, AlternateVLDiT])
def test_masked_padding_matches_compact_sequence(model_cls):
    torch.manual_seed(7)
    model = _make_model(model_cls)
    valid_hidden = torch.randn(1, 4, 8)
    padded_hidden = torch.cat((valid_hidden, torch.randn(1, 2, 8) * 1_000), dim=1)
    encoder_hidden = torch.randn(1, 5, 8)

    compact_output = _forward(
        model,
        valid_hidden,
        torch.ones(1, 4, dtype=torch.bool),
        encoder_hidden,
    )
    padded_output = _forward(
        model,
        padded_hidden,
        torch.tensor([[True, True, True, True, False, False]]),
        encoder_hidden,
    )

    torch.testing.assert_close(compact_output, padded_output[:, :4], atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(padded_output[:, 4:]) == 0


@pytest.mark.parametrize("model_cls", [DiT, AlternateVLDiT])
def test_masked_padding_has_zero_input_gradient(model_cls):
    torch.manual_seed(11)
    model = _make_model(model_cls)
    model.train()
    hidden_states = torch.randn(1, 6, 8, requires_grad=True)
    hidden_attention_mask = torch.tensor([[True, True, True, True, False, False]])
    encoder_hidden = torch.randn(1, 5, 8)

    output = _forward(model, hidden_states, hidden_attention_mask, encoder_hidden)
    output[:, :4].square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.count_nonzero(hidden_states.grad[:, 4:]) == 0


def test_hidden_attention_mask_shape_is_validated():
    model = _make_model(AlternateVLDiT)
    with pytest.raises(ValueError, match="hidden_attention_mask must have shape"):
        _forward(
            model,
            torch.randn(1, 6, 8),
            torch.ones(1, 5, dtype=torch.bool),
            torch.randn(1, 5, 8),
        )
