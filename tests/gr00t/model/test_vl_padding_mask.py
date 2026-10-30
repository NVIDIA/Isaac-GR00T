# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for padding isolation in the VL fusion transformer."""

import copy
from types import MethodType

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.dit import SelfAttentionTransformer
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _fusion_config():
    return dict(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
    )


def _head_config(**overrides):
    config = dict(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        max_num_embodiments=2,
        max_seq_len=16,
        state_history_length=1,
        num_inference_timesteps=2,
        state_dropout_prob=0.0,
        use_vlln=True,
        tune_vlln=True,
        tune_projector=True,
        tune_diffusion_model=True,
        use_alternate_vl_dit=True,
        strict_vl_padding_mask=True,
        attend_text_every_n_blocks=2,
        vl_self_attention_cfg=_fusion_config(),
        diffusion_model_cfg=dict(
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=4,
            output_dim=32,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
            norm_type="ada_norm",
            interleave_self_attention=True,
        ),
    )
    config.update(overrides)
    return Gr00tN1d7Config(**config)


def _inputs():
    torch.manual_seed(17)
    features = torch.randn(3, 8, 32)
    # Right padding, left padding, and an unpadded sample in the same batch.
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1], [1] * 8],
        dtype=torch.bool,
    )
    changed = features.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 20
    return features, changed, mask


def _backbone(features, mask):
    # Ensure both image and text tokens are present for AlternateVLDiT.
    image_mask = torch.zeros_like(mask)
    image_mask[:, 4] = True
    return BatchFeature(
        data={
            "backbone_features": features.clone(),
            "backbone_attention_mask": mask.clone(),
            "image_mask": image_mask,
        }
    )


def _actions():
    return BatchFeature(
        data={
            "state": torch.randn(3, 1, 7),
            "action": torch.randn(3, 4, 7),
            "action_mask": torch.ones(3, 4, 7),
            "embodiment_id": torch.zeros(3, dtype=torch.long),
        }
    )


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.long, torch.float32])
def test_fusion_ignores_changed_padding_at_every_layer(mask_dtype):
    model = SelfAttentionTransformer(**_fusion_config()).eval()
    features, changed, mask = _inputs()
    _, layers = model(features, True, attention_mask=mask.to(mask_dtype))
    _, changed_layers = model(changed, True, attention_mask=mask.to(mask_dtype))
    for original, perturbed in zip(layers[1:], changed_layers[1:], strict=True):
        torch.testing.assert_close(original[mask], perturbed[mask])


def test_fusion_padding_has_no_gradient_from_valid_tokens():
    model = SelfAttentionTransformer(**_fusion_config()).eval()
    features, _, mask = _inputs()
    features.requires_grad_()
    model(features, attention_mask=mask)[mask].square().mean().backward()
    assert torch.count_nonzero(features.grad[~mask]) == 0
    assert torch.count_nonzero(features.grad[mask]) > 0


def test_fusion_without_padding_preserves_existing_output():
    model = SelfAttentionTransformer(**_fusion_config()).eval()
    features, _, _ = _inputs()
    torch.testing.assert_close(
        model(features),
        model(features, attention_mask=torch.ones(features.shape[:2], dtype=torch.bool)),
    )
    # The pre-existing positional return_all_hidden_states argument still works.
    output, layers = model(features, True)
    torch.testing.assert_close(output, layers[-1])


@pytest.mark.parametrize("alternate", [False, True])
def test_action_head_passes_padding_mask_to_fusion(alternate):
    head = Gr00tN1d7ActionHead(_head_config(use_alternate_vl_dit=alternate)).eval()
    features, changed, mask = _inputs()
    original = head.process_backbone_output(_backbone(features, mask))
    perturbed = head.process_backbone_output(_backbone(changed, mask))
    torch.testing.assert_close(original.backbone_features[mask], perturbed.backbone_features[mask])
    torch.testing.assert_close(original.backbone_attention_mask, mask)


def test_action_head_without_fusion_accepts_padding_mask():
    head = Gr00tN1d7ActionHead(_head_config(vl_self_attention_cfg=None)).eval()
    features, _, mask = _inputs()
    output = head.process_backbone_output(_backbone(features, mask))
    torch.testing.assert_close(output.backbone_features, head.vlln(features))


@pytest.mark.parametrize("alternate", [False, True])
def test_padded_training_loss_and_gradients_are_isolated(alternate):
    head = Gr00tN1d7ActionHead(_head_config(use_alternate_vl_dit=alternate)).train()
    features, changed, mask = _inputs()
    features.requires_grad_()
    actions = _actions()
    torch.manual_seed(31)
    output = head(_backbone(features, mask), actions)
    torch.manual_seed(31)
    perturbed = head(_backbone(changed, mask), actions)
    torch.testing.assert_close(output["loss"], perturbed["loss"])
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert torch.count_nonzero(features.grad[~mask]) == 0
    assert torch.count_nonzero(features.grad[mask]) > 0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    before = head.vl_self_attention.transformer_blocks[0].attn1.to_q.weight.detach().clone()
    torch.optim.AdamW(head.parameters(), lr=1e-3).step()
    assert not torch.equal(before, head.vl_self_attention.transformer_blocks[0].attn1.to_q.weight)


@pytest.mark.parametrize("alternate", [False, True])
def test_padded_action_predictions_are_isolated(alternate):
    head = Gr00tN1d7ActionHead(_head_config(use_alternate_vl_dit=alternate)).eval()
    features, changed, mask = _inputs()
    actions = _actions()
    del actions["action"]
    torch.manual_seed(31)
    original = head.get_action(_backbone(features, mask), actions)
    torch.manual_seed(31)
    perturbed = head.get_action(_backbone(changed, mask), actions)
    torch.testing.assert_close(original.action_pred, perturbed.action_pred)


def _short_and_mixed_backbones(total_length, left_padding):
    """Keep a short sample's valid tokens fixed while adding a longer companion."""
    torch.manual_seed(43)
    short_features = torch.randn(1, 5, 32)
    short_mask = torch.ones(1, 5, dtype=torch.bool)
    short_image_mask = torch.zeros_like(short_mask)
    short_image_mask[:, 1] = True
    mixed_features = torch.randn(2, total_length, 32)
    mixed_mask = torch.ones(2, total_length, dtype=torch.bool)
    mixed_mask[0] = False
    start = total_length - 5 if left_padding else 0
    mixed_features[0, start : start + 5] = short_features[0]
    mixed_mask[0, start : start + 5] = True
    mixed_image_mask = torch.zeros_like(mixed_mask)
    mixed_image_mask[0, start + 1] = True
    mixed_image_mask[1, 1] = True
    short = BatchFeature(
        data={
            "backbone_features": short_features,
            "backbone_attention_mask": short_mask,
            "image_mask": short_image_mask,
        }
    )
    mixed = BatchFeature(
        data={
            "backbone_features": mixed_features,
            "backbone_attention_mask": mixed_mask,
            "image_mask": mixed_image_mask,
        }
    )
    return short, mixed


def _fix_diffusion_randomness(monkeypatch, head):
    # Changing batch shape can change RNG draws. Hold per-sample noise and time
    # fixed so differences measure padding rather than stochastic diffusion.
    noise = torch.linspace(-1, 1, 28).reshape(1, 4, 7)

    def fixed_noise(size, *, device, dtype):
        assert tuple(size[1:]) == (4, 7)
        return noise.to(device=device, dtype=dtype).repeat(size[0], 1, 1)

    monkeypatch.setattr(torch, "randn", fixed_noise)
    monkeypatch.setattr(
        head,
        "sample_time",
        lambda batch_size, device, dtype: torch.full(
            (batch_size,), 0.5, device=device, dtype=dtype
        ),
    )


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("left_padding", [False, True])
@pytest.mark.parametrize("total_length", [8, 12])
def test_short_sample_matches_mixed_length_batch(
    monkeypatch, alternate, left_padding, total_length
):
    head = Gr00tN1d7ActionHead(_head_config(use_alternate_vl_dit=alternate)).eval()
    short, mixed = _short_and_mixed_backbones(total_length, left_padding)
    short_actions = BatchFeature(
        data={key: value[:1].clone() for key, value in _actions().items() if key != "action"}
    )
    mixed_actions = BatchFeature(
        data={
            key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in short_actions.items()
        }
    )
    _fix_diffusion_randomness(monkeypatch, head)
    single = head.get_action(copy.deepcopy(short), short_actions)
    batched = head.get_action(copy.deepcopy(mixed), mixed_actions)
    valid = mixed.backbone_attention_mask[0]
    torch.testing.assert_close(
        single.backbone_features[0], batched.backbone_features[0, valid], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(single.action_pred[0], batched.action_pred[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("left_padding", [False, True])
@pytest.mark.parametrize("total_length", [8, 12])
def test_short_sample_training_loss_and_gradients_match_with_extra_padding(
    monkeypatch, alternate, left_padding, total_length
):
    head = Gr00tN1d7ActionHead(_head_config(use_alternate_vl_dit=alternate)).train()
    short, mixed = _short_and_mixed_backbones(total_length, left_padding)
    # Compare this sample's loss rather than a mean over different tasks.
    padded = BatchFeature(data={key: value[:1].clone() for key, value in mixed.items()})
    actions = BatchFeature(data={key: value[:1].clone() for key, value in _actions().items()})
    _fix_diffusion_randomness(monkeypatch, head)

    def loss_and_gradients(backbone):
        head.zero_grad(set_to_none=True)
        features = backbone.backbone_features.requires_grad_()
        loss = head(backbone, copy.deepcopy(actions))["loss"]
        loss.backward()
        return (
            loss.detach(),
            features.grad.detach().clone(),
            {
                name: p.grad.detach().clone()
                for name, p in head.named_parameters()
                if p.grad is not None
            },
        )

    short_loss, short_grad, short_parameters = loss_and_gradients(short)
    padded_loss, padded_grad, padded_parameters = loss_and_gradients(padded)
    valid = mixed.backbone_attention_mask[:1]
    torch.testing.assert_close(short_loss, padded_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(short_grad[0], padded_grad[valid], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(padded_grad[~valid]) == 0
    assert short_parameters.keys() == padded_parameters.keys()
    for name, gradient in short_parameters.items():
        torch.testing.assert_close(gradient, padded_parameters[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("strict", [None, False])
def test_legacy_checkpoint_preserves_features_loss_and_actions(alternate, strict):
    config_dict = _head_config(use_alternate_vl_dit=alternate).to_dict()
    if strict is None:
        # Released checkpoints predate this field entirely.
        del config_dict["strict_vl_padding_mask"]
    else:
        config_dict["strict_vl_padding_mask"] = strict
    head = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_dict(config_dict)).eval()
    assert head.config.strict_vl_padding_mask is False
    reference = copy.deepcopy(head)

    def legacy_process(self, backbone):
        features = self.vlln(backbone["backbone_features"])
        # Independent pre-patch fusion path: no mask in any block.
        features = features.contiguous()
        for block in self.vl_self_attention.transformer_blocks:
            features = block(features)
        backbone["backbone_features"] = features
        return backbone

    reference.process_backbone_output = MethodType(legacy_process, reference)
    features, _, mask = _inputs()
    torch.testing.assert_close(
        head.process_backbone_output(_backbone(features, mask)).backbone_features,
        reference.process_backbone_output(_backbone(features, mask)).backbone_features,
        rtol=0,
        atol=0,
    )
    actions = _actions()
    torch.manual_seed(31)
    actual_loss = head(_backbone(features, mask), actions)["loss"]
    torch.manual_seed(31)
    expected_loss = reference(_backbone(features, mask), actions)["loss"]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    del actions["action"]
    torch.manual_seed(31)
    actual = head.get_action(_backbone(features, mask), actions).action_pred
    torch.manual_seed(31)
    expected = reference.get_action(_backbone(features, mask), actions).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("strict", [False, True])
def test_checkpoint_roundtrip_preserves_strict_setting_and_weights(tmp_path, strict):
    head = Gr00tN1d7ActionHead(_head_config(strict_vl_padding_mask=strict)).eval()
    head.config.save_pretrained(tmp_path)
    restored = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(tmp_path)).eval()
    restored.load_state_dict(head.state_dict(), strict=True)
    assert restored.config.strict_vl_padding_mask is strict
    other = Gr00tN1d7ActionHead(_head_config(strict_vl_padding_mask=not strict))
    other.load_state_dict(head.state_dict(), strict=True)
    features, _, mask = _inputs()
    torch.testing.assert_close(
        head.process_backbone_output(_backbone(features, mask)).backbone_features,
        restored.process_backbone_output(_backbone(features, mask)).backbone_features,
        rtol=0,
        atol=0,
    )


def test_fusion_rejects_wrong_mask_shape():
    model = SelfAttentionTransformer(**_fusion_config()).eval()
    features, _, mask = _inputs()
    with pytest.raises(ValueError, match="attention_mask must have shape"):
        model(features, attention_mask=mask.unsqueeze(1))
