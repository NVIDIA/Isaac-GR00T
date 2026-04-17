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

"""
Test Gr00tN1d7 model forward pass and action generation with dummy data.

These tests construct a minimal Gr00tN1d7 model (with mocked backbone) and
verify that forward() computes a scalar loss and get_action() produces
action predictions of the expected shape.
"""

from unittest.mock import MagicMock, patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _make_small_config(**overrides) -> Gr00tN1d7Config:
    """Return a minimal config for fast instantiation."""
    defaults = dict(
        model_name="nvidia/Cosmos-Reason2-2B",
        backbone_model_type="qwen",
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        select_layer=1,
        reproject_vision=False,
        use_flash_attention=False,
        load_bf16=False,
        tune_top_llm_layers=0,
        backbone_trainable_params_fp32=False,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _make_mock_backbone(config, seq_len=8):
    """Return a mock backbone that produces correctly-shaped outputs."""
    backbone = MagicMock()

    def fake_forward(vl_input):
        B = 1
        # Try to infer batch size from input
        for v in vl_input.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                B = v.shape[0]
                break
        device = next(
            (v.device for v in vl_input.values() if isinstance(v, torch.Tensor)),
            torch.device("cpu"),
        )
        dtype = next(
            (
                v.dtype
                for v in vl_input.values()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            ),
            torch.float32,
        )
        return BatchFeature(
            data={
                "backbone_features": torch.randn(
                    B, seq_len, config.backbone_embedding_dim, device=device, dtype=dtype
                ),
                "backbone_attention_mask": torch.ones(B, seq_len, device=device, dtype=torch.long),
                "image_mask": torch.ones(B, seq_len, device=device, dtype=torch.bool),
            }
        )

    backbone.side_effect = fake_forward
    backbone.prepare_input = lambda x: BatchFeature(data=x)
    return backbone


@pytest.fixture
def small_model():
    """Build a Gr00tN1d7 with mocked backbone (no GPU/download required)."""
    config = _make_small_config()

    with patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls") as mock_get_cls:
        mock_get_cls.return_value = lambda **kwargs: _make_mock_backbone(config)
        with patch("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor"):
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

            model = Gr00tN1d7(config)

    model.eval()
    return model, config


def _make_dummy_inputs(config, batch_size=2):
    """Create dummy input tensors matching the model's expected format."""
    return {
        "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
        "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
    }


class TestGr00tN1d7Forward:
    """Test model forward pass produces valid loss."""

    def test_forward_returns_loss(self, small_model):
        model, config = small_model
        inputs = _make_dummy_inputs(config)
        output = model.forward(inputs)
        assert "loss" in output
        assert output["loss"].dim() == 0, "loss should be scalar"
        assert torch.isfinite(output["loss"]), "loss should be finite"

    def test_forward_returns_action_loss_and_mask(self, small_model):
        model, config = small_model
        inputs = _make_dummy_inputs(config)
        output = model.forward(inputs)
        assert "action_loss" in output
        assert "action_mask" in output
        assert output["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_loss_requires_grad(self, small_model):
        model, config = small_model
        model.train()
        inputs = _make_dummy_inputs(config)
        output = model.forward(inputs)
        assert output["loss"].requires_grad

    def test_forward_different_batch_sizes(self, small_model):
        model, config = small_model
        for bs in [1, 4]:
            inputs = _make_dummy_inputs(config, batch_size=bs)
            output = model.forward(inputs)
            assert output["loss"].dim() == 0


class TestGr00tN1d7GetAction:
    """Test model action generation."""

    def test_get_action_shape(self, small_model):
        model, config = small_model
        inputs = _make_dummy_inputs(config, batch_size=1)
        del inputs["action"]  # get_action uses diffusion denoising, not ground-truth
        output = model.get_action(inputs)
        assert "action_pred" in output
        assert output["action_pred"].shape == (1, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, small_model):
        model, config = small_model
        inputs = _make_dummy_inputs(config, batch_size=1)
        del inputs["action"]
        output = model.get_action(inputs)
        assert not output["action_pred"].requires_grad


class TestGr00tN1d7Config:
    """Test config creation and serialization."""

    def test_default_config(self):
        config = Gr00tN1d7Config()
        assert config.model_type == "Gr00tN1d7"
        assert config.max_state_dim == 132
        assert config.action_horizon == 40

    def test_custom_config(self):
        config = Gr00tN1d7Config(max_state_dim=10, action_horizon=8)
        assert config.max_state_dim == 10
        assert config.action_horizon == 8

    def test_to_filtered_dict(self):
        config = Gr00tN1d7Config()
        d = config.to_filtered_dict(exclude_augment=True)
        assert "random_rotation_angle" not in d
        assert "hidden_size" in d

    def test_to_filtered_json(self):
        config = Gr00tN1d7Config()
        j = config.to_filtered_json()
        assert isinstance(j, str)
        import json

        parsed = json.loads(j)
        assert parsed["model_type"] == "Gr00tN1d7"
