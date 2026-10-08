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

"""Tests for direct-bf16 checkpoint loading in Gr00tPolicy.

Gr00tPolicy passes ``torch_dtype=torch.bfloat16`` to ``AutoModel.from_pretrained``
so the checkpoint (stored in bf16) is materialized directly in bf16 instead of
being upcast to fp32 and cast back down, halving peak host memory during load.

These tests pin two guarantees:

1. Wiring: the policy requests bf16 from ``from_pretrained`` and still applies
   the final ``.to(device, dtype)`` normalization.
2. Parity: direct-bf16 loading is bit-identical to the legacy
   fp32-load-then-cast path — for every weight and for the produced actions —
   because bf16 -> fp32 -> bf16 is a lossless round-trip.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
import pytest
import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


EMBODIMENT = EmbodimentTag.NEW_EMBODIMENT.value


# ---------------------------------------------------------------------------
# TinyPolicyLoad — a minimal HF model with real weights so save/load round-trips
# exercise the actual transformers dtype machinery (unlike a MagicMock model).
# ---------------------------------------------------------------------------


class TinyPolicyLoadConfig(PretrainedConfig):
    model_type = "TinyPolicyLoad"

    def __init__(
        self,
        hidden_dim: int = 16,
        action_horizon: int = 4,
        max_action_dim: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.max_action_dim = max_action_dim


class TinyPolicyLoadModel(PreTrainedModel):
    """Deterministic model: actions are a pure function of weights and input."""

    config_class = TinyPolicyLoadConfig

    def __init__(self, config: TinyPolicyLoadConfig):
        super().__init__(config)
        self.norm = torch.nn.LayerNorm(config.hidden_dim)
        self.proj = torch.nn.Linear(
            config.hidden_dim, config.action_horizon * config.max_action_dim
        )
        self.register_buffer("scale", torch.ones(config.hidden_dim), persistent=True)

    def get_action(self, state: torch.Tensor, **kwargs) -> dict:
        batch_size = state.shape[0]
        hidden = self.norm(state.to(self.norm.weight.dtype) * self.scale)
        out = self.proj(hidden)
        return {
            "action_pred": out.reshape(
                batch_size, self.config.action_horizon, self.config.max_action_dim
            )
        }


AutoConfig.register("TinyPolicyLoad", TinyPolicyLoadConfig)
AutoModel.register(TinyPolicyLoadConfig, TinyPolicyLoadModel)


@pytest.fixture(scope="module")
def bf16_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint saved in bf16, mirroring production GR00T checkpoints."""
    torch.manual_seed(0)
    model = TinyPolicyLoadModel(TinyPolicyLoadConfig())
    model.to(torch.bfloat16)
    ckpt_dir = tmp_path_factory.mktemp("tiny_bf16_ckpt")
    model.save_pretrained(ckpt_dir)
    return ckpt_dir


def _legacy_load(ckpt_dir: Path) -> TinyPolicyLoadModel:
    """Reproduce the pre-change load path: fp32 materialization, then cast."""
    model = AutoModel.from_pretrained(ckpt_dir)
    # Documents the memory cost the direct-bf16 path removes: without
    # torch_dtype, transformers materializes the bf16 checkpoint in fp32.
    assert next(model.parameters()).dtype == torch.float32
    return model.to(dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Parity at the AutoModel level
# ---------------------------------------------------------------------------


class TestDirectBf16Load:
    def test_from_pretrained_materializes_bf16(self, bf16_checkpoint):
        """With torch_dtype, no fp32 copy of the weights is ever created."""
        model = AutoModel.from_pretrained(bf16_checkpoint, torch_dtype=torch.bfloat16)
        for name, param in model.named_parameters():
            assert param.dtype == torch.bfloat16, f"{name} not bf16"

    def test_state_dict_bitwise_identical_to_legacy(self, bf16_checkpoint):
        """Every weight and buffer matches the legacy path bit-for-bit."""
        direct = AutoModel.from_pretrained(bf16_checkpoint, torch_dtype=torch.bfloat16)
        legacy = _legacy_load(bf16_checkpoint)

        direct_sd, legacy_sd = direct.state_dict(), legacy.state_dict()
        assert direct_sd.keys() == legacy_sd.keys()
        for key in direct_sd:
            assert direct_sd[key].dtype == legacy_sd[key].dtype, f"dtype mismatch for {key}"
            assert torch.equal(direct_sd[key], legacy_sd[key]), f"value mismatch for {key}"

    def test_actions_bitwise_identical_to_legacy(self, bf16_checkpoint):
        """The user-facing guarantee: identical actions for identical inputs."""
        direct = AutoModel.from_pretrained(bf16_checkpoint, torch_dtype=torch.bfloat16)
        legacy = _legacy_load(bf16_checkpoint)

        torch.manual_seed(42)
        state = torch.randn(2, 16, dtype=torch.bfloat16)
        with torch.inference_mode():
            direct_action = direct.get_action(state=state)["action_pred"]
            legacy_action = legacy.get_action(state=state)["action_pred"]
        assert torch.equal(direct_action, legacy_action)


# ---------------------------------------------------------------------------
# Wiring and end state at the Gr00tPolicy level
# ---------------------------------------------------------------------------


def _make_mock_processor() -> MagicMock:
    processor = MagicMock()
    modality_configs = {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
            "action": ModalityConfig(delta_indices=list(range(4)), modality_keys=["joints"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }
    }
    processor.get_modality_configs.return_value = modality_configs
    return processor


class TestGr00tPolicyBf16Load:
    def test_policy_requests_bf16_from_pretrained(self, bf16_checkpoint):
        """The policy must ask from_pretrained for bf16 directly (no fp32 peak)."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        with (
            patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
            patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        ):
            MockAutoModel.from_pretrained.return_value = mock_model
            MockAutoProcessor.from_pretrained.return_value = _make_mock_processor()

            from gr00t.policy.gr00t_policy import Gr00tPolicy

            Gr00tPolicy(
                embodiment_tag=EMBODIMENT,
                model_path=str(bf16_checkpoint),
                device="cpu",
            )

        _, kwargs = MockAutoModel.from_pretrained.call_args
        assert kwargs.get("torch_dtype") == torch.bfloat16
        # The final device/dtype normalization must still happen.
        mock_model.to.assert_called_once_with(device="cpu", dtype=torch.bfloat16)
        mock_model.eval.assert_called_once()

    def test_policy_model_bitwise_matches_legacy_load(self, bf16_checkpoint):
        """Loading through the real policy yields the exact legacy end state."""
        with patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor:
            MockAutoProcessor.from_pretrained.return_value = _make_mock_processor()

            from gr00t.policy.gr00t_policy import Gr00tPolicy

            policy = Gr00tPolicy(
                embodiment_tag=EMBODIMENT,
                model_path=str(bf16_checkpoint),
                device="cpu",
            )

        assert not policy.model.training
        legacy = _legacy_load(bf16_checkpoint)
        policy_sd, legacy_sd = policy.model.state_dict(), legacy.state_dict()
        assert policy_sd.keys() == legacy_sd.keys()
        for key in policy_sd:
            assert policy_sd[key].dtype == legacy_sd[key].dtype, f"dtype mismatch for {key}"
            assert torch.equal(policy_sd[key], legacy_sd[key]), f"value mismatch for {key}"
