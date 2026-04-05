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
Test Gr00tPolicy: observation validation and inference pipeline.

Uses mocked model and processor to avoid downloading checkpoints.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from gr00t.data.types import ModalityConfig
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"

VIDEO_KEYS = ["observation.images.rgb.head_256_256", "observation.images.rgb.left_wrist_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


@pytest.fixture
def policy():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16

    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": torch.randn(1, 16, 7)})
    )

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.state_action_processor = MagicMock()
    mock_processor.action_dim = {EMBODIMENT: 7}
    mock_processor.max_action_dim = 128
    mock_processor.max_action_horizon = 50
    mock_processor.eval = MagicMock()
    mock_processor.training = False
    mock_processor.collator = MagicMock()

    def fake_process_observation(observation, embodiment_tag):
        return BatchFeature(
            data={
                "state": torch.randn(1, 1, 128),
                "action_mask": torch.ones(1, 16, 128),
                "embodiment_id": torch.zeros(1, dtype=torch.long),
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 256, 256),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }
        )

    mock_processor.process_observation = MagicMock(side_effect=fake_process_observation)

    def fake_decode_action(action, embodiment_tag, state=None):
        return {k: np.zeros((1, 16, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    # Patch both AutoModel and AutoProcessor, and also the processor_config.json check
    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        p = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
        )
    return p


def _make_observation(batch_size=1):
    return {
        "video": {
            k: np.random.randint(0, 255, (batch_size, 1, 256, 256, 3), dtype=np.uint8)
            for k in VIDEO_KEYS
        },
        "state": {
            k: np.random.randn(batch_size, 1, 1).astype(np.float32)
            for k in STATE_KEYS[:-1]  # all except gripper
        }
        | {"gripper": np.random.randn(batch_size, 1, 2).astype(np.float32)},
        "language": {
            LANGUAGE_KEY: [["pick up the apple"]] * batch_size,
        },
    }


class TestGr00tPolicyInit:
    def test_policy_has_model_and_processor(self, policy):
        assert policy.model is not None
        assert policy.processor is not None

    def test_policy_embodiment_tag(self, policy):
        assert policy.embodiment_tag is not None


class TestGr00tPolicyCheckObservation:
    def test_valid_observation_passes(self, policy):
        obs = _make_observation()
        policy.check_observation(obs)

    def test_missing_video_key_raises(self, policy):
        obs = _make_observation()
        del obs["video"][VIDEO_KEYS[0]]
        with pytest.raises((AssertionError, KeyError)):
            policy.check_observation(obs)

    def test_wrong_video_dtype_raises(self, policy):
        obs = _make_observation()
        obs["video"][VIDEO_KEYS[0]] = obs["video"][VIDEO_KEYS[0]].astype(np.float32)
        with pytest.raises(AssertionError):
            policy.check_observation(obs)


class TestGr00tPolicyGetAction:
    def test_get_action_returns_tuple(self, policy):
        obs = _make_observation()
        result = policy.get_action(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_action_returns_dict(self, policy):
        obs = _make_observation()
        action, info = policy.get_action(obs)
        assert isinstance(action, dict)
        assert isinstance(info, dict)
