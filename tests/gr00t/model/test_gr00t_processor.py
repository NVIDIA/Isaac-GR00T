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
Test Gr00tN1d7Processor: state/action processing, VLM input generation, decode_action.

Uses the fixture configs in tests/fixtures/processor_config/ with a mocked VLM
processor (no model download needed).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
import numpy as np
from PIL import Image
import pytest


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"


@pytest.fixture
def processor():
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    mock_vlm = MagicMock()
    mock_vlm.apply_chat_template.return_value = "mock text"
    mock_vlm.tokenizer.padding_side = "left"

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        proc = Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
    proc.eval()
    return proc


@pytest.fixture
def proc_config():
    with open(FIXTURE_DIR / "processor_config.json") as f:
        return json.load(f)["processor_kwargs"]


def _make_step_data(proc_config) -> VLAStepData:
    """Create synthetic VLAStepData matching the fixture config."""
    import json as _json

    mc = proc_config["modality_configs"][EMBODIMENT]
    state_keys = mc["state"]["modality_keys"]
    action_keys = mc["action"]["modality_keys"]
    video_keys = mc["video"]["modality_keys"]
    action_horizon = len(mc["action"]["delta_indices"])

    # Load statistics to get correct dimensions per key
    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = _json.load(f)

    images = {k: [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)] for k in video_keys}
    states = {
        k: np.random.randn(1, len(statistics[EMBODIMENT]["state"][k]["min"])).astype(np.float32)
        for k in state_keys
    }
    actions = {
        k: np.random.randn(action_horizon, len(statistics[EMBODIMENT]["action"][k]["min"])).astype(
            np.float32
        )
        for k in action_keys
    }

    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text="pick up the apple",
        embodiment=EmbodimentTag(EMBODIMENT),
    )


class TestProcessorCall:
    """Test __call__ with VLAStepData messages."""

    def test_call_returns_expected_keys(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert "state" in result
        assert "action" in result
        assert "embodiment_id" in result
        assert "action_mask" in result

    def test_state_padded_to_max_dim(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        state = result["state"]
        max_dim = proc_config["max_state_dim"]
        assert state.shape[-1] == max_dim, f"Expected state dim {max_dim}, got {state.shape[-1]}"

    def test_action_padded_to_max_dim(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        action = result["action"]
        max_dim = proc_config["max_action_dim"]
        assert action.shape[-1] == max_dim, f"Expected action dim {max_dim}, got {action.shape[-1]}"

    def test_action_mask_matches_action(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert result["action_mask"].shape == result["action"].shape

    def test_embodiment_id_is_integer(self, processor, proc_config):
        step_data = _make_step_data(proc_config)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        result = processor(messages)
        assert isinstance(result["embodiment_id"], (int, np.integer))


class TestProcessorVLMInputs:
    """Test VLM input generation."""

    def test_get_vlm_inputs(self, processor):
        image_keys = processor.modality_configs[EMBODIMENT]["video"].modality_keys
        mock_images = {
            k: [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
            for k in image_keys
        }
        vlm_inputs = processor._get_vlm_inputs(
            image_keys=image_keys,
            images=mock_images,
            image_transform=processor.eval_image_transform,
            language="pick up the apple",
            masks=None,
        )
        assert "vlm_content" in vlm_inputs


class TestProcessorDecodeAction:
    """Test action denormalization."""

    def test_decode_action_returns_all_keys(self, processor, proc_config):
        mc = proc_config["modality_configs"][EMBODIMENT]
        action_keys = mc["action"]["modality_keys"]
        # Load statistics so decode_action can denormalize
        with open(FIXTURE_DIR / "statistics.json") as f:
            stats = json.load(f)
        processor.set_statistics(stats)

        action_dim = processor.state_action_processor.get_action_dim(EMBODIMENT)
        action_horizon = len(mc["action"]["delta_indices"])
        dummy_action = np.random.randn(action_horizon, action_dim).astype(np.float32)
        result = processor.decode_action(dummy_action, EmbodimentTag(EMBODIMENT))
        assert set(result.keys()) == set(action_keys)


class TestProcessorStatistics:
    """Test statistics management."""

    def test_set_statistics(self, processor):
        with open(FIXTURE_DIR / "statistics.json") as f:
            stats = json.load(f)
        processor.set_statistics(stats)
        assert EMBODIMENT in processor.statistics

    def test_train_eval_mode(self, processor):
        processor.train()
        assert processor.training is True
        assert processor.state_action_processor.training is True
        processor.eval()
        assert processor.training is False
        assert processor.state_action_processor.training is False
