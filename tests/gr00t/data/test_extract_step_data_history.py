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

"""Regression tests for ``extract_step_data`` with history (negative) ``delta_indices``.

``DataFrame.iloc`` accepts negative positions, so a history offset such as the
DROID video ``delta_indices=[-15, 0]`` used to silently read frames from the END
of the episode for the first 15 steps whenever ``allow_padding=False`` (the
training default).
"""

from unittest.mock import MagicMock, patch

from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.types import EmbodimentTag, ModalityConfig
import numpy as np
import pandas as pd
import pytest


EPISODE_LENGTH = 20
HISTORY = 15
TAG = EmbodimentTag.NEW_EMBODIMENT


def _episode(length: int = EPISODE_LENGTH) -> pd.DataFrame:
    # Encode the row index in every value so a wrong frame is unmistakable.
    rows = np.arange(length, dtype=np.float32)
    return pd.DataFrame(
        {
            "video.cam": [np.full((2, 2, 3), r, dtype=np.uint8) for r in rows],
            "state.x": [np.array([r, r]) for r in rows],
            "action.x": [np.array([r]) for r in rows],
            "language.task": ["task"] * length,
        }
    )


def _configs(
    action_delta_indices: list[int] | None = None, history: int = HISTORY
) -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[-history, 0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=[-history, 0], modality_keys=["x"]),
        "action": ModalityConfig(
            delta_indices=action_delta_indices or list(range(4)), modality_keys=["x"]
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


@pytest.mark.parametrize("step_index", [0, 7, HISTORY - 1])
def test_history_before_episode_start_clamps_to_first_frame(step_index):
    step = extract_step_data(_episode(), step_index, _configs(), TAG, allow_padding=False)
    history, current = step.states["x"]
    # Before the fix this was row EPISODE_LENGTH + step_index - HISTORY (episode tail).
    assert history[0] == 0.0
    assert current[0] == step_index
    # List-type modalities (video / mask) go through the same index list.
    history_frame, current_frame = step.images["cam"]
    assert history_frame[0, 0, 0] == 0
    assert current_frame[0, 0, 0] == step_index


def test_history_inside_episode_is_untouched():
    step_index = HISTORY + 1  # action window [16, 20) still fits in the episode
    step = extract_step_data(_episode(), step_index, _configs(), TAG, allow_padding=False)
    history, current = step.states["x"]
    assert history[0] == step_index - HISTORY
    assert current[0] == step_index
    assert step.images["cam"][0][0, 0, 0] == step_index - HISTORY


def test_episode_shorter_than_history_clamps_to_first_frame():
    # Previously raised (pandas out-of-bounds) because -HISTORY < -len(episode).
    step = extract_step_data(_episode(length=5), 0, _configs(), TAG, allow_padding=False)
    assert step.states["x"][0][0] == 0.0


def test_future_index_past_episode_end_raises_without_padding():
    # pandas already raised IndexError here; this pins the explicit message.
    with pytest.raises(IndexError, match="out of range"):
        extract_step_data(_episode(), EPISODE_LENGTH - 1, _configs(), TAG, allow_padding=False)


def test_negative_action_index_raises_without_padding():
    # Actions are future targets; a negative index is a config error, not history.
    with pytest.raises(IndexError, match="action index -1"):
        extract_step_data(
            _episode(), 0, _configs(action_delta_indices=[-1, 0, 1]), TAG, allow_padding=False
        )


def test_allow_padding_still_clamps_both_ends():
    step_index = EPISODE_LENGTH - 1
    step = extract_step_data(_episode(), step_index, _configs(), TAG, allow_padding=True)
    assert step.states["x"][0][0] == step_index - HISTORY
    assert np.all(step.actions["x"][:, 0] == step_index)


def test_sharded_dataset_default_path_pads_history_with_first_frame():
    """End to end through ShardedSingleStepDataset.get_shard with the default allow_padding."""
    episode = _episode()
    with patch(
        "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
    ) as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.episode_lengths = [EPISODE_LENGTH]
        mock_loader.get_episode_length = lambda idx: EPISODE_LENGTH
        mock_loader.__getitem__.return_value = episode
        mock_loader_cls.return_value = mock_loader
        dataset = ShardedSingleStepDataset(
            dataset_path="/fake/dataset",
            embodiment_tag=TAG,
            modality_configs=_configs(),
            shard_size=1024,
            episode_sampling_rate=1.0,
            seed=0,
        )
    dataset.processor = lambda messages: messages[0]["content"]  # identity processor

    datapoints = dataset.get_shard(0)
    scheduled = {int(dp.states["x"][1][0]) for dp in datapoints}
    assert set(range(HISTORY)) <= scheduled, "sharder must still schedule the early steps"
    for dp in datapoints:
        step_index = int(dp.states["x"][1][0])
        assert dp.states["x"][0][0] == max(0, step_index - HISTORY)
