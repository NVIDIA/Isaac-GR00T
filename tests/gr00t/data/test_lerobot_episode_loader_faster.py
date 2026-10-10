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
Test LeRobotEpisodeLoaderFaster (sparse video loading for shard step indices).

The faster loader decodes only the video frames needed by the requested shard
steps instead of every frame of an episode. Video decoding is mocked here so the
tests run without a video backend; the assertions verify which frame indices
the loader asks the decoder for and how the sparse frames are placed back into
the episode DataFrame.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gr00t.data.dataset.lerobot_episode_loader_faster import LeRobotEpisodeLoaderFaster
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd
import pytest


def _make_fake_loader(
    episode_lengths=(100, 50),
    video_delta_indices=(-15, 0),
):
    """Build a LeRobotEpisodeLoaderFaster with mocked internals (no dataset on disk).

    ``_load_video_data`` is a recording stub: it returns frame objects equal to
    their position in the requested index array, and records the requested
    indices on the returned loader under ``requested_video_indices``.
    """
    loader = LeRobotEpisodeLoaderFaster.__new__(LeRobotEpisodeLoaderFaster)
    loader.episodes_metadata = [
        {"episode_index": i, "length": n} for i, n in enumerate(episode_lengths)
    ]
    loader.modality_configs = {
        "video": ModalityConfig(delta_indices=list(video_delta_indices), modality_keys=["cam"]),
        "mask": ModalityConfig(delta_indices=[0], modality_keys=["seg"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["x"]),
    }

    def fake_parquet(episode_id):
        n = episode_lengths[episode_id]
        return pd.DataFrame({"observation.state": [np.zeros(6)] * n})

    def fake_language(episode_meta, n, lang_key):
        return [f"task_{episode_meta['episode_index']}"] * n

    def fake_video(episode_id, indices):
        loader.requested_video_indices.append(np.asarray(indices))
        return {"cam": np.arange(len(indices))}

    def fake_mask(episode_id, indices):
        return {"seg": np.arange(len(indices))}

    loader.requested_video_indices = []
    loader._load_parquet_data = fake_parquet
    loader.create_language_from_meta = fake_language
    loader._load_video_data = fake_video
    loader._load_mask_data = fake_mask
    return loader


class TestGetVideoIndicesFromSteps:
    @pytest.mark.parametrize(
        ("steps", "delta", "episode_length", "allow_padding", "expected"),
        [
            # In-bounds steps with a negative delta
            (np.array([10, 11, 12]), [-15, 0], 100, False, [-5, -4, -3, 10, 11, 12]),
            # allow_padding clamps out-of-range indices to [0, episode_length - 1]
            (np.array([0, 1]), [-3, 0], 100, True, [0, 1]),
            # Without padding, out-of-range indices are passed through
            (np.array([0, 1]), [-3, 0], 100, False, [-3, -2, 0, 1]),
            # Wide delta range (action horizon) spanning the whole episode
            (np.array([50]), list(range(40)), 100, True, list(range(50, 90))),
            # Duplicates from different steps are deduplicated
            (np.array([5, 6]), [0, 1], 100, False, [5, 6, 7]),
        ],
    )
    def test_index_computation(self, steps, delta, episode_length, allow_padding, expected):
        loader = _make_fake_loader()
        indices = loader._get_video_indices_from_steps(
            steps, delta, episode_length, allow_padding=allow_padding
        )
        assert indices.tolist() == expected


class TestCallSparseLoading:
    def test_decodes_only_requested_frames(self):
        loader = _make_fake_loader()
        steps = np.array([10, 11, 12])
        loader(0, steps, allow_padding=True)

        # The decoder must be asked for exactly the union of step + delta
        # indices (clamped), not the full episode range 0..99.
        assert len(loader.requested_video_indices) == 1
        requested = loader.requested_video_indices[0]
        assert requested.tolist() == [0, 10, 11, 12]

    def test_frames_placed_at_requested_positions_none_elsewhere(self):
        loader = _make_fake_loader()
        df = loader(0, np.array([10, 11, 12]), allow_padding=True)

        assert len(df) == 100
        col = df["video.cam"]
        # Frame objects equal their position in the decoded batch; they must
        # land exactly at the requested frame indices.
        assert col.iloc[0] == 0
        assert col.iloc[10] == 1
        assert col.iloc[11] == 2
        assert col.iloc[12] == 3
        # All other positions must stay empty.
        populated = {0, 10, 11, 12}
        assert col.iloc[[i for i in range(100) if i not in populated]].isna().all()

    def test_mask_still_fully_loaded(self):
        loader = _make_fake_loader()
        df = loader(0, np.array([10, 11, 12]), allow_padding=True)
        assert df["mask.seg"].iloc[0] == 0
        assert df["mask.seg"].iloc[99] == 99

    def test_no_padding_keeps_negative_indices(self):
        loader = _make_fake_loader()
        loader(0, np.array([0, 1]), allow_padding=False)
        requested = loader.requested_video_indices[-1]
        assert requested.tolist() == [-15, -14, 0, 1]

    def test_trims_to_nominal_length(self):
        loader = _make_fake_loader()
        df = loader(1, np.array([5]), allow_padding=True)
        assert len(df) == 50

    def test_out_of_bounds_raises(self):
        loader = _make_fake_loader()
        with pytest.raises(IndexError):
            loader(99, np.array([0]))

    def test_non_video_modalities_loaded_for_full_episode(self):
        loader = _make_fake_loader()
        df = loader(0, np.array([10]), allow_padding=True)
        assert df["language.task"].iloc[0] == "task_0"
        assert df["observation.state"].iloc[0].shape == (6,)


class TestOriginalGetitemPreserved:
    def test_getitem_still_loads_full_episode(self):
        loader = _make_fake_loader()
        df = loader[0]
        assert len(df["video.cam"]) == 100
        assert df["video.cam"].iloc[99] == 99  # fully populated, no None


class TestDatasetWiring:
    """ShardedSingleStepDataset picks the loader class based on the flag."""

    MODALITY_CONFIGS = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=list(range(4)), modality_keys=["x"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        "mask": ModalityConfig(delta_indices=[0], modality_keys=["seg"]),
    }

    def _episode_df(self, n):
        """DataFrame with the columns extract_step_data expects."""
        return pd.DataFrame(
            {
                "state.x": [np.zeros(1) for _ in range(n)],
                "action.x": [np.zeros(1) for _ in range(n)],
                "video.cam": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)],
                "mask.seg": [np.zeros((8, 8), dtype=np.uint8) for _ in range(n)],
                "language.task": [f"task_{i}" for i in range(n)],
            }
        )

    def _make_dataset(self, use_faster, faster_loader):
        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
        from gr00t.data.embodiment_tags import EmbodimentTag

        with patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
        ) as MockLoader:
            mock_loader = MagicMock()
            mock_loader.episode_lengths = [50]
            mock_loader.get_episode_length = lambda idx: 50
            MockLoader.return_value = mock_loader

            if use_faster:
                with patch(
                    "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoaderFaster"
                ) as MockFaster:
                    MockFaster.return_value = faster_loader
                    dataset = ShardedSingleStepDataset(
                        dataset_path="/fake/dataset",
                        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                        modality_configs=self.MODALITY_CONFIGS,
                        shard_size=1024,
                        episode_sampling_rate=1.0,
                        use_faster_episode_loader=True,
                    )
            else:
                dataset = ShardedSingleStepDataset(
                    dataset_path="/fake/dataset",
                    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                    modality_configs=self.MODALITY_CONFIGS,
                    shard_size=1024,
                    episode_sampling_rate=1.0,
                )
        return dataset

    class _RecordingFasterLoader:
        """Stand-in for LeRobotEpisodeLoaderFaster that records __call__ args."""

        def __init__(self, episode_df):
            self.episode_lengths = [50]
            self._episode_df = episode_df
            self.calls = []

        def get_episode_length(self, episode_index):
            return 50

        def __call__(self, ep_idx, step_indices, allow_padding=False):
            self.calls.append((ep_idx, np.asarray(step_indices), allow_padding))
            return self._episode_df

    def test_faster_loader_called_with_shard_steps(self):
        # The faster loader's __call__ must receive the shard's step indices
        # and the dataset's allow_padding setting. The loader returns the full
        # episode (50 rows, as the real loader does); the dataset trims steps
        # to the effective length (47 = 50 - 4 + 1).
        faster_loader = self._RecordingFasterLoader(self._episode_df(50))
        dataset = self._make_dataset(use_faster=True, faster_loader=faster_loader)
        dataset.processor = lambda messages: messages[0]["content"]

        dataset.get_shard(0)

        assert len(faster_loader.calls) == 1
        ep_idx, step_indices, allow_padding = faster_loader.calls[0]
        assert ep_idx == 0
        assert isinstance(step_indices, np.ndarray)
        assert len(step_indices) == 47  # 50 steps - 4 action horizon + 1
        assert allow_padding is False

    def test_default_uses_original_loader(self):
        # Default (flag off) must keep using the original __getitem__ path and
        # never instantiate the faster loader class.
        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
        from gr00t.data.embodiment_tags import EmbodimentTag

        with (
            patch(
                "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
            ) as MockLoader,
            patch(
                "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoaderFaster"
            ) as MockFaster,
        ):
            mock_loader = MagicMock()
            mock_loader.episode_lengths = [50]
            mock_loader.get_episode_length = lambda idx: 50
            MockLoader.return_value = mock_loader

            dataset = ShardedSingleStepDataset(
                dataset_path="/fake/dataset",
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_configs=self.MODALITY_CONFIGS,
                shard_size=1024,
                episode_sampling_rate=1.0,
            )

        assert dataset.episode_loader is mock_loader
        MockFaster.assert_not_called()
