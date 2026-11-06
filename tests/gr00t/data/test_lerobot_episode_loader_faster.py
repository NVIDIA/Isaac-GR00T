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
            # History before episode start repeats the first observation
            (np.array([10, 11, 12]), [-15, 0], 100, False, [0, 10, 11, 12]),
            # allow_padding clamps out-of-range indices to [0, episode_length - 1]
            (np.array([0, 1]), [-3, 0], 100, True, [0, 1]),
            # Observation history is clamped even when padding is disabled
            (np.array([0, 1]), [-3, 0], 100, False, [0, 1]),
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

    def test_no_padding_repeats_first_observation(self):
        loader = _make_fake_loader()
        loader(0, np.array([0, 1]), allow_padding=False)
        requested = loader.requested_video_indices[-1]
        assert requested.tolist() == [0, 1]

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


@pytest.mark.parametrize("steps,length", [([7], 20), ([7, 14], 20), ([0], 5)])
def test_sparse_history_agrees_with_step_reader(steps, length):
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import EmbodimentTag

    loader = _make_fake_loader(episode_lengths=(length,))
    loader.modality_configs.pop("mask")
    loader.modality_configs["state"].delta_indices = [-15, 0]
    loader.modality_configs["action"].delta_indices = [0]
    loader._load_parquet_data = lambda _: pd.DataFrame(
        {
            "state.x": [np.array([i]) for i in range(length)],
            "action.x": [np.array([i]) for i in range(length)],
        }
    )
    # Backend-free unit check; the accompanying native-video study uses real MP4.
    loader._load_video_data = lambda _, indices: {
        "cam": [np.full((2, 2, 3), i, dtype=np.int64) for i in indices]
    }
    data = loader(0, np.array(steps), allow_padding=False)
    for index in steps:
        step = extract_step_data(
            data, index, loader.modality_configs, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False
        )
        assert step.images["cam"][0] is not None
        np.testing.assert_array_equal(step.images["cam"][0], 0)
        np.testing.assert_array_equal(step.images["cam"][1], index)
        np.testing.assert_array_equal(step.states["x"][:, 0], [0, index])


@pytest.mark.parametrize("length,step", [(20, 20), (0, 0)])
def test_invalid_video_index_rejected_before_decoding(length, step):
    loader = _make_fake_loader()
    with pytest.raises(IndexError, match="video index"):
        loader._get_video_indices_from_steps(np.array([step]), [0], length)


@pytest.mark.parametrize("steps", [[7], [0, 7], [7, 0], [7, 7], [7, 14]])
def test_shard_composition_preserves_sample_and_matches_full_decoding(steps):
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.types import EmbodimentTag

    loader = _make_fake_loader(episode_lengths=(20,))
    loader.episode_lengths = [20]
    loader.modality_configs.pop("mask")
    loader.modality_configs["video"].modality_keys = ["cam", "wrist"]
    loader.modality_configs["state"].delta_indices = [-15, 0]
    loader.modality_configs["action"].delta_indices = [0, 1]
    loader._load_parquet_data = lambda _: pd.DataFrame(
        {
            "state.x": [np.array([i]) for i in range(20)],
            "action.x": [np.array([100 + i]) for i in range(20)],
        }
    )
    # Pixels encode both frame and camera identity, including on the full path.
    loader._load_video_data = lambda _, indices: {
        camera: [np.full((2, 2, 3), i + offset, dtype=np.uint8) for i in indices]
        for camera, offset in [("cam", 0), ("wrist", 50)]
    }
    reference = extract_step_data(
        loader[0], 7, loader.modality_configs, EmbodimentTag.NEW_EMBODIMENT
    )
    with patch(
        "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoaderFaster",
        return_value=loader,
    ):
        dataset = ShardedSingleStepDataset(
            "/fake/dataset",
            EmbodimentTag.NEW_EMBODIMENT,
            loader.modality_configs,
            episode_sampling_rate=1.0,
            use_faster_episode_loader=True,
        )
    dataset.processor = lambda messages: messages[0]["content"]
    dataset.sharded_episodes = [[(0, np.array(steps))]]
    samples = dataset.get_shard(0)

    for index, sample in zip(steps, samples, strict=True):
        if index != 7:
            continue
        for camera, offset in [("cam", 0), ("wrist", 50)]:
            assert all(frame is not None for frame in sample.images[camera])
            np.testing.assert_array_equal(sample.images[camera], reference.images[camera])
            assert [frame[0, 0, 0] for frame in sample.images[camera]] == [offset, 7 + offset]
        np.testing.assert_array_equal(sample.states["x"], reference.states["x"])
        np.testing.assert_array_equal(sample.actions["x"], reference.actions["x"])
        assert sample.text == reference.text


@pytest.mark.parametrize("allow_padding", [False, True])
def test_empty_episode_rejected_before_video_backend(allow_padding):
    loader = _make_fake_loader(episode_lengths=(0,))
    loader._load_video_data = MagicMock()
    with pytest.raises(IndexError, match="video index"):
        loader(0, np.array([0]), allow_padding=allow_padding)
    loader._load_video_data.assert_not_called()


def test_future_overflow_rejected_before_video_backend():
    loader = _make_fake_loader(episode_lengths=(20,), video_delta_indices=(0, 1))
    loader._load_video_data = MagicMock()
    with pytest.raises(IndexError, match="video index 20"):
        loader(0, np.array([19]), allow_padding=False)
    loader._load_video_data.assert_not_called()


def test_explicit_padding_clamps_both_ends_before_decoding():
    loader = _make_fake_loader(episode_lengths=(20,), video_delta_indices=(-30, 0, 1))
    loader(0, np.array([19]), allow_padding=True)
    assert loader.requested_video_indices[-1].tolist() == [0, 19]
