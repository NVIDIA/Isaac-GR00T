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
Test ShardedSingleStepDataset and ShardedMixtureDataset.

ShardedSingleStepDataset requires a real LeRobot-format dataset on disk,
so we mock the episode loader to test sharding logic in isolation.
ShardedMixtureDataset tests use lightweight mock ShardedDataset instances.
"""

from unittest.mock import MagicMock, patch

from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset, merge_statistics
from gr00t.data.interfaces import ShardedDataset
import numpy as np


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockShardedDataset(ShardedDataset):
    """Minimal ShardedDataset for testing mixture logic."""

    def __init__(self, dataset_path, num_shards=5, shard_length=100, embodiment_tag="robot_a"):
        super().__init__(dataset_path)
        self.num_shards = num_shards
        self._shard_length = shard_length
        self.embodiment_tag = type("ET", (), {"value": embodiment_tag})()
        self.shard_lengths = np.full(num_shards, shard_length)
        self._statistics = {
            "state": {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.2],
                    "q01": [0.05],
                    "q99": [0.95],
                },
            },
            "action": {
                "x": {
                    "min": [-1.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.3],
                    "q01": [-0.9],
                    "q99": [0.9],
                },
            },
        }

    def __len__(self):
        return self.num_shards

    def get_shard_length(self, idx):
        return self._shard_length

    def get_shard(self, idx):
        return [{"dummy": i} for i in range(self._shard_length)]

    def get_dataset_statistics(self):
        return self._statistics


# ---------------------------------------------------------------------------
# merge_statistics tests
# ---------------------------------------------------------------------------


class TestMergeStatistics:
    """Test weighted statistics merging used by ShardedMixtureDataset."""

    def test_single_dataset_passthrough(self):
        stats = [
            {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.2],
                    "q01": [0.1],
                    "q99": [0.9],
                }
            }
        ]
        merged = merge_statistics(stats, [1.0])
        assert "x" in merged
        np.testing.assert_allclose(merged["x"]["mean"], [0.5])
        np.testing.assert_allclose(merged["x"]["min"], [0.0])
        np.testing.assert_allclose(merged["x"]["max"], [1.0])

    def test_two_datasets_weighted_mean(self):
        stats = [
            {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.1],
                    "q01": [0.0],
                    "q99": [1.0],
                }
            },
            {
                "x": {
                    "min": [0.0],
                    "max": [2.0],
                    "mean": [1.0],
                    "std": [0.1],
                    "q01": [0.0],
                    "q99": [2.0],
                }
            },
        ]
        merged = merge_statistics(stats, [0.5, 0.5])
        np.testing.assert_allclose(merged["x"]["mean"], [0.5])
        np.testing.assert_allclose(merged["x"]["max"], [2.0])  # global max

    def test_weights_are_normalized(self):
        stats = [
            {
                "x": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.0],
                    "std": [0.1],
                    "q01": [0.0],
                    "q99": [1.0],
                }
            },
            {
                "x": {
                    "min": [0.0],
                    "max": [2.0],
                    "mean": [2.0],
                    "std": [0.1],
                    "q01": [0.0],
                    "q99": [2.0],
                }
            },
        ]
        merged = merge_statistics(stats, [3.0, 1.0])
        # weighted mean = (0*0.75 + 2*0.25) = 0.5
        np.testing.assert_allclose(merged["x"]["mean"], [0.5])


# ---------------------------------------------------------------------------
# ShardedMixtureDataset tests
# ---------------------------------------------------------------------------


class TestShardedMixtureDataset:
    """Test mixture dataset sampling and iteration."""

    def _make_mixture(self, num_datasets=2, training=True, num_shards_per_epoch=10):
        datasets = [
            MockShardedDataset(f"/fake/path_{i}", num_shards=5, shard_length=100)
            for i in range(num_datasets)
        ]
        weights = [1.0 / num_datasets] * num_datasets
        processor = MagicMock()
        processor.set_statistics = MagicMock()
        with patch("torch.distributed.is_initialized", return_value=False):
            return ShardedMixtureDataset(
                datasets=datasets,
                weights=weights,
                processor=processor,
                seed=42,
                training=training,
                num_shards_per_epoch=num_shards_per_epoch,
            )

    def test_length_equals_schedule(self):
        mixture = self._make_mixture()
        assert len(mixture.shard_sampling_schedule) > 0

    def test_eval_mode_visits_all_shards(self):
        mixture = self._make_mixture(training=False)
        schedule = mixture.shard_sampling_schedule
        # In eval mode, should visit every shard exactly once
        total_shards = sum(len(d) for d in mixture.datasets)
        assert len(schedule) == total_shards

    def test_get_dataset_statistics(self):
        mixture = self._make_mixture()
        stats = mixture.get_dataset_statistics()
        assert isinstance(stats, dict)

    def test_processor_receives_statistics(self):
        datasets = [MockShardedDataset("/fake/path_0")]
        processor = MagicMock()
        processor.set_statistics = MagicMock()
        with patch("torch.distributed.is_initialized", return_value=False):
            ShardedMixtureDataset(
                datasets=datasets,
                weights=[1.0],
                processor=processor,
                seed=42,
            )
        processor.set_statistics.assert_called_once()


# ---------------------------------------------------------------------------
# ShardedSingleStepDataset tests (with mocked episode loader)
# ---------------------------------------------------------------------------


class TestShardedSingleStepDataset:
    """Test sharding logic with mocked episode loader."""

    def test_shard_creation(self):
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import ModalityConfig

        modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
            "action": ModalityConfig(delta_indices=list(range(4)), modality_keys=["x"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }

        with patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
        ) as MockLoader:
            mock_loader = MagicMock()
            # 3 episodes, 50 steps each
            mock_loader.episode_lengths = [50, 50, 50]
            mock_loader.get_episode_length = lambda idx: 50
            MockLoader.return_value = mock_loader

            from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

            dataset = ShardedSingleStepDataset(
                dataset_path="/fake/dataset",
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_configs=modality_configs,
                shard_size=64,
                episode_sampling_rate=0.5,
                seed=42,
            )

        assert len(dataset) > 0
        assert all(length > 0 for length in dataset.shard_lengths)

    def test_effective_episode_length(self):
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import ModalityConfig

        modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
            "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["x"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }

        with patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
        ) as MockLoader:
            mock_loader = MagicMock()
            mock_loader.episode_lengths = [50]
            mock_loader.get_episode_length = lambda idx: 50
            MockLoader.return_value = mock_loader

            from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

            dataset = ShardedSingleStepDataset(
                dataset_path="/fake/dataset",
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_configs=modality_configs,
                shard_size=1024,
                episode_sampling_rate=1.0,
            )

        # effective = 50 - 8 + 1 = 43
        assert dataset.get_effective_episode_length(0) == 43
