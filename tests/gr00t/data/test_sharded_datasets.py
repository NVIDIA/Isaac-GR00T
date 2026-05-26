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

from gr00t.data.dataset.sharded_mixture_dataset import (
    ShardedMixtureDataset,
    _get_default_pg_tensor_device,
    merge_statistics,
)
from gr00t.data.interfaces import ShardedDataset
import numpy as np
import pytest
import torch


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

    def test_sidecar_metadata_is_skipped(self):
        """merge_statistics treats only entries that look like stats dicts (i.e.
        carry a 'mean' field) as action keys. Any sibling metadata producers
        co-locate at the top level — regardless of naming convention — must be
        ignored, not merged."""
        action_entry = {
            "min": [0.0],
            "max": [1.0],
            "mean": [0.5],
            "std": [0.2],
            "q01": [0.05],
            "q99": [0.95],
        }
        stats = [
            {
                "x": action_entry,
                # Real-world example: cache fingerprints written by generate_rel_stats (!313).
                "__fingerprints__": {"x": "sha256:deadbeef"},
                # Hypothetical future sidecars without dunder convention.
                "_provenance": {"source": "manual"},
                "schema_version": "1.0",
                "row_count": 1234,
            }
        ]
        merged = merge_statistics(stats, [1.0])
        assert set(merged.keys()) == {"x"}
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

    # -----------------------------------------------------------------------
    # Distributed seeding invariant: rank-symmetric seed enforcement
    # -----------------------------------------------------------------------
    #
    # ShardedMixtureDataset's shard partitioning works only when every rank
    # generates the same shard_sampling_schedule and slices into it
    # disjointly. A rank-asymmetric seed silently breaks this (see the class
    # docstring's "Distributed seeding invariant" section). The
    # _assert_seed_rank_symmetric guard turns that silent failure into a
    # fail-fast at __init__ / reset_seed time.
    #
    # These tests use unittest.mock to stand in for the torch.distributed
    # collective so we can simulate a multi-rank environment without spawning
    # real processes.

    @staticmethod
    def _make_fake_all_gather(per_rank_seeds):
        """Build a fake ``dist.all_gather`` that fills the output list with
        ``per_rank_seeds`` cast to ``int64`` tensors."""

        def _fake(out_list, _in_tensor, group=None):
            assert len(out_list) == len(per_rank_seeds), "world_size mismatch in test setup"
            for slot, seed in zip(out_list, per_rank_seeds):
                slot.copy_(torch.tensor([seed], dtype=torch.long))

        return _fake

    def _make_mixture_dist(self, world_size, this_rank_seed, per_rank_seeds):
        """Construct a ShardedMixtureDataset under a faked distributed env
        where ``dist.all_gather`` returns ``per_rank_seeds``."""
        datasets = [MockShardedDataset("/fake/path_0")]
        processor = MagicMock()
        processor.set_statistics = MagicMock()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.all_gather",
                side_effect=self._make_fake_all_gather(per_rank_seeds),
            ),
        ):
            return ShardedMixtureDataset(
                datasets=datasets,
                weights=[1.0],
                processor=processor,
                seed=this_rank_seed,
            )

    def test_seed_collective_uses_cpu_for_cpu_backend(self):
        with patch("torch.distributed.get_backend", return_value="gloo"):
            assert _get_default_pg_tensor_device() == torch.device("cpu")

    def test_seed_collective_uses_current_cuda_device_for_nccl_backend(self):
        with (
            patch("torch.distributed.get_backend", return_value="nccl"),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=3),
        ):
            assert _get_default_pg_tensor_device() == torch.device("cuda", 3)

    def test_seed_collective_raises_for_nccl_without_cuda(self):
        with (
            patch("torch.distributed.get_backend", return_value="nccl"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="requires CUDA tensors"):
                _get_default_pg_tensor_device()

    def test_init_passes_when_seeds_match_across_ranks(self):
        # All ranks pass the same seed: __init__ must succeed.
        mixture = self._make_mixture_dist(world_size=2, this_rank_seed=42, per_rank_seeds=[42, 42])
        assert mixture.seed == 42

    def test_init_raises_on_seed_mismatch_across_ranks(self):
        # rank 0 has seed=42, rank 1 has seed=43 → must fail loudly with a
        # message pointing at the docstring section.
        with pytest.raises(ValueError, match="seed must be identical on every rank"):
            self._make_mixture_dist(world_size=2, this_rank_seed=42, per_rank_seeds=[42, 43])

    def test_init_raises_on_plus_rank_pattern(self):
        # The classic "drive-by fix" pattern: caller did `seed = base + rank`
        # so each rank passes a different value. The error message should
        # surface all observed seeds.
        with pytest.raises(ValueError, match=r"\[42, 43, 44, 45\]"):
            self._make_mixture_dist(
                world_size=4,
                this_rank_seed=42,
                per_rank_seeds=[42, 43, 44, 45],
            )

    def test_reset_seed_raises_on_mismatch_across_ranks(self):
        # Resume path: reset_seed must enforce the same invariant. We first
        # construct a single-rank mixture (fast, no collective), then flip
        # world_size and patch dist for the reset_seed call.
        mixture = self._make_mixture()
        mixture.world_size = 2
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.all_gather",
                side_effect=self._make_fake_all_gather([100, 101]),
            ),
        ):
            with pytest.raises(ValueError, match="seed must be identical on every rank"):
                mixture.reset_seed(100)

    def test_assert_is_noop_in_single_rank(self):
        # The hot path: not distributed at all. Must not call all_gather.
        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.all_gather") as mock_all_gather,
        ):
            self._make_mixture()  # no exception
            mock_all_gather.assert_not_called()

    def test_assert_is_noop_when_world_size_one(self):
        # Distributed but world_size == 1 (degenerate single-process dist
        # init). Still no collective.
        datasets = [MockShardedDataset("/fake/path_0")]
        processor = MagicMock()
        processor.set_statistics = MagicMock()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.all_gather") as mock_all_gather,
        ):
            ShardedMixtureDataset(
                datasets=datasets,
                weights=[1.0],
                processor=processor,
                seed=42,
            )
            mock_all_gather.assert_not_called()


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
