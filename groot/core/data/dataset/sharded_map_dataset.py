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

import math
import time

import numpy as np
import structlog
from structlog.contextvars import bound_contextvars
from torch.utils.data import Dataset

from groot.core.data.types import EmbodimentTag, ShardedDataset, VLAStepDataId


class _FeistelPerm:
    """O(1)-memory pseudo-random permutation on [0, n).

    A 4-round Feistel network on [0, 2^k) (smallest even k with 2^k >= n) plus
    cycle walking restricts outputs to [0, n). The round function is a
    SplitMix64-style keyed mixer -- statistically indistinguishable from random
    for shuffling, but not cryptographically strong. Decorrelates within- and
    inter-shard order, fixing the arithmetic-progression artifact of an
    affine permutation `(b + a*i) mod n`.
    """

    _M1 = np.uint64(0xBF58476D1CE4E5B9)
    _M2 = np.uint64(0x94D049BB133111EB)
    _M3 = np.uint64(0xC4CEB9FE1A85EC53)
    _S30 = np.uint64(30)
    _S27 = np.uint64(27)
    _S31 = np.uint64(31)

    def __init__(self, n: int, seed: int, rounds: int = 4):
        self.n = int(n)
        bits = max(2, max(self.n - 1, 1).bit_length())
        if bits % 2:
            bits += 1
        self.half_bits = np.uint64(bits // 2)
        self.half_mask = (np.uint64(1) << self.half_bits) - np.uint64(1)
        key_rng = np.random.default_rng([int(seed), 0xFE15741])
        self.round_keys = key_rng.integers(0, np.iinfo(np.uint64).max, size=rounds, dtype=np.uint64)

    def _round(self, r: np.ndarray, key: np.uint64) -> np.ndarray:
        x = r ^ key
        x = x * self._M1
        x = x ^ (x >> self._S30)
        x = x * self._M2
        x = x ^ (x >> self._S27)
        x = x * self._M3
        x = x ^ (x >> self._S31)
        return x & self.half_mask

    def _encrypt(self, x: np.ndarray) -> np.ndarray:
        left = (x >> self.half_bits) & self.half_mask
        right = x & self.half_mask
        for k in self.round_keys:
            left, right = right, (left ^ self._round(right, k)) & self.half_mask
        return (left << self.half_bits) | right

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.ascontiguousarray(x, dtype=np.uint64).copy()
        y = self._encrypt(y)
        n_u = np.uint64(self.n)
        # Cycle-walk values that landed outside [0, n). Geometric distribution
        # with success rate n / 2^k >= 1/2, so a couple of iterations suffice.
        while True:
            mask = y >= n_u
            if not mask.any():
                break
            y[mask] = self._encrypt(y[mask])
        return y.astype(np.int64, copy=False)


class ShardedMapDataset(ShardedDataset):
    """Generic sharded dataset wrapper for any PyTorch Dataset.

    This class provides a simple yet effective sharding mechanism that can wrap any standard
    PyTorch map-style Dataset, enabling its use with the ShardedMixtureDataset for high-performance
    distributed data loading. It organizes data into balanced shards for distributed training
    by using random permutation to distribute dataset indices across shards, ensuring good
    load balancing while maintaining simplicity.

    Key features:
    - Wraps any existing PyTorch Dataset with sharding capability
    - Random permutation-based sharding for balanced load distribution
    - Statistics and embodiment tag passthrough from underlying dataset
    - Configurable shard sizes for memory and performance tuning

    The sharding strategy uses a round-robin distribution of randomly shuffled indices,
    which provides good balance across shards while being computationally efficient.
    This approach works well for datasets where individual samples are independent
    and don't require special grouping considerations.

    Args:
        dataset: Any PyTorch Dataset to be sharded
        shard_size: Target number of samples per shard (actual sizes may vary slightly)
        seed: Random seed for reproducible index shuffling and shard assignment

    Example:
        >>> # Wrap an existing dataset with sharding
        >>> original_dataset = MyCustomDataset(...)
        >>> sharded_dataset = ShardedMapDataset(
        ...     dataset=original_dataset,
        ...     shard_size=1024,
        ...     seed=42
        ... )
        >>> # Set processor and use in training pipeline
        >>> sharded_dataset.set_processor(my_processor)
        >>> shard_data = sharded_dataset.get_shard(0)
    """

    def __init__(
        self,
        dataset: Dataset,
        shard_size: int = 2**10,
        seed: int = 42,
        enable_preload_indices: bool = False,
    ):
        """Initialize the sharded map dataset with the given configuration.

        Sets up the sharding structure by randomly permuting dataset indices and
        distributing them across shards using a round-robin approach.

        Args:
            dataset: PyTorch Dataset to wrap with sharding functionality
            shard_size: Target number of samples per shard
            seed: Random seed for reproducible sharding
            enable_preload_indices: If True, forward shard indices to the wrapped
                dataset's preload hook before loading a shard.
        """
        self.dataset = dataset
        self.processor = None
        self.shard_size = shard_size
        self.seed = seed
        self.enable_preload_indices = enable_preload_indices

        # Lazy, O(1)-memory sharding: instead of materializing a full
        # rng.permutation(length) plus one index array per shard (which blows up
        # host RAM at init for large trajectory-mode datasets with many shards),
        # derive each shard's indices on demand via a Feistel permutation over
        # round-robin positions. See _get_shard_indices / _FeistelPerm.
        self.length = len(self.dataset)  # type: ignore[arg-type]
        assert self.length > 0, "Dataset is empty"
        self.num_shards = math.ceil(self.length / self.shard_size)
        self._perm = _FeistelPerm(self.length, seed)

    @staticmethod
    def _shard_length(idx: int, length: int, num_shards: int) -> int:
        if idx < 0 or idx >= num_shards:
            raise IndexError(f"Shard index {idx} out of range for {num_shards} shards")
        return (length - 1 - idx) // num_shards + 1

    def _get_shard_indices(self, idx: int) -> np.ndarray:
        """Materialize the dataset indices for shard `idx` on demand.

        Round-robin positions (idx, idx+num_shards, ...) are mapped through the
        Feistel permutation, reproducing the old shuffled round-robin sharding
        without materializing the full permutation or per-shard arrays.
        """
        if idx < 0 or idx >= self.num_shards:
            raise IndexError(f"Shard index {idx} out of range for {self.num_shards} shards")
        positions = np.arange(idx, self.length, self.num_shards, dtype=np.uint64)
        return self._perm(positions)

    def get_shard(self, idx: int) -> list:
        """Load and process all samples in a specific shard.

        Retrieves all dataset samples assigned to the specified shard and applies
        the configured processor to each sample. The processor transforms raw
        dataset samples into the format expected by the training pipeline.

        Args:
            idx: Shard index to load (0 to len(self) - 1)

        Returns:
            List of processed samples ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
            IndexError: If shard index is out of bounds
        """
        assert self.processor is not None, "Processor must be set before getting shard"

        shard_indices = self._get_shard_indices(idx)
        if self.enable_preload_indices:
            preload_indices = getattr(self.dataset, "preload_indices", None)
            if callable(preload_indices):
                preload_indices(shard_indices)

        with bound_contextvars(
            component="get_shard",
            dataset_path=str(getattr(self.dataset, "dataset_path", "unknown")),
            shard_idx=int(idx),
            dataset_idx=int(getattr(self, "dataset_index", 0)),
        ):
            start_t = time.perf_counter()
            shard_samples = []
            for sample_idx, i in enumerate(shard_indices):
                sequence_idx = int(i)
                with bound_contextvars(sample_idx=sample_idx, sequence_idx=sequence_idx):
                    sample_t0 = time.perf_counter()
                    data_sample = self.dataset[sequence_idx]
                    if data_sample:
                        processed = self.processor(data_sample)
                        shard_samples.append(processed)
                    # NOTE: episode_idx is the same for all messages under one sample.
                    # Consider adding this to a message header instead of reading from step metadata.
                    episode_idx = None
                    if data_sample:
                        for m in data_sample:
                            if (
                                isinstance(m, dict)
                                and m.get("type") == "episode_step"
                                and hasattr(m.get("content"), "metadata")
                            ):
                                sid = m["content"].metadata.get(VLAStepDataId.STEP_DATA_ID_KEY)
                                if sid is not None:
                                    episode_idx = sid.episode_idx
                                break
                    structlog.get_logger().debug(
                        "get_datapoint_completed",
                        episode_idx=episode_idx,
                        num_messages=(
                            len(data_sample)
                            if data_sample and hasattr(data_sample, "__len__")
                            else None
                        ),
                        num_frames=(
                            sum(
                                m["content"].num_frames
                                for m in data_sample
                                if isinstance(m, dict)
                                and m.get("type") == "episode_step"
                                and hasattr(m.get("content"), "num_frames")
                            )
                            if data_sample
                            else None
                        ),
                        duration_ms=round((time.perf_counter() - sample_t0) * 1000, 2),
                    )
            assert len(shard_samples) > 0, "No valid data samples encountered in the shard"
            structlog.get_logger().info(
                "get_shard_completed",
                shard_length=len(shard_samples),
                duration_ms=round((time.perf_counter() - start_t) * 1000, 2),
            )
        return shard_samples

    def __len__(self):
        """Return the number of shards in the dataset."""
        return self.num_shards

    def get_shard_length(self, idx: int) -> int:
        """Get the number of samples in a specific shard.

        Args:
            idx: Shard index to query

        Returns:
            Number of samples in the specified shard
        """
        return self._shard_length(idx, self.length, self.num_shards)

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying dataset."""
        try:
            return self.dataset.get_dataset_statistics()
        except NotImplementedError:
            return {}

    @property
    def embodiment_tag(self) -> EmbodimentTag | None:
        """Get the embodiment tag from the underlying dataset.

        Propagates embodiment information from the wrapped dataset to enable
        proper handling in multi-embodiment training scenarios.

        Returns:
            EmbodimentTag if available from underlying dataset, None otherwise
        """
        return getattr(self.dataset, "embodiment_tag", None)

    # ---- passthroughs so DeltaCAT introspection/stats work through the wrapper ----
    # The DeltaCAT sampled-stats preflight (deltacat_sampled_stats /
    # factory._has_deltacat_dataset / _install_deltacat_sampled_stats) introspects
    # the concrete dataset for dataset_path / episode_loader / modality_configs and
    # installs stats via set_dataset_statistics. In trajectory mode the dataset is
    # wrapped in this ShardedMapDataset, so forward those to the inner dataset
    # (e.g. EpisodeDataset -> DeltaCatPackDSEpisodeLoader).
    @property
    def dataset_path(self):
        return getattr(self.dataset, "dataset_path", None)

    @property
    def episode_loader(self):
        return getattr(self.dataset, "episode_loader", None)

    @property
    def modality_configs(self):
        return getattr(self.dataset, "modality_configs", None)

    def set_dataset_statistics(self, statistics) -> None:
        set_statistics = getattr(self.dataset, "set_dataset_statistics", None)
        if callable(set_statistics):
            set_statistics(statistics)
