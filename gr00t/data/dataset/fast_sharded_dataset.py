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

"""Streaming variant of the sharded loader (``data.loader=fast``).

Same shard schedule, the same samples built by the same code, and the same processor as
the stock loader; three things change:

1. No shard walls (any ``ShardedDataset``). Instead of building a whole shard before
   serving any of it, each worker loads shard splits (``get_shard_splits`` /
   ``load_split``) continuously on a small thread pool and serves samples from a
   shuffle buffer of about one shard. Samples are reordered within about
   ``shuffle_buffer`` samples, so the first epoch's worth of samples can include up to
   ``shuffle_buffer`` samples of the next epoch, with as many current-epoch samples
   served later instead, and with several datasets a batch usually mixes them instead
   of mostly coming from one shard. For a given seed the order is reproducible: it
   does not depend on decode timing.
2. Sparse decode (LeRobot datasets). A shard holds ``episode_sampling_rate`` of each of
   its episodes' steps, but the stock loader decodes every frame of every camera of
   those episodes. Only the frames the shard's steps read are decoded here.
3. Bounded FFmpeg threads (LeRobot datasets). The stock decoder uses
   ``num_ffmpeg_threads=0`` (one thread per core) in every dataloader worker, which
   oversubscribes the host when there are many ranks x workers.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gr00t.data.types import MessageType

from .sharded_mixture_dataset import ShardedMixtureDataset
from .sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data


FAST_LOADER_DEFAULTS = {
    "decode_threads": 2,
    "ffmpeg_threads": 1,
    "shuffle_buffer": 0,
    "min_fill": 0,
}


def fast_loader_settings(overrides: dict | None) -> dict:
    settings = dict(FAST_LOADER_DEFAULTS)
    for key, value in dict(overrides or {}).items():
        if key not in FAST_LOADER_DEFAULTS:
            raise KeyError(f"unknown fast loader setting {key!r}; known: {sorted(settings)}")
        settings[key] = int(value)
    return settings


class FastShardedSingleStepDataset(ShardedSingleStepDataset):
    """``ShardedSingleStepDataset`` that decodes only the frames its steps read."""

    def __init__(self, *args, ffmpeg_threads: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_loader.decoder_kwargs = {"num_ffmpeg_threads": int(ffmpeg_threads)}

    def frame_indices(self, step_indices: np.ndarray, length: int) -> np.ndarray:
        """Rows of a ``length``-row episode that ``extract_step_data`` reads video from."""
        if "video" not in self.modality_configs:
            return np.zeros(0, dtype=np.int64)
        deltas = np.asarray(self.modality_configs["video"].delta_indices, dtype=np.int64)
        rows = np.asarray(step_indices, dtype=np.int64)[:, None] + deltas[None, :]
        if self.allow_padding:
            rows = np.clip(rows, 0, length - 1)
        else:
            # Matches pandas .iloc, which reads negative rows from the episode end.
            rows = np.where(rows < 0, rows + length, rows)
        return np.unique(rows)

    def get_shard_splits(self, idx: int) -> list:
        return self.sharded_episodes[idx]

    def load_split(self, split) -> list:
        ep_idx, step_indices = split
        episode_data = self.episode_loader.load_episode(
            ep_idx, select_frames=lambda length: self.frame_indices(step_indices, length)
        )
        return [
            extract_step_data(
                episode_data, step, self.modality_configs, self.embodiment_tag, self.allow_padding
            )
            for step in step_indices
        ]

    def process_split_item(self, item) -> dict:
        assert self.processor is not None, "Processor must be set before getting datapoints"
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": item}])

    def get_shard(self, idx: int) -> list:
        return [
            self.process_split_item(item)
            for split in self.get_shard_splits(idx)
            for item in self.load_split(split)
        ]


class FastShardedMixtureDataset(ShardedMixtureDataset):
    """``ShardedMixtureDataset`` that streams shard splits through a shuffle buffer.

    Works with any ``ShardedDataset``. Each worker walks the same shard schedule as the
    stock loader and serves the same samples, reordered within about ``shuffle_buffer``
    samples instead of within each shard. Samples are processed on the iterating thread
    in the order they are served, and the buffer is refilled based only on how many
    samples have been served, so a given seed always gives the same order.
    """

    def __init__(
        self,
        *args,
        decode_threads: int = 2,
        shuffle_buffer: int = 0,
        min_fill: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decode_threads = max(1, int(decode_threads))
        default_buffer = int(
            max(
                getattr(ds, "shard_size", None)
                or max(ds.get_shard_length(i) for i in range(len(ds)))
                for ds in self.datasets
            )
        )
        self.shuffle_buffer = max(1, int(shuffle_buffer) or default_buffer)
        self.min_fill = min(self.shuffle_buffer, max(1, int(min_fill) or self.shuffle_buffer // 4))

    def _worker_schedule(self) -> list:
        schedule = self.filter_shard_sample_schedule()
        if not schedule:
            raise ValueError(
                f"Dataloader worker {self.worker_id} on rank {self.rank} has no shards: "
                f"num_shards_per_epoch={self.num_shards_per_epoch} must be at least "
                f"world_size * dataloader_num_workers = {self.world_size * self.num_workers}."
            )
        return schedule

    def _iter_splits(self, schedule: list):
        """This worker's (dataset, split) work items, epoch after epoch."""
        while True:
            for dataset_index, shard_index in schedule:
                for split in self.datasets[dataset_index].get_shard_splits(shard_index):
                    yield dataset_index, split
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            schedule = self._worker_schedule()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        splits = self._iter_splits(self._worker_schedule())
        # Loaded splits are the only slack against decode stalls, since the buffer is
        # refilled on a fixed schedule rather than whenever a split happens to be ready.
        max_in_flight = 8 * self.decode_threads
        pending: deque = deque()
        buffer: list = []
        served = 0
        with ThreadPoolExecutor(max_workers=self.decode_threads) as executor:
            in_flight: deque = deque()

            def submit():
                dataset_index, split = next(splits)
                future = executor.submit(self.datasets[dataset_index].load_split, split)
                in_flight.append((dataset_index, future))

            for _ in range(max_in_flight):
                submit()
            while True:
                # The target grows from min_fill to shuffle_buffer by one per served sample.
                while len(buffer) < min(self.shuffle_buffer, self.min_fill + served):
                    if not pending:
                        dataset_index, future = in_flight.popleft()
                        pending.extend((dataset_index, item) for item in future.result())
                        submit()
                    buffer.append(pending.popleft())
                pick = int(rng.integers(len(buffer)))
                buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
                dataset_index, item = buffer.pop()
                served += 1
                yield self.datasets[dataset_index].process_split_item(item)
