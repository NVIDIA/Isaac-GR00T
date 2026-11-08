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

"""The "fast" loader must serve exactly the stock loader's samples."""

from collections import Counter
import hashlib
from itertools import islice
import shutil
import threading
import time
from types import SimpleNamespace

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset import sharded_mixture_dataset
from gr00t.data.dataset.fast_sharded_dataset import (
    FastShardedMixtureDataset,
    FastShardedSingleStepDataset,
)
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.stats import generate_stats
import numpy as np
import pandas as pd
import pytest
from test_support.runtime import get_root


pytest.importorskip("torchcodec.decoders")

TAG = EmbodimentTag.SIMPLER_ENV_WIDOWX


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "simplerenv_bridge_sample"
    shutil.copytree(get_root() / "demo_data" / "simplerenv_bridge_sample", path)
    generate_stats(path)
    return path


class IdentityProcessor:
    """Returns the raw VLAStepData so samples can be compared before augmentation."""

    def set_statistics(self, statistics, override=False):
        pass

    def __call__(self, messages):
        return messages[0]["content"]


def sample_key(step) -> str:
    h = hashlib.sha256(step.text.encode())
    for group in (step.images, step.states, step.actions):
        for key in sorted(group):
            for value in np.atleast_1d(np.asarray(group[key])):
                h.update(key.encode())
                h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def make_dataset(path, cls, **kwargs):
    ds = cls(
        dataset_path=path,
        embodiment_tag=TAG,
        modality_configs=MODALITY_CONFIGS[TAG.value],
        shard_size=16,
        episode_sampling_rate=0.25,
        seed=7,
        **kwargs,
    )
    ds.set_processor(IdentityProcessor())
    return ds


def build_mixture(cls, ds, **kwargs):
    return cls(
        datasets=[ds],
        weights=[1.0],
        processor=IdentityProcessor(),
        seed=11,
        num_shards_per_epoch=len(ds),
        **kwargs,
    )


def test_get_shard_matches_stock_sample_for_sample(dataset_path):
    stock = make_dataset(dataset_path, ShardedSingleStepDataset)
    fast = make_dataset(dataset_path, FastShardedSingleStepDataset)
    assert stock.sharded_episodes is not fast.sharded_episodes
    assert len(stock) == len(fast) > 1
    for idx in range(len(stock)):
        a = [sample_key(s) for s in stock.get_shard(idx)]
        b = [sample_key(s) for s in fast.get_shard(idx)]
        assert a == b, f"shard {idx} differs"


def test_frame_indices_cover_exactly_the_rows_read(dataset_path):
    fast = make_dataset(dataset_path, FastShardedSingleStepDataset)
    ep_idx, steps = fast.sharded_episodes[0][0]
    length = fast.episode_loader.get_episode_length(ep_idx)
    rows = fast.frame_indices(steps, length)
    assert sorted(rows.tolist()) == sorted(set(steps.tolist()))


def test_episodes_shorter_than_their_metadata_match_stock(dataset_path, tmp_path):
    # Rows are resolved against the rows actually loaded, as stock does, not the
    # metadata length.
    path = tmp_path / "short"
    shutil.copytree(dataset_path, path)
    for parquet in sorted((path / "data" / "chunk-000").glob("*.parquet")):
        df = pd.read_parquet(parquet)
        df.iloc[: len(df) - 3].to_parquet(parquet)
    stock = make_dataset(path, ShardedSingleStepDataset, allow_padding=True)
    fast = make_dataset(path, FastShardedSingleStepDataset, allow_padding=True)
    for idx in range(len(stock)):
        a = [sample_key(s) for s in stock.get_shard(idx)]
        b = [sample_key(s) for s in fast.get_shard(idx)]
        assert a == b, f"shard {idx} differs"


def test_one_epoch_serves_the_same_samples_as_stock(dataset_path):
    stock_mix = build_mixture(
        ShardedMixtureDataset, make_dataset(dataset_path, ShardedSingleStepDataset)
    )
    fast_mix = build_mixture(
        FastShardedMixtureDataset,
        make_dataset(dataset_path, FastShardedSingleStepDataset),
        decode_threads=2,
        shuffle_buffer=1,
        min_fill=1,
    )
    epoch = int(sum(stock_mix.datasets[0].shard_lengths))
    stock = Counter(sample_key(s) for s in islice(iter(stock_mix), epoch))
    fast = Counter(sample_key(s) for s in islice(iter(fast_mix), epoch))
    assert stock == fast


class ListShardDataset(ShardedDataset):
    """A non-LeRobot dataset that only implements the base ``ShardedDataset`` contract."""

    def __init__(self, source: ShardedSingleStepDataset):
        super().__init__(source.dataset_path)
        self.embodiment_tag = source.embodiment_tag
        self.shards = [source.get_shard(i) for i in range(len(source))]
        self.statistics = source.get_dataset_statistics()

    def __len__(self):
        return len(self.shards)

    def get_shard_length(self, idx):
        return len(self.shards[idx])

    def get_shard(self, idx):
        return self.shards[idx]

    def get_dataset_statistics(self):
        return self.statistics


def test_streams_any_sharded_dataset(dataset_path):
    stock_ds = make_dataset(dataset_path, ShardedSingleStepDataset)
    stock_mix = build_mixture(ShardedMixtureDataset, stock_ds)
    generic_ds = ListShardDataset(stock_ds)
    default_mix = build_mixture(FastShardedMixtureDataset, generic_ds)
    assert default_mix.shuffle_buffer == max(len(s) for s in generic_ds.shards)
    fast_mix = build_mixture(FastShardedMixtureDataset, generic_ds, shuffle_buffer=1, min_fill=1)
    epoch = int(sum(stock_ds.shard_lengths))
    stock = Counter(sample_key(s) for s in islice(iter(stock_mix), epoch))
    fast = Counter(sample_key(s) for s in islice(iter(fast_mix), epoch))
    assert stock == fast


def test_default_buffer_blurs_the_epoch_boundary_by_at_most_one_buffer(dataset_path):
    stock_mix = build_mixture(
        ShardedMixtureDataset, make_dataset(dataset_path, ShardedSingleStepDataset)
    )
    fast_mix = build_mixture(
        FastShardedMixtureDataset, make_dataset(dataset_path, FastShardedSingleStepDataset)
    )
    epoch = int(sum(stock_mix.datasets[0].shard_lengths))
    stock = Counter(sample_key(s) for s in islice(iter(stock_mix), epoch))
    fast = Counter(sample_key(s) for s in islice(iter(fast_mix), epoch))
    assert sum((stock - fast).values()) <= fast_mix.shuffle_buffer


class JitteryDataset(ListShardDataset):
    """Splits shards into single samples that take a random time to load."""

    def get_shard_splits(self, idx):
        return [(idx, i) for i in range(len(self.shards[idx]))]

    def load_split(self, split):
        time.sleep(np.random.uniform(0, 0.003))
        idx, i = split
        return [self.shards[idx][i]]


class ThreadRecordingProcessor(IdentityProcessor):
    def __init__(self):
        self.threads = set()

    def __call__(self, messages):
        self.threads.add(threading.get_ident())
        return super().__call__(messages)


def test_order_is_reproducible_and_processing_stays_on_one_thread(dataset_path):
    source = make_dataset(dataset_path, ShardedSingleStepDataset)
    orders = []
    for _ in range(2):
        processor = ThreadRecordingProcessor()
        fast = make_dataset(dataset_path, FastShardedSingleStepDataset)
        mix = build_mixture(
            FastShardedMixtureDataset, fast, decode_threads=3, shuffle_buffer=24, min_fill=4
        )
        fast.set_processor(processor)
        orders.append([sample_key(s) for s in islice(iter(mix), 80)])
        assert processor.threads == {threading.get_ident()}
    assert orders[0] == orders[1]

    jittery = [
        [
            sample_key(s)
            for s in islice(
                iter(
                    build_mixture(
                        FastShardedMixtureDataset,
                        JitteryDataset(source),
                        decode_threads=4,
                        shuffle_buffer=24,
                        min_fill=4,
                    )
                ),
                80,
            )
        ]
        for _ in range(2)
    ]
    assert jittery[0] == jittery[1]


def test_worker_without_shards_fails_instead_of_hanging(dataset_path, monkeypatch):
    mix = build_mixture(
        FastShardedMixtureDataset, make_dataset(dataset_path, FastShardedSingleStepDataset)
    )
    mix.num_shards_per_epoch = 1
    mix.shard_sampling_schedule = mix.generate_shard_sampling_schedule()
    monkeypatch.setattr(
        sharded_mixture_dataset, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=4)
    )
    with pytest.raises(ValueError, match="num_shards_per_epoch"):
        next(iter(mix))


def test_shuffle_buffer_only_reorders(dataset_path):
    stock_mix = build_mixture(
        ShardedMixtureDataset, make_dataset(dataset_path, ShardedSingleStepDataset)
    )
    fast_mix = build_mixture(
        FastShardedMixtureDataset,
        make_dataset(dataset_path, FastShardedSingleStepDataset),
        shuffle_buffer=32,
        min_fill=8,
    )
    epoch = int(sum(stock_mix.datasets[0].shard_lengths))
    known = set(sample_key(s) for s in islice(iter(stock_mix), epoch))
    served = Counter(sample_key(s) for s in islice(iter(fast_mix), epoch))
    assert set(served) <= known
    assert max(served.values()) <= 2
    assert len(served) > 0.8 * len(known)
