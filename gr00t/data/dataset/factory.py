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

import numpy as np
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.data.types import ModalityConfig
from gr00t.utils.dist_utils import run_or_wait_on_rank0


def split_episode_indices(
    n_episodes: int, eval_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Partition episode ids into disjoint train and eval sets.

    The split is episode-level so no trajectory leaks across sets. ``eval_ratio``
    is the fraction assigned to eval. At least one episode remains in each set.
    """
    if n_episodes < 2:
        raise ValueError(
            f"eval_strategy requires at least 2 episodes to split, got {n_episodes}"
        )
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_set_split_ratio must be in (0, 1), got {eval_ratio}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_episodes)
    n_eval = int(round(n_episodes * eval_ratio))
    n_eval = min(max(n_eval, 1), n_episodes - 1)
    eval_ids = sorted(int(i) for i in shuffled[:n_eval])
    train_ids = sorted(int(i) for i in shuffled[n_eval:])
    return train_ids, eval_ids


def _episode_count(
    dataset_path: str, modality_configs: dict[str, ModalityConfig]
) -> int:
    return len(
        LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
        )
    )


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def _make_single_step(
        self,
        dataset_path: str,
        embodiment_tag: str,
        *,
        episode_indices: list[int] | None,
        episode_sampling_rate: float,
    ) -> ShardedSingleStepDataset:
        return ShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=EmbodimentTag(embodiment_tag),
            modality_configs=self.config.data.modality_configs[embodiment_tag],
            shard_size=self.config.data.shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=self.config.data.seed,
            allow_padding=self.config.data.allow_padding,
            episode_indices=episode_indices,
        )

    def _mix(
        self,
        datasets: list[ShardedSingleStepDataset],
        weights: list[float],
        processor: BaseProcessor,
        *,
        training: bool,
    ) -> ShardedMixtureDataset:
        mixed_weights: list[float] = list(weights)
        alpha = self.config.data.ds_weights_alpha
        if alpha is not None and len(datasets) > 1:
            ds_lengths = np.array([len(dataset) for dataset in datasets], dtype=np.float64)
            mixed_weights = (
                np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)
            ).tolist()
            print(
                f"Applied ds_weights_alpha={alpha} across {len(datasets)} datasets; "
                "this overrides per-dataset mix_ratio sampling weights."
            )
        return ShardedMixtureDataset(
            datasets=datasets,
            weights=mixed_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=training,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        want_eval = self.config.training.eval_strategy != "no"
        eval_ratio = self.config.training.eval_set_split_ratio

        train_datasets: list[ShardedSingleStepDataset] = []
        train_weights: list[float] = []
        eval_datasets: list[ShardedSingleStepDataset] = []
        eval_weights: list[float] = []

        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            path_train: list[ShardedSingleStepDataset] = []
            path_eval: list[ShardedSingleStepDataset] = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                modality_configs = self.config.data.modality_configs[embodiment_tag]
                with run_or_wait_on_rank0(label=f"generate_stats({dataset_path})") as is_rank0:
                    if is_rank0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))

                train_ids: list[int] | None = None
                eval_ids: list[int] | None = None
                if want_eval:
                    n_episodes = _episode_count(dataset_path, modality_configs)
                    train_ids, eval_ids = split_episode_indices(
                        n_episodes, eval_ratio, self.config.data.seed
                    )

                path_train.append(
                    self._make_single_step(
                        dataset_path,
                        embodiment_tag,
                        episode_indices=train_ids,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                    )
                )
                if want_eval:
                    path_eval.append(
                        self._make_single_step(
                            dataset_path,
                            embodiment_tag,
                            episode_indices=eval_ids,
                            episode_sampling_rate=1.0,
                        )
                    )

            train_lengths = np.array([len(dataset) for dataset in path_train])
            train_relative = train_lengths / train_lengths.sum()
            for dataset, relative_length in zip(path_train, train_relative):
                train_datasets.append(dataset)
                train_weights.append(relative_length * dataset_spec.mix_ratio)
            if want_eval:
                eval_lengths = np.array([len(dataset) for dataset in path_eval])
                eval_relative = eval_lengths / eval_lengths.sum()
                for dataset, relative_length in zip(path_eval, eval_relative):
                    eval_datasets.append(dataset)
                    eval_weights.append(relative_length * dataset_spec.mix_ratio)

        train_mixture = self._mix(
            train_datasets, train_weights, processor, training=True
        )
        eval_mixture = None
        if want_eval:
            eval_mixture = self._mix(
                eval_datasets, eval_weights, processor, training=False
            )
        return train_mixture, eval_mixture
