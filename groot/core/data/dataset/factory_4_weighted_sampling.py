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
"""Weighted BC dataset factory (v2) with action-source-aware sampling.

Splits rollout datasets into human-chunk and policy-chunk subsets and assigns
sampling weights so that clean demos, human chunks, and policy chunks each
receive a target fraction of the total training budget.

See the plan doc for the full algorithm:
1. initial_demo_ratio = total_clean / total_all
2. final_demo_ratio  = min(initial_demo_ratio, demo_ratio_cap)
3. rollout_human_target_prob  = configurable (default 0.5)
4. rollout_policy_target_prob = 1 - rollout_human_target_prob - final_demo_ratio
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch.distributed as dist
from tqdm import tqdm

from groot.core.utils.distributed import worker_logging
from groot.core.data import enums
from groot.core.data import schema
from groot.core.training.configs.data import filter_config
from groot.core.training.configs.data.data_config import DataConfig
from groot.core.data import types
from groot.core.data.dataset import (
    factory,
    sharded_mixture_dataset,
    sharded_single_step_dataset,
    sharded_single_step_rl_dataset,
)
from groot.core.data.dataset.filter import annotation_source, sys2_filter
from groot.core.data.hydrator import service
from groot.core.data.hydrator.config import HydratorConfig

HUMAN_ACTION_SOURCE = enums.ActionSourceEnum.HUMAN.value  # 2


def _raise_unsupported_packds_path(dataset_path: object) -> None:
    path = str(dataset_path)
    if factory._is_deltacat_packds_path(path):
        raise ValueError(
            "DeltaCAT dc:// datasets are supported only by the default "
            "DatasetFactory; set data.dataset_factory_type=default and data.mode=single_turn."
        )
    if factory._is_packds_path(path):
        raise ValueError("WeightedBC factory v2 does not support PackDS datasets")


def _compute_valid_anchor_indices(
    episode_len: int,
    min_delta_index: int,
    max_delta_index: int,
    allow_padding: bool,
) -> np.ndarray:
    if episode_len <= 0:
        return np.empty(0, dtype=np.int64)
    if allow_padding:
        return np.arange(episode_len, dtype=np.int64)

    start = max(0, -min_delta_index)
    stop = episode_len - max_delta_index
    if stop <= start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def _read_action_source_column(parquet_path: Path) -> np.ndarray | None:
    """Read action source values from a parquet file.

    Tries ``rl_info.action_source_mask`` first (gr00t_SH convention), then
    falls back to ``action.source`` (legacy convention).  Returns None if
    neither column exists.
    """
    for col_name in ("rl_info.action_source_mask", "action.source"):
        try:
            raw_df = pd.read_parquet(parquet_path, columns=[col_name])
            values = raw_df[col_name].values
            if values.ndim > 1:
                values = values[:, 0]
            elif values.dtype == object:
                values = np.array([int(v) if np.isscalar(v) else int(v[0]) for v in values])
            return values.astype(np.int64)
        except (KeyError, ValueError):
            continue
    return None


def _classify_steps_for_dataset(
    dataset_path: str | Path,
    min_delta_index: int,
    max_delta_index: int,
    lookahead: int,
    allow_padding: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Pre-scan parquet files to classify each valid step as human-chunk or policy-chunk.

    A step is a "human chunk" if any of the first ``lookahead`` steps in its
    action horizon window has ``action source == HUMAN``.  Everything else
    is a "policy chunk".

    Returns two dicts mapping episode_index -> np.ndarray of valid step indices
    for human chunks and policy chunks respectively.
    """
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"

    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    with open(meta_dir / "episodes.jsonl") as f:
        episodes_meta = [json.loads(line) for line in f]

    data_path_pattern = info["data_path"]
    chunk_size = info["chunks_size"]

    human_indices: dict[int, np.ndarray] = {}
    policy_indices: dict[int, np.ndarray] = {}

    for ep_meta in episodes_meta:
        ep_idx = ep_meta["episode_index"]
        ep_len = int(ep_meta["length"])
        valid_steps = _compute_valid_anchor_indices(
            episode_len=ep_len,
            min_delta_index=min_delta_index,
            max_delta_index=max_delta_index,
            allow_padding=allow_padding,
        )
        if len(valid_steps) == 0:
            continue

        chunk_idx = ep_idx // chunk_size
        parquet_filename = data_path_pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        parquet_path = dataset_path / parquet_filename

        action_source = _read_action_source_column(parquet_path)
        if action_source is None:
            action_source = np.full(ep_len, HUMAN_ACTION_SOURCE, dtype=np.int64)

        is_human = action_source == HUMAN_ACTION_SOURCE

        human_steps = []
        policy_steps = []
        for step in valid_steps:
            window_end = min(step + lookahead, ep_len)
            if np.any(is_human[step:window_end]):
                human_steps.append(step)
            else:
                policy_steps.append(step)

        if human_steps:
            human_indices[ep_idx] = np.array(human_steps, dtype=np.int64)
        if policy_steps:
            policy_indices[ep_idx] = np.array(policy_steps, dtype=np.int64)

    return human_indices, policy_indices


class DatasetFactoryWeightedSampling:
    """Factory that computes sampling weights via action-source-aware splitting.

    Expects each ``SingleDatasetConfig`` to have ``dataset_role`` set to either
    ``"clean_demo"`` or ``"rollout"``.  Datasets with ``dataset_role="default"``
    are treated as clean demos.
    """

    def __init__(
        self,
        data_config: DataConfig,
        experiment_name: str | None = None,
        use_relative_action: bool | None = None,
        eval_strategy: str = "no",
    ):
        self.data_config = data_config
        from groot.core.data.dataset.factory import _derive_experiment_name
        self.experiment_name = _derive_experiment_name(experiment_name)
        self.use_relative_action = use_relative_action
        self.eval_strategy = eval_strategy
        demo_cap = data_config.weighted_sample_batch_demo_ratio_cap
        human_prob = data_config.weighted_sample_batch_rollout_human_target_prob
        if demo_cap + human_prob > 1.0:
            raise ValueError(
                f"weighted_sample_batch_demo_ratio_cap ({demo_cap}) + "
                f"weighted_sample_batch_rollout_human_target_prob ({human_prob}) "
                f"exceeds 1.0 -- no budget left for policy chunks"
            )

    def _create_hydrator(self) -> service.Hydrator:
        """Create a Hydrator for the current process/rank."""
        worker_id = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        job_name = self.experiment_name
        hydrator_config = getattr(self.data_config, "hydrator_config", None) or HydratorConfig()
        return service.Hydrator(job_name=job_name, worker_id=worker_id, config=hydrator_config)

    def build(self, processor: types.BaseProcessor) -> tuple[
        sharded_mixture_dataset.ShardedMixtureDataset,
        sharded_mixture_dataset.ShardedMixtureDataset | None,
    ]:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        assert self.eval_strategy == "no", "Sharded dataset does not support evaluation sets"

        if self.data_config.hydrator_enabled:
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactoryWeightedSampling.build: creating hydrator ...", rank
            )
            hydrator = self._create_hydrator()
            self._hydrator = hydrator
            hydrator_client = hydrator.get_ipc_client()
        else:
            hydrator = None
            hydrator_client = None

        factory._generate_all_stats(self.data_config, hydrator, self.use_relative_action)

        data_filter = self._build_data_filter()
        lookahead = self.data_config.weighted_sample_rollout_human_chunk_lookahead

        # Phase 1: build all datasets and classify rollout steps
        # Each entry is (dataset, mix_ratio) so _compute_weights can use mix_ratio
        # as a within-category weighting multiplier.
        demo_datasets: list[tuple[sharded_single_step_dataset.ShardedSingleStepDataset, float]] = []
        rollout_human_datasets: list[
            tuple[sharded_single_step_dataset.ShardedSingleStepDataset, float]
        ] = []
        rollout_policy_datasets: list[
            tuple[sharded_single_step_dataset.ShardedSingleStepDataset, float]
        ] = []

        for dataset_spec in tqdm(
            self.data_config.datasets,
            total=len(self.data_config.datasets),
            desc="[WeightedBC] Initializing datasets",
        ):
            role = getattr(dataset_spec, "dataset_role", "default")
            mix_ratio = getattr(dataset_spec, "mix_ratio", 1.0)
            if role in ("default", "clean_demo"):
                for dataset_path in dataset_spec.dataset_paths:
                    _raise_unsupported_packds_path(dataset_path)
                    ds = self._build_single_turn_dataset(
                        dataset_path, dataset_spec, data_filter, hydrator_client=hydrator_client
                    )
                    demo_datasets.append((ds, mix_ratio))

            elif role == "rollout":
                for dataset_path in dataset_spec.dataset_paths:
                    _raise_unsupported_packds_path(dataset_path)
                    embodiment_tag = dataset_spec.embodiment_tag
                    assert embodiment_tag is not None, "Rollout datasets require embodiment_tag"
                    modality_configs = self.data_config.modality_configs[embodiment_tag]
                    min_delta = min(min(c.delta_indices) for c in modality_configs.values())
                    max_delta = max(max(c.delta_indices) for c in modality_configs.values())

                    print(
                        f"[WeightedBC] Pre-scanning {dataset_path} for action source classification..."
                    )
                    human_step_filter, policy_step_filter = _classify_steps_for_dataset(
                        dataset_path,
                        min_delta,
                        max_delta,
                        lookahead,
                        allow_padding=self.data_config.allow_padding,
                    )

                    human_count = sum(len(v) for v in human_step_filter.values())
                    policy_count = sum(len(v) for v in policy_step_filter.values())
                    print(
                        f"[WeightedBC]   -> human chunks: {human_count}, "
                        f"policy chunks: {policy_count}"
                    )

                    if human_count > 0:
                        ds_human = self._build_single_turn_dataset(
                            dataset_path,
                            dataset_spec,
                            data_filter,
                            step_indices_filter=human_step_filter,
                            hydrator_client=hydrator_client,
                        )
                        rollout_human_datasets.append((ds_human, mix_ratio))

                    if policy_count > 0:
                        ds_policy = self._build_single_turn_dataset(
                            dataset_path,
                            dataset_spec,
                            data_filter,
                            step_indices_filter=policy_step_filter,
                            hydrator_client=hydrator_client,
                        )
                        rollout_policy_datasets.append((ds_policy, mix_ratio))
            else:
                raise ValueError(f"Unknown dataset_role: {role}")

        # Phase 2: compute weights
        all_datasets, all_weights = self._compute_weights(
            demo_datasets, rollout_human_datasets, rollout_policy_datasets
        )

        return (
            sharded_mixture_dataset.ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.data_config.seed,
                training=True,
                num_shards_per_epoch=self.data_config.num_shards_per_epoch,
                override_pretraining_statistics=self.data_config.override_pretraining_statistics,
            ),
            None,
        )

    def _compute_weights(
        self,
        demo_datasets: list[tuple],
        human_datasets: list[tuple],
        policy_datasets: list[tuple],
    ) -> tuple[list, list[float]]:
        demo_cap = self.data_config.weighted_sample_batch_demo_ratio_cap
        human_target = self.data_config.weighted_sample_batch_rollout_human_target_prob

        total_demo_steps = sum(sum(ds.shard_lengths) for ds, _ in demo_datasets)
        total_human_steps = sum(sum(ds.shard_lengths) for ds, _ in human_datasets)
        total_policy_steps = sum(sum(ds.shard_lengths) for ds, _ in policy_datasets)
        total_all = total_demo_steps + total_human_steps + total_policy_steps

        if total_all == 0:
            raise ValueError("No data found across all datasets")

        initial_demo_ratio = total_demo_steps / total_all if total_demo_steps > 0 else 0.0
        final_demo_ratio = min(initial_demo_ratio, demo_cap)
        policy_target = 1.0 - human_target - final_demo_ratio

        if policy_target < 0:
            print(
                f"[WeightedBC] WARNING: rollout_policy_target_prob is negative "
                f"({policy_target:.4f}). Clamping to 0. "
                f"demo_ratio={final_demo_ratio:.4f}, human_target={human_target:.4f}"
            )
            policy_target = 0.0

        # Weighted step counts: mix_ratio acts as a multiplier on effective step count
        # within each category. When all mix_ratio=1.0, this is identical to plain step counts.
        weighted_demo = sum(sum(ds.shard_lengths) * ratio for ds, ratio in demo_datasets)
        weighted_human = sum(sum(ds.shard_lengths) * ratio for ds, ratio in human_datasets)
        weighted_policy = sum(sum(ds.shard_lengths) * ratio for ds, ratio in policy_datasets)

        print("[WeightedBC] Sampling weights summary:")
        print(f"  total_demo_steps   = {total_demo_steps}")
        print(f"  total_human_steps  = {total_human_steps}")
        print(f"  total_policy_steps = {total_policy_steps}")
        print(f"  weighted_demo (steps*mix_ratio)   = {weighted_demo:.1f}")
        print(f"  weighted_human (steps*mix_ratio)   = {weighted_human:.1f}")
        print(f"  weighted_policy (steps*mix_ratio)  = {weighted_policy:.1f}")
        print(f"  initial_demo_ratio = {initial_demo_ratio:.4f}")
        print(f"  final_demo_ratio   = {final_demo_ratio:.4f}")
        print(f"  rollout_human_target_prob  = {human_target:.4f}")
        print(f"  rollout_policy_target_prob = {policy_target:.4f}")

        self.sampling_summary = {
            "wbc/total_demo_steps": total_demo_steps,
            "wbc/total_human_steps": total_human_steps,
            "wbc/total_policy_steps": total_policy_steps,
            "wbc/weighted_demo": weighted_demo,
            "wbc/weighted_human": weighted_human,
            "wbc/weighted_policy": weighted_policy,
            "wbc/initial_demo_ratio": initial_demo_ratio,
            "wbc/final_demo_ratio": final_demo_ratio,
            "wbc/rollout_human_target_prob": human_target,
            "wbc/rollout_policy_target_prob": policy_target,
        }

        all_datasets = []
        all_weights = []

        for ds, ratio in demo_datasets:
            ds_steps = sum(ds.shard_lengths)
            w = final_demo_ratio * (ds_steps * ratio / weighted_demo) if weighted_demo > 0 else 0.0
            all_datasets.append(ds)
            all_weights.append(w)

        for ds, ratio in human_datasets:
            ds_steps = sum(ds.shard_lengths)
            w = human_target * (ds_steps * ratio / weighted_human) if weighted_human > 0 else 0.0
            all_datasets.append(ds)
            all_weights.append(w)

        for ds, ratio in policy_datasets:
            ds_steps = sum(ds.shard_lengths)
            w = policy_target * (ds_steps * ratio / weighted_policy) if weighted_policy > 0 else 0.0
            all_datasets.append(ds)
            all_weights.append(w)

        print(f"[WeightedBC] Final per-dataset weights: {all_weights}")
        return all_datasets, all_weights

    def _build_single_turn_dataset(
        self,
        dataset_path,
        dataset_spec,
        data_filter,
        step_indices_filter=None,
        hydrator_client=None,
    ) -> sharded_single_step_dataset.ShardedSingleStepDataset:
        embodiment_tag = dataset_spec.embodiment_tag
        assert embodiment_tag is not None, "embodiment_tag is required"
        _raise_unsupported_packds_path(dataset_path)

        if self.data_config.mode == "single_turn":
            dataset_class = sharded_single_step_dataset.ShardedSingleStepDataset
        elif self.data_config.mode == "single_turn_rl":
            dataset_class = sharded_single_step_rl_dataset.ShardedSingleStepRLDataset
        else:
            raise ValueError(f"WeightedBC factory does not support mode: {self.data_config.mode}")

        return dataset_class(
            dataset_path=dataset_path,
            embodiment_tag=schema.EmbodimentTag(embodiment_tag),
            modality_configs=self.data_config.modality_configs[embodiment_tag],
            hydrator=hydrator_client,
            video_backend=self.data_config.video_backend,
            resample_on_invalid_frames=self.data_config.resample_on_invalid_frames,
            shard_size=self.data_config.shard_size,
            episode_sampling_rate=self.data_config.episode_sampling_rate,
            seed=self.data_config.seed,
            allow_padding=self.data_config.allow_padding,
            data_filter=data_filter,
            step_indices_filter=step_indices_filter,
        )

    def _build_data_filter(self):
        if self.data_config.filter.filter_type is None:
            return None
        filter_type = self.data_config.filter.filter_type
        if filter_type == filter_config.FilterType.ANNOTATION_SOURCE:
            if self.data_config.filter.subtask_source_filter is None:
                raise ValueError("AnnotationSourceFilter requires subtask_source_filter to be set")
            return annotation_source.AnnotationSourceFilter(
                self.data_config.filter.subtask_source_filter
            )
        elif filter_type == filter_config.FilterType.SYS2:
            return sys2_filter.Sys2Filter(
                self.data_config.filter.subtask_source_filter,
                language_mode=self.data_config.filter.language_mode,
                subtask_frac=self.data_config.filter.subtask_frac,
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def stop(self) -> None:
        """Shutdown the hydrator (e.g. when training ends)."""
        hydrator = getattr(self, "_hydrator", None)
        if hydrator is not None:
            hydrator.shutdown()
