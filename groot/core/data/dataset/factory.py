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

import os
from pathlib import Path
import time

import numpy as np
import torch.distributed as dist
from tqdm import tqdm

from groot.core.data import schema, stats, types
from groot.core.data.dataset import (
    build_artifacts,
    deltacat_sampled_stats,
    episode_dataset,
    factory_4_weighted_sampling,
    factory_4_weighted_sampling_v2,
    packds_single_step_dataset,
    sharded_map_dataset,
    sharded_mixture_dataset,
    sharded_single_step_chunk_dataset,
    sharded_single_step_dataset,
    sharded_single_step_rl_dataset,
    source_resolver,
    vision_language,
)
from groot.core.data.dataset.filter import annotation_source, sys2_filter
from groot.core.data.dataset.path_detection import is_packds_path
from groot.core.data.hydrator import client, service
from groot.core.data.hydrator.config import HydratorConfig
from groot.core.training.configs.data import filter_config
from groot.core.training.configs.data.data_config import DataConfig, DeltaCatDatasetConfig
from groot.core.utils.distributed import concurrent as concurrent_utils
from groot.core.utils.distributed import worker_logging


def _derive_experiment_name(explicit_name: str | None) -> str:
    """Resolve the experiment_name used as the Hydrator's per-job cache subdir.

    Base name: ``explicit_name`` if the caller passed one, else the
    ``JOB_NAME`` env var, else the literal ``"train"``.

    When ``GROOT_RUN_ID`` is set (``gear sky launch-job`` injects it as
    ``<username>_<launch_timestamp>``, identical across all ranks of the
    same submission), append it as a suffix so re-runs don't collide on
    the same hydrator_cache dir. Without ``GROOT_RUN_ID`` the base name is
    used as-is — callers that need per-run isolation outside of gear-sky
    must set ``GROOT_RUN_ID`` themselves.
    """
    base = explicit_name or os.environ.get("JOB_NAME") or "train"
    run_id = os.environ.get("GROOT_RUN_ID")
    return f"{base}_{run_id}" if run_id else base


def _is_packds_path(path: str) -> bool:
    """Check if a dataset path is a PackDS dataset (incl. ``dc://`` and ``yam_packds_*``)."""
    return is_packds_path(path)


def _is_deltacat_packds_path(path: str) -> bool:
    """Return whether the path points at a DeltaCAT PackDS table."""
    return str(path).startswith("dc://")


def _resolve_one_path(p: str) -> str:
    """Resolve a single dataset path via ``Hydrator.resolve_path``.

    Explicit ``s3://`` paths are returned unchanged (no bucket probe needed).
    PackDS paths are returned unchanged. Local paths go through the hydrator
    resolver, which may rewrite Lustre prefixes to ``s3://`` URIs.
    """
    s = str(p)
    if _is_packds_path(s) or s.startswith("s3://"):
        return s
    return service.Hydrator.resolve_path(s)


def resolve_dataset_paths(dataset_paths: list) -> list[str]:
    """Resolve a list of raw dataset paths via ``Hydrator.resolve_path``."""
    return [_resolve_one_path(p) for p in dataset_paths]


def _resolve_all_dataset_sources(
    data_config: DataConfig,
) -> list[list[source_resolver.ResolvedDatasetSource]]:
    """Resolve source indirection once and reuse it through dataset setup.

    DeltaCAT-backed configs may expand one logical dataset entry into many
    concrete LeRobot/PackDS roots.  Keeping the resolved form threaded through
    hydrator enablement, stats generation, and dataset construction prevents
    those code paths from each doing their own catalog lookup or format guess.
    """
    return [
        source_resolver.resolve_dataset_sources(dataset_spec.dataset_paths)
        for dataset_spec in data_config.datasets
    ]


def _deltacat_episode_loader_kwargs(
    deltacat_config: DeltaCatDatasetConfig | None,
) -> dict[str, object]:
    """Translate YAML DeltaCAT options into create_episode_loader kwargs."""
    if deltacat_config is None:
        return {}
    return {
        "episode_query": deltacat_config.episode_query,
        "steps_query": deltacat_config.steps_query,
        "as_of_timestamp": deltacat_config.as_of_timestamp,
        "key_remap": deltacat_config.key_remap,
    }


def _broadcast_object_from_rank0(value: object | None) -> object | None:
    """Broadcast one Python object from rank 0 through a temporary Gloo group."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    group = dist.new_group(backend="gloo")
    try:
        payload = [value if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=group)
        return payload[0]
    finally:
        dist.destroy_process_group(group)


def _rank0_log(message: str, *args: object) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    worker_logging.get_unified_logger().info(message, *args)


def _has_deltacat_dataset(datasets: list[types.ShardedDataset]) -> bool:
    """Return whether any concrete training dataset is DeltaCAT-backed."""
    for dataset in datasets:
        if str(getattr(dataset, "dataset_path", "")).startswith("dc://"):
            return True
        loader = getattr(dataset, "episode_loader", None)
        if loader is not None and loader.__class__.__name__ == "DeltaCatPackDSEpisodeLoader":
            return True
    return False


def _all_stats_cached(
    data_config: DataConfig,
    resolved_dataset_sources: list[list[source_resolver.ResolvedDatasetSource]] | None = None,
) -> bool:
    """Return whether eager LeRobot stats generation can be skipped.

    This is only a fast path for the eager stats pre-pass in
    ``_generate_all_stats``. PackDS stores stats in its own table metadata, so it
    is skipped here. Local and S3-backed LeRobot roots must already have both
    stats files; S3 checks use HEAD only and do not download file contents.
    """
    resolved_dataset_sources = resolved_dataset_sources or _resolve_all_dataset_sources(data_config)
    for dataset_spec, dataset_sources in zip(
        data_config.datasets, resolved_dataset_sources, strict=True
    ):
        embodiment_tag = dataset_spec.embodiment_tag
        if embodiment_tag is None:
            continue
        for source in dataset_sources:
            if source.format in ("packds", "deltacat"):
                continue
            resolved = _resolve_one_path(source.uri)
            for relative_path in (
                stats.LE_ROBOT_STATS_FILENAME,
                stats.LE_ROBOT_REL_STATS_FILENAME,
            ):
                meta_path = service.Hydrator.join_path(resolved, relative_path)
                if meta_path.startswith("s3://"):
                    if not service.Hydrator.remote_file_exists(meta_path):
                        return False
                elif not Path(meta_path).exists():
                    return False
    return True


def _generate_stats_for_path_worker(
    dataset_path_str: str,
    embodiment_tag_value: str,
    use_reservoir_sampling: bool,
    worker_id: int,
    hydrator_client: client.HydratorClient,
    use_relative_action: bool = False,
) -> tuple[str, str, str | None]:
    """Top-level worker for ProcessPoolExecutor.

    Uses passed HydratorClient (IPC to main-process Hydrator).
    Returns (path, stats_reason, rel_stats_reason).
    """
    print("[STATS_WORKER] process entered (before any logger)", flush=True)  # noqa: T201
    worker_log = worker_logging.get_unified_logger()
    worker_log.info("[STATS_WORKER] start path=%r worker_id=%s", dataset_path_str, worker_id)
    try:
        stats_reason = stats.generate_stats(dataset_path_str, hydrator_client) or "n/a"
        if use_relative_action:
            worker_log.info("[STATS_WORKER] generate_stats done, running generate_rel_stats ...")
            rel_reason = stats.generate_rel_stats(
                dataset_path_str,
                types.EmbodimentTag(embodiment_tag_value),
                use_reservoir_sampling=use_reservoir_sampling,
                hydrator=hydrator_client,
            )
        else:
            rel_reason = None
        worker_log.info("[STATS_WORKER] done path=%r", dataset_path_str)
        return (dataset_path_str, stats_reason, rel_reason)
    finally:
        hydrator_client.shutdown(wait=True)


def _needs_relative_stats(embodiment_tag: str, data_config: DataConfig) -> bool:
    """Return True if the embodiment has any RELATIVE action representation in its modality config."""
    modality_configs = getattr(data_config, "modality_configs", {}) or {}
    emb_config = modality_configs.get(embodiment_tag)
    if emb_config is None:
        return False
    # emb_config is dict[str, ModalityConfig]
    action_modality = emb_config.get("action") if isinstance(emb_config, dict) else None
    if action_modality is None:
        return False
    action_configs = getattr(action_modality, "action_configs", None)
    if not action_configs:
        return False
    return any(
        getattr(ac, "rep", None) == types.ActionRepresentation.RELATIVE for ac in action_configs
    )


def _generate_all_stats(
    data_config: DataConfig,
    hydrator: service.Hydrator | None,
    use_relative_action: bool | None = None,
    resolved_dataset_sources: list[list[source_resolver.ResolvedDatasetSource]] | None = None,
) -> None:
    """Generate stats/relative stats for all datasets.

    Uses the provided hydrator (single-process) or per-worker hydrators (multi-process).

    When hydrator is None (no S3 paths detected), stats are generated using local paths only.
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    worker_logging.get_unified_logger().info(
        "[RANK %s] _generate_all_stats: starting (may be slow if meta/stats.json missing on S3)",
        rank,
    )
    use_reservoir_sampling = getattr(data_config, "use_reservoir_sampling", True)
    auto_generate = getattr(data_config, "auto_generate_stats", True)
    resolved_dataset_sources = resolved_dataset_sources or _resolve_all_dataset_sources(data_config)
    if not dist.is_available() or not dist.is_initialized() or hydrator is None:
        if rank == 0:
            for dataset_spec, dataset_sources in zip(
                data_config.datasets, resolved_dataset_sources, strict=True
            ):
                embodiment_tag = dataset_spec.embodiment_tag
                if embodiment_tag is None:
                    continue
                for source in dataset_sources:
                    if source.format in ("packds", "deltacat"):
                        continue
                    path_str = service.Hydrator.normalize_path(_resolve_one_path(source.uri))
                    stats.generate_stats(path_str, hydrator, auto_generate=auto_generate)
                    use_rel = (
                        use_relative_action
                        if use_relative_action is not None
                        else _needs_relative_stats(embodiment_tag, data_config)
                    )
                    if use_rel:
                        stats.generate_rel_stats(
                            path_str,
                            schema.EmbodimentTag(embodiment_tag),
                            use_reservoir_sampling=use_reservoir_sampling,
                            hydrator=hydrator,
                            auto_generate=auto_generate,
                        )

        if dist.is_available() and dist.is_initialized():
            _gloo_group = dist.new_group(backend="gloo")
            dist.barrier(group=_gloo_group)
            dist.destroy_process_group(_gloo_group)
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Fast path: if all stats are already cached, skip the barrier entirely.
    # This avoids NCCL socket errors during dist.barrier() on large clusters
    # when no actual stats computation is needed.
    if _all_stats_cached(data_config, resolved_dataset_sources):
        if rank == 0:
            print(  # noqa: T201
                "All dataset stats already cached, skipping stats generation barrier."
            )
        return

    # Build a deterministic global task list; deduplicate by (path, embodiment_tag).
    raw_tasks: list[tuple[str, object]] = []
    for dataset_spec, dataset_sources in zip(
        data_config.datasets, resolved_dataset_sources, strict=True
    ):
        embodiment_tag = dataset_spec.embodiment_tag
        if embodiment_tag is None:
            continue
        for source in dataset_sources:
            if source.format in ("packds", "deltacat"):
                continue
            raw_tasks.append(
                (
                    service.Hydrator.normalize_path(_resolve_one_path(source.uri)),
                    embodiment_tag,
                )
            )
    seen: set[tuple[str, object]] = set()
    tasks: list[tuple[str, object]] = []
    for path, tag in raw_tasks:
        key = (path, getattr(tag, "value", tag))
        if key in seen:
            continue
        seen.add(key)
        tasks.append((path, tag))

    # Shard tasks across ranks; each pool worker uses HydratorClient (IPC to main process).
    rank_tasks = []
    for idx, (dataset_path, embodiment_tag) in enumerate(tasks):
        if idx % world_size != rank or _is_packds_path(dataset_path):
            continue
        tag_str = getattr(embodiment_tag, "value", str(embodiment_tag))
        use_rel = (
            use_relative_action
            if use_relative_action is not None
            else _needs_relative_stats(tag_str, data_config)
        )
        rank_tasks.append(
            (
                str(dataset_path),
                tag_str,
                use_reservoir_sampling,
                len(rank_tasks),
                use_rel,
            )
        )
    if rank_tasks:
        max_workers = min(10, len(rank_tasks))
        hydrator_client = hydrator.get_ipc_client()
        concurrent_utils.run_with_process_pool(
            _generate_stats_for_path_worker,
            max_workers,
            [t[0] for t in rank_tasks],
            [t[1] for t in rank_tasks],
            [t[2] for t in rank_tasks],
            [t[3] for t in rank_tasks],
            [hydrator_client] * len(rank_tasks),
            [t[4] for t in rank_tasks],
        )
        # Do not stop IPC server here: build() needs get_ipc_client() after this, and
        # DataLoader workers use it during training. Shutdown via factory.stop() / pipeline.stop().

    # Ensure all ranks see completed stats before continuing.
    # Use a temporary Gloo process group for the barrier instead of the default
    # NCCL group.  This is a CPU-only synchronisation point (no tensors are
    # exchanged), so NCCL is unnecessary.  More importantly, NCCL barriers can
    # fail with "Cuda failure 999" at large scale (128+ GPUs) when the CUDA
    # context is stressed by model loading that happened just before this call.
    # Gloo barriers use TCP sockets and are immune to CUDA state issues.
    _gloo_group = dist.new_group(backend="gloo")
    dist.barrier(group=_gloo_group)
    dist.destroy_process_group(_gloo_group)


def get_dataset_factory(
    data_config: DataConfig,
    experiment_name: str | None = None,
    use_relative_action: bool | None = None,
    eval_strategy: str = "no",
    dataset_build_artifacts: build_artifacts.DatasetBuildArtifacts | None = None,
):
    """Return the appropriate dataset factory based on ``data_config.dataset_factory_type``."""
    factory_type = getattr(data_config, "dataset_factory_type", "default")
    if factory_type == "default":
        return DatasetFactory(
            data_config,
            experiment_name,
            use_relative_action,
            dataset_build_artifacts=dataset_build_artifacts,
        )
    elif factory_type == "weighted_sampling":
        return factory_4_weighted_sampling.DatasetFactoryWeightedSampling(
            data_config,
            experiment_name=experiment_name,
            use_relative_action=use_relative_action,
            eval_strategy=eval_strategy,
        )
    elif factory_type == "quality_weighted":
        return factory_4_weighted_sampling_v2.DatasetFactoryGroupedSampling(
            data_config,
            factory_4_weighted_sampling_v2.QualityPriorityGrouper(),
            experiment_name=experiment_name,
            use_relative_action=use_relative_action,
            eval_strategy=eval_strategy,
        )
    else:
        raise ValueError(f"Unknown dataset_factory_type: {factory_type}")


class DatasetFactory:
    """Factory class for building training datasets. Model-agnostic."""

    def __init__(
        self,
        data_config: DataConfig,
        experiment_name: str | None = None,
        use_relative_action: bool | None = None,
        dataset_build_artifacts: build_artifacts.DatasetBuildArtifacts | None = None,
    ):
        self.data_config = data_config
        materialize_as_of = getattr(self.data_config, "materialize_deltacat_as_of_timestamps", None)
        if callable(materialize_as_of):
            materialize_as_of()
        self.experiment_name = _derive_experiment_name(experiment_name)
        self.use_relative_action = use_relative_action
        self.dataset_build_artifacts = (
            dataset_build_artifacts or build_artifacts.DatasetBuildArtifacts()
        )

    def _create_hydrator(self) -> service.Hydrator:
        """Create a Hydrator for the current process/rank."""
        worker_id = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        hydrator_config = getattr(self.data_config, "hydrator_config", None) or HydratorConfig()
        return service.Hydrator(
            job_name=self.experiment_name, worker_id=worker_id, config=hydrator_config
        )

    def _get_data_filter(self):
        """Build the data filter from config, or return None."""
        if self.data_config.filter.filter_type is None:
            return None
        filter_type = self.data_config.filter.filter_type
        if filter_type == filter_config.FilterType.ANNOTATION_SOURCE:
            if self.data_config.filter.subtask_source_filter is None:
                raise ValueError("AnnotationSourceFilter requires subtask_source_filter to be set")
            return annotation_source.AnnotationSourceFilter(
                self.data_config.filter.subtask_source_filter,
            )
        elif filter_type == filter_config.FilterType.SYS2:
            return sys2_filter.Sys2Filter(
                self.data_config.filter.subtask_source_filter,
                language_mode=self.data_config.filter.language_mode,
                subtask_frac=self.data_config.filter.subtask_frac,
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def _create_sharded_dataset(
        self,
        dataset_path,
        embodiment_tag,
        data_filter,
        hydrator=None,
        dataset_format: source_resolver.DatasetSourceFormat | None = None,
        deltacat_config: DeltaCatDatasetConfig | None = None,
    ):
        """Create a single ShardedDataset from a path and embodiment tag."""
        if embodiment_tag is None:
            return sharded_map_dataset.ShardedMapDataset(
                dataset=vision_language.VisionLanguageDataset(dataset_path),
                shard_size=self.data_config.shard_size,
                seed=self.data_config.seed,
            )

        is_deltacat_dataset_path = dataset_format == "deltacat" or _is_deltacat_packds_path(
            dataset_path
        )
        if is_deltacat_dataset_path and self.data_config.mode not in ("single_turn", "trajectory"):
            raise ValueError(
                "DeltaCAT PackDS training supports dataset_factory_type=default "
                "with data.mode in {single_turn, trajectory} only."
            )

        if self.data_config.mode in ["single_turn", "single_turn_rl"]:
            is_packds = (
                dataset_format == "packds"
                if dataset_format is not None
                else _is_packds_path(dataset_path)
            )
            is_deltacat_packds = (
                dataset_format == "deltacat"
                if dataset_format is not None
                else _is_deltacat_packds_path(dataset_path)
            )
            if is_packds and not is_deltacat_packds:
                if self.data_config.mode == "single_turn_rl":
                    raise ValueError("PackDS does not support single_turn_rl mode yet")
                return packds_single_step_dataset.PackSingleStepDataset(
                    packds_path=dataset_path,
                    embodiment_tag=schema.EmbodimentTag(embodiment_tag),
                    modality_configs=self.data_config.modality_configs[embodiment_tag],
                    shard_size=self.data_config.shard_size,
                    episode_sampling_rate=self.data_config.episode_sampling_rate,
                    seed=self.data_config.seed,
                )
            else:
                if is_deltacat_packds and self.data_config.mode == "single_turn_rl":
                    raise ValueError("DeltaCAT PackDS does not support single_turn_rl mode yet")
                if self.data_config.mode == "single_turn":
                    dataset_class = sharded_single_step_dataset.ShardedSingleStepDataset
                elif self.data_config.mode == "single_turn_rl":
                    dataset_class = sharded_single_step_rl_dataset.ShardedSingleStepRLDataset
                else:
                    raise ValueError(f"Invalid mode: {self.data_config.mode}")
                return dataset_class(
                    dataset_path=dataset_path,
                    embodiment_tag=schema.EmbodimentTag(embodiment_tag),
                    modality_configs=self.data_config.modality_configs[embodiment_tag],
                    video_backend=self.data_config.video_backend,
                    resample_on_invalid_frames=self.data_config.resample_on_invalid_frames,
                    shard_size=self.data_config.shard_size,
                    episode_sampling_rate=self.data_config.episode_sampling_rate,
                    seed=self.data_config.seed,
                    allow_padding=self.data_config.allow_padding,
                    data_filter=data_filter,
                    hydrator=hydrator,
                    episode_loader_kwargs=(
                        _deltacat_episode_loader_kwargs(deltacat_config)
                        if is_deltacat_packds
                        else None
                    ),
                    enable_batched_step_extraction=is_deltacat_packds,
                )
        elif self.data_config.mode == "chunk":
            return sharded_single_step_chunk_dataset.ShardedSingleStepChunkDataset(
                dataset_path=dataset_path,
                embodiment_tag=schema.EmbodimentTag(embodiment_tag),
                modality_configs=self.data_config.modality_configs[embodiment_tag],
                video_backend=self.data_config.video_backend,
                resample_on_invalid_frames=self.data_config.resample_on_invalid_frames,
                shard_size=self.data_config.shard_size,
                episode_sampling_rate=self.data_config.episode_sampling_rate,
                seed=self.data_config.seed,
                allow_padding=self.data_config.allow_padding,
                data_filter=data_filter,
                num_chunks=self.data_config.num_chunks,
                hydrator=hydrator,
            )
        elif self.data_config.mode == "trajectory":
            return sharded_map_dataset.ShardedMapDataset(
                dataset=episode_dataset.EpisodeDataset(
                    dataset_path=str(dataset_path),
                    embodiment_tag=schema.EmbodimentTag(embodiment_tag),
                    modality_configs=self.data_config.modality_configs[embodiment_tag],
                    video_backend=self.data_config.video_backend,
                    resample_on_invalid_frames=self.data_config.resample_on_invalid_frames,
                    data_filter=data_filter,
                    max_len=self.data_config.max_len,
                    step_stride=self.data_config.trajectory_step_stride,
                    hydrator=hydrator,
                    episode_loader_kwargs=(
                        _deltacat_episode_loader_kwargs(deltacat_config)
                        if is_deltacat_dataset_path
                        else None
                    ),
                ),
                shard_size=self.data_config.shard_size,
                seed=self.data_config.seed,
                enable_preload_indices=is_deltacat_dataset_path,
            )
        else:
            raise ValueError(f"Invalid mode: {self.data_config.mode}")

    def build_val_datasets(self, processor: types.BaseProcessor) -> dict[str, object]:
        """Build per-embodiment validation datasets keyed by embodiment tag.

        Returns an empty dict if no val_dataset_path is configured.
        Each dataset has its processor set so it can be iterated directly.
        """
        data_filter = self._get_data_filter()
        val_datasets: dict[str, object] = {}

        for dataset_spec in self.data_config.datasets:
            val_path = dataset_spec.val_dataset_path
            if val_path is None:
                continue
            embodiment_tag = dataset_spec.embodiment_tag
            if embodiment_tag is None:
                continue
            val_path_str = _resolve_one_path(str(val_path))
            val_ds = self._create_sharded_dataset(
                val_path_str,
                embodiment_tag,
                data_filter,
                dataset_format="deltacat" if _is_deltacat_packds_path(val_path_str) else None,
                deltacat_config=dataset_spec.deltacat,
            )
            val_ds.set_processor(processor)
            val_datasets[embodiment_tag] = val_ds

        return val_datasets

    def build(self, processor: types.BaseProcessor) -> tuple[
        sharded_mixture_dataset.ShardedMixtureDataset,
        sharded_mixture_dataset.ShardedMixtureDataset | None,
    ]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        resolved_dataset_sources = _resolve_all_dataset_sources(self.data_config)

        hydrator_enabled = any(
            _resolve_one_path(source.uri).startswith("s3://")
            for dataset_sources in resolved_dataset_sources
            for source in dataset_sources
            if source.format not in ("packds", "deltacat")
        )
        if hydrator_enabled:
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactory.build: auto-enabling hydrator (path mapping resolved to S3)",
                rank,
            )

        if hydrator_enabled:
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactory.build: creating hydrator ...", rank
            )
            hydrator = self._create_hydrator()
            self._hydrator = hydrator
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactory.build: hydrator created", rank
            )
            hydrator_client: client.HydratorClient | None = hydrator.get_ipc_client()
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactory.build: hydrator IPC server started"
                " (HydratorClient for dataloader workers)",
                rank,
            )
        else:
            worker_logging.get_unified_logger().info(
                "[RANK %s] DatasetFactory.build: no S3 paths detected, skipping hydrator", rank
            )
            hydrator = None
            hydrator_client = None

        # Pre-generate statistics for all datasets.
        _generate_all_stats(
            self.data_config,
            hydrator,
            self.use_relative_action,
            resolved_dataset_sources,
        )

        all_datasets = []
        all_weights = []
        data_filter = self._get_data_filter()

        for dataset_spec, dataset_sources in tqdm(
            zip(self.data_config.datasets, resolved_dataset_sources, strict=True),
            total=len(self.data_config.datasets),
            desc="Initializing datasets",
        ):
            # -- training datasets (one per path) --
            datasets = []
            embodiment_tag = dataset_spec.embodiment_tag
            for source in dataset_sources:
                dataset_path = _resolve_one_path(source.uri)
                dataset = self._create_sharded_dataset(
                    dataset_path,
                    embodiment_tag,
                    data_filter,
                    hydrator=hydrator_client,
                    dataset_format=source.format,
                    deltacat_config=dataset_spec.deltacat,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths, strict=True):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        precomputed_stats_by_embodiment = self._compute_deltacat_sampled_stats(
            all_datasets,
            all_weights,
            processor,
        )

        train_dataset = sharded_mixture_dataset.ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.data_config.seed,
            training=True,
            num_shards_per_epoch=self.data_config.num_shards_per_epoch,
            override_pretraining_statistics=self.data_config.override_pretraining_statistics,
            precomputed_stats_by_embodiment=precomputed_stats_by_embodiment,
            enable_shard_prefetch_pipeline=_has_deltacat_dataset(all_datasets),
        )
        return (train_dataset, None)

    def _compute_deltacat_sampled_stats(
        self,
        datasets: list[types.ShardedDataset],
        weights: list[float],
        processor: types.BaseProcessor,
    ) -> dict[str, dict]:
        """Compute and install DeltaCAT sampled stats before mixture construction."""
        if not _has_deltacat_dataset(datasets):
            return {}
        if self.data_config.mode not in ("single_turn", "trajectory"):
            raise ValueError(
                "DeltaCAT sampled stats supports data.mode in {single_turn, trajectory} only."
            )

        started_at = time.perf_counter()
        precomputed = self.dataset_build_artifacts.deltacat_stats_by_embodiment
        if precomputed:
            _rank0_log(
                "DeltaCAT sampled stats preflight: using precomputed stats for %s embodiment(s)",
                len(precomputed),
            )
            self._install_deltacat_sampled_stats(datasets, precomputed)
            return precomputed

        _rank0_log(
            "DeltaCAT sampled stats preflight: starting for %s dataset(s)",
            len(datasets),
        )

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            computed = _broadcast_object_from_rank0(None)
        else:
            computed = deltacat_sampled_stats.compute_from_datasets(
                datasets,
                weights,
                processor=processor,
                data_config=self.data_config,
            )
            computed = _broadcast_object_from_rank0(computed)

        precomputed = computed or {}
        self._install_deltacat_sampled_stats(datasets, precomputed)
        _rank0_log(
            "DeltaCAT sampled stats preflight: finished in %.2fs",
            time.perf_counter() - started_at,
        )
        return precomputed

    @staticmethod
    def _install_deltacat_sampled_stats(
        datasets: list[types.ShardedDataset],
        stats_by_embodiment: dict[str, dict],
    ) -> None:
        for dataset in datasets:
            if not _has_deltacat_dataset([dataset]):
                continue
            emb = getattr(dataset, "embodiment_tag", None)
            embodiment = emb.value if hasattr(emb, "value") else str(emb)
            stats_for_emb = stats_by_embodiment.get(embodiment)
            if stats_for_emb is None:
                continue
            dataset.set_dataset_statistics(stats_for_emb)

    def stop(self) -> None:
        """Shutdown the hydrator (e.g. when training ends). Stops IPC server and cache."""
        hydrator = getattr(self, "_hydrator", None)
        if hydrator is not None:
            hydrator.shutdown()
