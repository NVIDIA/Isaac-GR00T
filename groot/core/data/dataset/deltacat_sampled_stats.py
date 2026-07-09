# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-time sampled statistics for DeltaCAT PackDS datasets.

Public API:

- ``compute_from_config(...)`` runs before training dataset construction. It is
  used by ``experiment_ray.py`` so Ray launches can sample stats on CPU tasks
  before GPU training actors are created.
- ``compute_from_datasets(...)`` is the fallback used by ``DatasetFactory`` after
  concrete ``ShardedDataset`` objects exist.

Both functions are side-effect free: they return ``stats_by_embodiment`` and do
not mutate ``DataConfig``, datasets, or episode loaders. ``DatasetFactory`` is
the only caller that installs the returned stats onto datasets/loaders.

Sampling matches the effective training distribution for DeltaCAT v1 by
stratifying each ``dc://`` dataset within an embodiment, normalizing final
mixture weights within that embodiment, sampling valid training anchor steps,
and computing exact weighted q01/q99 bounds over the sampled low-dimensional
state/action values. Video columns are never requested by this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np

from groot.core.data import schema, types
from groot.core.data import stats as data_stats
from groot.core.data.dataset import source_resolver

logger = logging.getLogger(__name__)

STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")
_LOWDIM_READ_EPISODE_BATCH_SIZE = 64

__all__ = [
    "compute_from_config",
    "compute_from_datasets",
]


class _DeltaCatStatsPreflightNotSupported(RuntimeError):
    """Raised when config-only stats preflight cannot exactly match training."""


def _log_info(message: str, *args: Any) -> None:
    logger.info(message, *args)
    try:
        from groot.core.utils.distributed import worker_logging

        worker_logging.get_unified_logger().info(message, *args)
    except Exception:
        logging.info(message, *args)


@dataclass(frozen=True)
class _DeltaCatStatsSource:
    """One concrete DeltaCAT sharded dataset stratum in the final train mixture."""

    dataset: types.ShardedDataset
    weight: float
    embodiment: str
    dataset_path: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _DeltaCatStatsStratumSpec:
    """Serializable DeltaCAT stratum spec for pre-training stats workers."""

    dataset_path: str
    embodiment: str
    modality_configs: dict[str, types.ModalityConfig]
    weight: float
    episode_query: str | None
    steps_query: str | None
    as_of_timestamp: int | None
    key_remap: dict[str, str]
    shard_size: int
    episode_sampling_rate: float
    seed: int
    allow_padding: bool
    video_backend: str
    resample_on_invalid_frames: bool


@dataclass
class _DeltaCatStratumSamples:
    """Raw weighted samples returned by one stratum worker."""

    embodiment: str
    metadata: dict[str, Any]
    state: dict[str, "_WeightedSamples"]
    action: dict[str, "_WeightedSamples"]
    relative_action: dict[str, "_WeightedSamples"]


class _WeightedSamples:
    """Collect weighted samples for one stats key."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.values: list[np.ndarray] = []
        self.weights: list[float] = []
        self.null_count = 0
        self.drop_count = 0

    def add(self, value: Any, weight: float) -> None:
        if value is None:
            self.null_count += 1
            return
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            self.drop_count += 1
            return
        self.values.append(arr)
        self.weights.append(float(weight))

    @property
    def count(self) -> int:
        return len(self.values)

    def extend(self, other: "_WeightedSamples") -> None:
        self.values.extend(other.values)
        self.weights.extend(other.weights)
        self.null_count += other.null_count
        self.drop_count += other.drop_count

    def to_stats(self, *, min_non_null_samples: int, context: str) -> dict[str, Any]:
        if self.count < min_non_null_samples:
            raise ValueError(
                "DeltaCAT sampled stats found too few non-null samples for "
                f"{self.key!r}: count={self.count}, required={min_non_null_samples}. {context}"
            )
        values = np.stack(self.values, axis=0).astype(np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        return {
            "min": np.min(values, axis=0).astype(np.float32).tolist(),
            "max": np.max(values, axis=0).astype(np.float32).tolist(),
            "mean": _weighted_mean(values, weights).astype(np.float32).tolist(),
            "std": _weighted_std(values, weights).astype(np.float32).tolist(),
            "q01": _weighted_quantile(values, weights, 0.01).astype(np.float32).tolist(),
            "q99": _weighted_quantile(values, weights, 0.99).astype(np.float32).tolist(),
        }


def _is_deltacat_dataset(dataset: types.ShardedDataset) -> bool:
    """Return whether a built sharded dataset is backed by DeltaCAT PackDS."""
    if str(getattr(dataset, "dataset_path", "")).startswith("dc://"):
        return True
    loader = getattr(dataset, "episode_loader", None)
    return loader is not None and loader.__class__.__name__ == "DeltaCatPackDSEpisodeLoader"


def _collect_deltacat_sources(
    datasets: list[types.ShardedDataset],
    weights: list[float],
) -> list[_DeltaCatStatsSource]:
    return [
        _make_deltacat_source(dataset, float(weight))
        for dataset, weight in zip(datasets, weights, strict=True)
        if _is_deltacat_dataset(dataset)
    ]


def _make_deltacat_source(dataset: types.ShardedDataset, weight: float) -> _DeltaCatStatsSource:
    emb = getattr(dataset, "embodiment_tag", None)
    if emb is None:
        raise ValueError(f"DeltaCAT dataset {dataset.dataset_path!r} is missing embodiment_tag.")
    reader = getattr(getattr(dataset, "episode_loader", None), "_reader", None)
    return _DeltaCatStatsSource(
        dataset=dataset,
        weight=weight,
        embodiment=emb.value if hasattr(emb, "value") else str(emb),
        dataset_path=str(dataset.dataset_path),
        metadata={
            "episode_query": getattr(reader, "episode_query", None),
            "steps_query": getattr(reader, "steps_query", None),
            "as_of_timestamp": getattr(reader, "as_of_timestamp", None),
            "key_remap": getattr(reader, "key_remap", {}),
        },
    )


def _validate_no_mixed_source_embodiments(datasets: list[types.ShardedDataset]) -> None:
    """Reject v1 same-embodiment mixes of DeltaCAT and non-DeltaCAT sources."""
    sources_by_emb: dict[str, set[str]] = defaultdict(set)
    paths_by_emb: dict[str, list[str]] = defaultdict(list)
    for dataset in datasets:
        emb = getattr(dataset, "embodiment_tag", None)
        if emb is None:
            continue
        emb_value = emb.value if hasattr(emb, "value") else str(emb)
        source = "deltacat" if _is_deltacat_dataset(dataset) else "other"
        sources_by_emb[emb_value].add(source)
        paths_by_emb[emb_value].append(str(getattr(dataset, "dataset_path", dataset)))

    for emb, source_types in sources_by_emb.items():
        if "deltacat" in source_types and "other" in source_types:
            raise ValueError(
                "DeltaCAT sampled stats v1 does not support mixing DeltaCAT and "
                f"non-DeltaCAT datasets for the same embodiment {emb!r}. "
                f"Datasets: {paths_by_emb[emb]}"
            )


def compute_from_config(
    data_config: Any,
    *,
    model_config: Any | None = None,
    allow_ray: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute DeltaCAT stats before model/training dataset construction.

    This path is config/spec based so it can run on the Ray driver before GPU
    training actors are spawned. If the config shape cannot be represented
    exactly without constructing training datasets, the caller falls back to the
    existing DatasetFactory path.
    """
    try:
        specs = _build_preflight_specs(data_config)
    except _DeltaCatStatsPreflightNotSupported as exc:
        _log_info("Skipping config-level DeltaCAT stats preflight: %s", exc)
        return {}
    if not specs:
        return {}

    _validate_percentile_model_config(specs, model_config)
    started_at = time.perf_counter()
    _log_info(
        "DeltaCAT sampled stats config preflight: starting for %d stratum spec(s)",
        len(specs),
    )

    try:
        import torch.distributed as dist
    except ImportError:  # torch is optional for unit tests that only import this module.
        dist = None

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            stats = _run_preflight_specs(
                specs,
                stats_config=data_config.deltacat_stats,
                default_seed=data_config.seed,
                allow_ray=allow_ray,
            )
        else:
            stats = None
        object_list = [stats]
        group = dist.new_group(backend="gloo")
        dist.broadcast_object_list(object_list, src=0, group=group)
        dist.destroy_process_group(group)
        stats = object_list[0] or {}
    else:
        stats = _run_preflight_specs(
            specs,
            stats_config=data_config.deltacat_stats,
            default_seed=data_config.seed,
            allow_ray=allow_ray,
        )

    _log_info(
        "DeltaCAT sampled stats config preflight: finished in %.2fs",
        time.perf_counter() - started_at,
    )
    return stats


def _build_preflight_specs(data_config: Any) -> list[_DeltaCatStatsStratumSpec]:
    """Build serializable DeltaCAT stats specs from a training DataConfig."""
    specs: list[_DeltaCatStatsStratumSpec] = []
    sources_by_emb: dict[str, set[str]] = defaultdict(set)
    factory_type = getattr(data_config, "dataset_factory_type", "default")

    for dataset_spec in getattr(data_config, "datasets", []):
        embodiment = getattr(dataset_spec, "embodiment_tag", None)
        if embodiment is None:
            continue
        embodiment = str(embodiment)
        resolved_sources: list[source_resolver.ResolvedDatasetSource] = []
        for dataset_path in getattr(dataset_spec, "dataset_paths", []):
            resolved_sources.extend(source_resolver.resolve_dataset_source(dataset_path))

        has_deltacat = any(_is_deltacat_source(source) for source in resolved_sources)
        for source in resolved_sources:
            sources_by_emb[embodiment].add("deltacat" if _is_deltacat_source(source) else "other")

        if not has_deltacat:
            continue
        if factory_type != "default":
            raise ValueError(
                "DeltaCAT sampled stats v1 supports dataset_factory_type='default' only; "
                f"got {factory_type!r}."
            )
        if getattr(data_config, "mode", None) not in ("single_turn", "trajectory"):
            raise ValueError(
                "DeltaCAT sampled stats supports data.mode in {single_turn, trajectory} only."
            )
        if len(resolved_sources) != 1:
            raise _DeltaCatStatsPreflightNotSupported(
                "config-level preflight currently requires one resolved dataset source "
                f"per dataset entry; got {len(resolved_sources)} for embodiment {embodiment!r}"
            )
        source = resolved_sources[0]
        if not _is_deltacat_source(source):
            continue

        deltacat_config = getattr(dataset_spec, "deltacat", None)
        specs.append(
            _DeltaCatStatsStratumSpec(
                dataset_path=str(source.uri),
                embodiment=embodiment,
                modality_configs=data_config.modality_configs[embodiment],
                weight=float(getattr(dataset_spec, "mix_ratio", 1.0)),
                episode_query=getattr(deltacat_config, "episode_query", None),
                steps_query=getattr(deltacat_config, "steps_query", None),
                as_of_timestamp=getattr(deltacat_config, "as_of_timestamp", None),
                key_remap=dict(getattr(deltacat_config, "key_remap", {}) or {}),
                shard_size=int(getattr(data_config, "shard_size", 2**10)),
                episode_sampling_rate=float(getattr(data_config, "episode_sampling_rate", 0.1)),
                seed=int(getattr(data_config, "seed", 42)),
                allow_padding=bool(getattr(data_config, "allow_padding", False)),
                video_backend=str(getattr(data_config, "video_backend", "torchcodec")),
                resample_on_invalid_frames=bool(
                    getattr(data_config, "resample_on_invalid_frames", False)
                ),
            )
        )

    for embodiment, source_types in sources_by_emb.items():
        if "deltacat" in source_types and "other" in source_types:
            raise ValueError(
                "DeltaCAT sampled stats v1 does not support mixing DeltaCAT and "
                f"non-DeltaCAT datasets for the same embodiment {embodiment!r}."
            )
    return specs


def _is_deltacat_source(source: source_resolver.ResolvedDatasetSource) -> bool:
    return source.format == "deltacat" or str(source.uri).startswith("dc://")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    """Compute exact weighted quantiles independently for every feature dimension."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must have the same first dimension")
    if values.shape[0] == 0:
        raise ValueError("cannot compute a quantile over zero samples")
    if np.any(weights < 0) or float(weights.sum()) <= 0.0:
        raise ValueError("sample weights must be non-negative and sum to a positive value")

    flat = values.reshape(values.shape[0], -1)
    out = np.empty(flat.shape[1], dtype=np.float64)
    for dim in range(flat.shape[1]):
        order = np.argsort(flat[:, dim], kind="stable")
        sorted_values = flat[order, dim]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        idx = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
        out[dim] = sorted_values[min(idx, len(sorted_values) - 1)]
    return out.reshape(values.shape[1:])


def _run_preflight_specs(
    specs: list[_DeltaCatStatsStratumSpec],
    *,
    stats_config: Any,
    default_seed: int,
    allow_ray: bool,
) -> dict[str, dict[str, Any]]:
    tasks = _preflight_tasks(specs, stats_config=stats_config, default_seed=default_seed)
    use_ray = allow_ray and _ray_is_initialized()
    _log_info(
        "DeltaCAT sampled stats config preflight mode=%s tasks=%d",
        "ray" if use_ray else "inline",
        len(tasks),
    )

    if use_ray:
        samples = _run_ray_preflight_tasks(tasks)
    else:
        samples = [_compute_stratum_samples_worker(task) for task in tasks]
    return _stats_from_stratum_samples(samples)


def _preflight_tasks(
    specs: list[_DeltaCatStatsStratumSpec],
    *,
    stats_config: Any,
    default_seed: int,
) -> list[dict[str, Any]]:
    seed = getattr(stats_config, "seed", None)
    rng = np.random.default_rng(default_seed if seed is None else seed)
    specs_by_embodiment: dict[str, list[_DeltaCatStatsStratumSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_embodiment[spec.embodiment].append(spec)

    tasks: list[dict[str, Any]] = []
    for embodiment, emb_specs in sorted(specs_by_embodiment.items()):
        weights = np.asarray([max(0.0, spec.weight) for spec in emb_specs], dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(emb_specs), dtype=np.float64)
        masses = weights / weights.sum()
        total_target, min_per_stratum = _sampling_params_for_embodiment(stats_config, embodiment)
        sample_targets = _allocate_sample_counts(masses, total_target, min_per_stratum)
        for source_index, (spec, target_count, mass) in enumerate(
            zip(emb_specs, sample_targets, masses, strict=True)
        ):
            tasks.append(
                {
                    "spec": spec,
                    "target_count": int(target_count),
                    "target_mass": float(mass),
                    "rng_seed": int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
                    "batch_episode_count": _LOWDIM_READ_EPISODE_BATCH_SIZE,
                    "source_index": source_index,
                }
            )
    return tasks


def _ray_is_initialized() -> bool:
    try:
        import ray
    except ImportError:
        return False
    return bool(ray.is_initialized())


def _run_ray_preflight_tasks(tasks: list[dict[str, Any]]) -> list[_DeltaCatStratumSamples]:
    import ray

    remote_fn = ray.remote(num_cpus=1, num_gpus=0)(_compute_stratum_samples_worker)
    refs = [remote_fn.remote(task) for task in tasks]
    return list(ray.get(refs))


def _compute_stratum_samples_worker(task: dict[str, Any]) -> _DeltaCatStratumSamples:
    spec: _DeltaCatStatsStratumSpec = task["spec"]
    from groot.core.data.dataset import sharded_single_step_dataset

    dataset = sharded_single_step_dataset.ShardedSingleStepDataset(
        dataset_path=spec.dataset_path,
        embodiment_tag=schema.EmbodimentTag(spec.embodiment),
        modality_configs=spec.modality_configs,
        video_backend=spec.video_backend,
        resample_on_invalid_frames=spec.resample_on_invalid_frames,
        shard_size=spec.shard_size,
        episode_sampling_rate=spec.episode_sampling_rate,
        seed=spec.seed,
        allow_padding=spec.allow_padding,
        data_filter=None,
        episode_loader_kwargs={
            "episode_query": spec.episode_query,
            "steps_query": spec.steps_query,
            "as_of_timestamp": spec.as_of_timestamp,
            "key_remap": spec.key_remap,
        },
    )
    source = _make_deltacat_source(dataset, spec.weight)
    return _sample_source_to_accumulators(
        source,
        target_count=int(task["target_count"]),
        mass=float(task["target_mass"]),
        rng=np.random.default_rng(int(task["rng_seed"])),
        batch_episode_count=int(task["batch_episode_count"]),
        source_index=int(task["source_index"]),
    )


def compute_from_datasets(
    datasets: list[types.ShardedDataset],
    weights: list[float],
    processor: types.BaseProcessor,
    *,
    data_config: Any,
) -> dict[str, dict[str, Any]]:
    """Compute sampled q01/q99 stats from concrete training datasets.

    This fallback is used when config-level preflight cannot exactly represent
    the training dataset construction. It returns stats only; callers install the
    result onto datasets separately.
    """
    _validate_no_mixed_source_embodiments(datasets)
    sources = _collect_deltacat_sources(datasets, weights)
    if not sources:
        return {}
    _validate_percentile_processor(sources, processor)

    stats_config = data_config.deltacat_stats
    default_seed = data_config.seed
    seed = getattr(stats_config, "seed", None)
    rng = np.random.default_rng(default_seed if seed is None else seed)
    sources_by_emb: dict[str, list[_DeltaCatStatsSource]] = defaultdict(list)
    for source in sources:
        sources_by_emb[source.embodiment].append(source)

    stats_by_embodiment: dict[str, dict[str, Any]] = {}
    for embodiment, emb_sources in sorted(sources_by_emb.items()):
        stats = _compute_embodiment_stats(
            embodiment,
            emb_sources,
            stats_config=stats_config,
            rng=rng,
        )
        stats_by_embodiment[embodiment] = stats
    return stats_by_embodiment


def _compute_embodiment_stats(
    embodiment: str,
    sources: list[_DeltaCatStatsSource],
    *,
    stats_config: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    weights = np.asarray([max(0.0, source.weight) for source in sources], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(sources), dtype=np.float64)
    masses = weights / weights.sum()

    total_target, min_per_stratum = _sampling_params_for_embodiment(stats_config, embodiment)
    batch_episode_count = _LOWDIM_READ_EPISODE_BATCH_SIZE
    sample_targets = _allocate_sample_counts(masses, total_target, min_per_stratum)

    samples: list[_DeltaCatStratumSamples] = []
    for source_index, (source, target_count, mass) in enumerate(
        zip(sources, sample_targets, masses, strict=True)
    ):
        samples.append(
            _sample_source_to_accumulators(
                source,
                target_count=int(target_count),
                mass=float(mass),
                rng=rng,
                batch_episode_count=batch_episode_count,
                source_index=source_index,
            )
        )
    return _stats_from_stratum_samples(samples)[embodiment]


def _sample_source_to_accumulators(
    source: _DeltaCatStatsSource,
    *,
    target_count: int,
    mass: float,
    rng: np.random.Generator,
    batch_episode_count: int,
    source_index: int,
) -> _DeltaCatStratumSamples:
    anchors = _dataset_anchor_pairs(source.dataset)
    if not anchors:
        raise ValueError(
            f"DeltaCAT dataset {source.dataset_path!r} has zero valid anchor steps "
            f"after episode/step filters for embodiment {source.embodiment!r}."
        )
    selected = _sample_anchors(anchors, target_count, rng)
    state_acc: dict[str, _WeightedSamples] = {}
    action_acc: dict[str, _WeightedSamples] = {}
    relative_acc: dict[str, _WeightedSamples] = {}
    metadata = {
        "table_uri": source.dataset_path,
        **source.metadata,
        "target_mass": float(mass),
        "valid_anchor_count": len(anchors),
        "sampled_anchor_count": len(selected),
    }
    if selected:
        _accumulate_source_samples(
            source.dataset,
            selected,
            per_anchor_weight=float(mass) / len(selected),
            batch_episode_count=batch_episode_count,
            state_acc=state_acc,
            action_acc=action_acc,
            relative_acc=relative_acc,
            context=(
                f"table={source.dataset_path!r}, embodiment={source.embodiment!r}, "
                f"stratum_index={source_index}"
            ),
        )
    return _DeltaCatStratumSamples(
        embodiment=source.embodiment,
        metadata=metadata,
        state=state_acc,
        action=action_acc,
        relative_action=relative_acc,
    )


def _stats_from_stratum_samples(
    samples: list[_DeltaCatStratumSamples],
) -> dict[str, dict[str, Any]]:
    samples_by_embodiment: dict[str, list[_DeltaCatStratumSamples]] = defaultdict(list)
    for sample in samples:
        samples_by_embodiment[sample.embodiment].append(sample)

    stats_by_embodiment: dict[str, dict[str, Any]] = {}
    for embodiment, embodiment_samples in sorted(samples_by_embodiment.items()):
        state_acc: dict[str, _WeightedSamples] = {}
        action_acc: dict[str, _WeightedSamples] = {}
        relative_acc: dict[str, _WeightedSamples] = {}
        metadata: dict[str, Any] = {"embodiment": embodiment, "strata": []}
        for sample in embodiment_samples:
            metadata["strata"].append(sample.metadata)
            _merge_accumulators(state_acc, sample.state)
            _merge_accumulators(action_acc, sample.action)
            _merge_accumulators(relative_acc, sample.relative_action)

        metadata["sample_counts"] = {
            "state": _accumulator_counts(state_acc),
            "action": _accumulator_counts(action_acc),
            "relative_action": _accumulator_counts(relative_acc),
        }
        context = f"embodiment={embodiment!r}, metadata={metadata!r}"
        logger.info("DeltaCAT sampled stats metadata: %s", metadata)
        min_samples = 1
        stats_by_embodiment[embodiment] = {
            "state": _stats_dict(state_acc, min_samples, context),
            "action": _stats_dict(action_acc, min_samples, context),
            "relative_action": _stats_dict(relative_acc, min_samples, context),
        }
    return stats_by_embodiment


def _merge_accumulators(
    target: dict[str, _WeightedSamples],
    source: dict[str, _WeightedSamples],
) -> None:
    for key, samples in source.items():
        target.setdefault(key, _WeightedSamples(samples.key)).extend(samples)


def _sampling_params_for_embodiment(stats_config: Any, embodiment: str) -> tuple[int, int]:
    total_target = int(getattr(stats_config, "sample_anchor_steps_per_embodiment", 50_000))
    min_per_stratum = int(getattr(stats_config, "min_anchor_steps_per_stratum", 1_024))
    overrides = getattr(stats_config, "embodiments", {}) or {}
    override = overrides.get(embodiment)
    if override is not None:
        total_override = getattr(override, "sample_anchor_steps_per_embodiment", None)
        min_override = getattr(override, "min_anchor_steps_per_stratum", None)
        if total_override is not None:
            total_target = int(total_override)
        if min_override is not None:
            min_per_stratum = int(min_override)
    return total_target, min_per_stratum


def _allocate_sample_counts(
    masses: np.ndarray,
    total_target: int,
    min_per_stratum: int,
) -> np.ndarray:
    proportional = np.ceil(masses * max(0, total_target)).astype(int)
    floors = np.full_like(proportional, max(0, min_per_stratum))
    return np.maximum(proportional, floors)


def _dataset_anchor_pairs(dataset: Any) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    for shard in getattr(dataset, "sharded_episodes", []):
        for episode_index, step_indices in shard:
            for step_index in step_indices:
                anchors.append((int(episode_index), int(step_index)))
    return anchors


def _sample_anchors(
    anchors: list[tuple[int, int]],
    target_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if target_count >= len(anchors):
        selected = list(anchors)
        rng.shuffle(selected)
        return selected
    indices = rng.choice(len(anchors), size=max(0, int(target_count)), replace=False)
    return [anchors[int(index)] for index in indices]


def _accumulate_source_samples(
    dataset: Any,
    selected_anchors: list[tuple[int, int]],
    *,
    per_anchor_weight: float,
    batch_episode_count: int,
    state_acc: dict[str, _WeightedSamples],
    action_acc: dict[str, _WeightedSamples],
    relative_acc: dict[str, _WeightedSamples],
    context: str,
) -> None:
    loader = getattr(dataset, "episode_loader", None)
    reader = getattr(loader, "_reader", None)
    if reader is None:
        raise TypeError(f"DeltaCAT sampled stats expected a PackDSStepsReader: {context}")

    modality_configs = dataset.modality_configs
    columns = _lowdim_columns(modality_configs)
    anchors_by_episode: dict[int, list[int]] = defaultdict(list)
    for episode_index, step_index in selected_anchors:
        anchors_by_episode[episode_index].append(step_index)

    episode_indices = list(anchors_by_episode)
    for batch_start in range(0, len(episode_indices), max(1, batch_episode_count)):
        batch_indices = episode_indices[batch_start : batch_start + max(1, batch_episode_count)]
        episode_ids = [
            str(loader.episodes_metadata[episode_index]["trajectory_id"]) for episode_index in batch_indices
        ]
        try:
            _plan, lf = reader.lowdim_lazyframe_for_episode_ids(episode_ids, columns)
            steps_df = lf.collect()
        except Exception as exc:
            raise RuntimeError(
                "DeltaCAT low-dimensional sampled-stats read failed. "
                "If steps_query references a column that is not state/action or a "
                "simple predicate identifier, it must be rewritten for v1 sampled stats. "
                f"{context}; requested_columns={columns}; steps_query={getattr(reader, 'steps_query', None)!r}"
            ) from exc

        rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in steps_df.to_dicts():
            rows_by_episode[str(row["episode_id"])].append(row)
        for rows in rows_by_episode.values():
            rows.sort(key=lambda row: int(row["step_index"]))

        for episode_index, episode_id in zip(batch_indices, episode_ids, strict=True):
            rows = rows_by_episode.get(episode_id, [])
            if not rows:
                raise ValueError(
                    "DeltaCAT sampled stats selected an episode with no lowdim rows. "
                    f"episode_id={episode_id!r}; {context}"
                )
            for anchor_step in anchors_by_episode[episode_index]:
                _accumulate_anchor(
                    rows,
                    anchor_step,
                    modality_configs,
                    per_anchor_weight=per_anchor_weight,
                    state_acc=state_acc,
                    action_acc=action_acc,
                    relative_acc=relative_acc,
                    context=f"{context}; episode_id={episode_id!r}; anchor_step={anchor_step}",
                )


def _lowdim_columns(modality_configs: dict[str, types.ModalityConfig]) -> list[str]:
    columns: list[str] = []
    state_config = modality_configs.get("state")
    if state_config is not None:
        columns.extend(f"state:{key}" for key in state_config.modality_keys)
    action_config = modality_configs.get("action")
    if action_config is not None:
        columns.extend(f"action:{key}" for key in action_config.modality_keys)
        if action_config.action_configs is not None:
            for action_key, config in zip(
                action_config.modality_keys,
                action_config.action_configs,
                strict=True,
            ):
                if config.rep == types.ActionRepresentation.RELATIVE:
                    columns.append(f"state:{config.state_key or action_key}")
    return list(dict.fromkeys(columns))


def _accumulate_anchor(
    rows: list[dict[str, Any]],
    anchor_step: int,
    modality_configs: dict[str, types.ModalityConfig],
    *,
    per_anchor_weight: float,
    state_acc: dict[str, _WeightedSamples],
    action_acc: dict[str, _WeightedSamples],
    relative_acc: dict[str, _WeightedSamples],
    context: str,
) -> None:
    state_config = modality_configs.get("state")
    if state_config is not None:
        state_weight = per_anchor_weight / max(1, len(state_config.delta_indices))
        for key in state_config.modality_keys:
            acc = state_acc.setdefault(key, _WeightedSamples(f"state.{key}"))
            for delta in state_config.delta_indices:
                acc.add(_row_value(rows, anchor_step, delta, f"state:{key}"), state_weight)

    action_config = modality_configs.get("action")
    if action_config is None:
        return

    action_weight = per_anchor_weight / max(1, len(action_config.delta_indices))
    action_configs = action_config.action_configs or [None] * len(action_config.modality_keys)
    for key, key_action_config in zip(
        action_config.modality_keys,
        action_configs,
        strict=True,
    ):
        raw_acc = action_acc.setdefault(key, _WeightedSamples(f"action.{key}"))
        action_rows = [
            _row_value(rows, anchor_step, delta, f"action:{key}") for delta in action_config.delta_indices
        ]
        for value in action_rows:
            raw_acc.add(value, action_weight)

        if key_action_config is None or key_action_config.rep != types.ActionRepresentation.RELATIVE:
            continue
        state_key = key_action_config.state_key or key
        ref_state = _row_value(rows, anchor_step, 0, f"state:{state_key}")
        if ref_state is None or any(value is None for value in action_rows):
            relative_acc.setdefault(key, _WeightedSamples(f"relative_action.{key}")).null_count += 1
            continue
        try:
            relative_actions = data_stats.compute_relative_action(
                np.stack([np.asarray(value, dtype=np.float32) for value in action_rows], axis=0),
                np.asarray(ref_state, dtype=np.float32),
                key_action_config.type,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to compute relative action stats: {context}") from exc
        relative_acc.setdefault(key, _WeightedSamples(f"relative_action.{key}")).add(
            relative_actions,
            per_anchor_weight,
        )


def _row_value(
    rows: list[dict[str, Any]],
    anchor_step: int,
    delta: int,
    column: str,
) -> Any:
    idx = max(0, min(anchor_step + int(delta), len(rows) - 1))
    return rows[idx].get(column)


def _stats_dict(
    accumulators: dict[str, _WeightedSamples],
    min_non_null_samples: int,
    context: str,
) -> dict[str, Any]:
    return {
        key: acc.to_stats(min_non_null_samples=min_non_null_samples, context=context)
        for key, acc in accumulators.items()
    }


def _accumulator_counts(accumulators: dict[str, _WeightedSamples]) -> dict[str, dict[str, int]]:
    return {
        key: {
            "non_null": acc.count,
            "null": acc.null_count,
            "dropped": acc.drop_count,
        }
        for key, acc in accumulators.items()
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=0, weights=weights)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = _weighted_mean(values, weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    return np.sqrt(np.maximum(variance, 0.0))


def _validate_percentile_processor(
    sources: list[_DeltaCatStatsSource],
    processor: types.BaseProcessor,
) -> None:
    needs_stats = any(_dataset_has_lowdim(source.dataset) for source in sources)
    if not needs_stats:
        return
    state_action_processor = getattr(processor, "state_action_processor", None)
    if state_action_processor is None:
        return
    if not bool(getattr(state_action_processor, "use_percentiles", False)):
        paths = [source.dataset_path for source in sources]
        raise ValueError(
            "DeltaCAT PackDS state/action training requires use_percentiles=True "
            "so sampled q01/q99 bounds are used for normalization. "
            f"DeltaCAT datasets: {paths}"
        )


def _validate_percentile_model_config(
    specs: list[_DeltaCatStatsStratumSpec],
    model_config: Any | None,
) -> None:
    needs_stats = any(_modality_configs_have_lowdim(spec.modality_configs) for spec in specs)
    if not needs_stats or model_config is None or not hasattr(model_config, "use_percentiles"):
        return
    if not bool(model_config.use_percentiles):
        paths = [spec.dataset_path for spec in specs]
        raise ValueError(
            "DeltaCAT PackDS state/action training requires use_percentiles=True "
            "so sampled q01/q99 bounds are used for normalization. "
            f"DeltaCAT datasets: {paths}"
        )


def _dataset_has_lowdim(dataset: Any) -> bool:
    modality_configs = getattr(dataset, "modality_configs", {}) or {}
    return _modality_configs_have_lowdim(modality_configs)


def _modality_configs_have_lowdim(modality_configs: dict[str, types.ModalityConfig]) -> bool:
    for modality in ("state", "action"):
        config = modality_configs.get(modality)
        if config is not None and config.modality_keys:
            return True
    return False
