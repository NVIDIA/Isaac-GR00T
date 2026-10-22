# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Grouped weighted-sampling dataset factory (v2).

Decouples two concerns:

- **Weighted sampling** (``compute_sampling_weights``): given arbitrary groups with
  target budgets, compute per-dataset sampling weights.  Pure math, no domain knowledge.
- **Grouping** (``DatasetGrouper`` subclasses): classify datasets into groups and
  assign each group a target budget.  Domain-specific, strategy-swappable.

The factory ``DatasetFactoryGroupedSampling`` composes a grouper with the weighted
sampler, plus shared infrastructure (hydrator, dataset building, data filtering).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog
import torch.distributed as dist
from tqdm import tqdm

from groot.core.data import enums, schema
from groot.core.data.dataset import (
    factory,
    sharded_mixture_dataset,
    sharded_mixture_dataset_grouped,
    sharded_single_step_dataset,
    sharded_single_step_dataset_grouped,
    sharded_single_step_rl_dataset,
    sharded_single_step_rl_dataset_grouped,
)
from groot.core.data.dataset.filter import annotation_source, sys2_filter
from groot.core.data.hydrator import service
from groot.core.data.hydrator.config import HydratorConfig
from groot.core.training.configs.data import filter_config
from groot.core.training.configs.data.data_config import DataConfig
from groot.core.utils.distributed.worker_logging import get_unified_logger

if TYPE_CHECKING:
    from groot.core.data import types

HUMAN_ACTION_SOURCE = enums.ActionSourceEnum.HUMAN.value  # 2


def _raise_unsupported_packds_path(dataset_path: object, *, factory_name: str) -> None:
    path = str(dataset_path)
    if factory._is_deltacat_packds_path(path):  # noqa: SLF001 - shared factory helper
        raise ValueError(
            f"{factory_name} does not support DeltaCAT dc:// datasets. "
            "Use dataset_factory_type='default' with data.mode='single_turn'."
        )
    if factory._is_packds_path(path):  # noqa: SLF001 - shared factory helper
        raise ValueError(f"{factory_name} does not support PackDS datasets")


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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetGroup:
    """A named group of datasets with a target sampling budget.

    Attributes:
        name: Human-readable label used in logs and metrics.
        datasets: List of ``(dataset, mix_ratio)`` tuples.  ``mix_ratio`` acts
            as a multiplier on effective step counts *within* this group.
        target_budget: Fraction of total training budget allocated to this group.
            The sum of all groups' budgets should be approximately 1.0.
    """

    name: str
    datasets: list = field(default_factory=list)
    target_budget: float = 0.0
    # Per-dataset group weight maps for rejection sampling (one per dataset in this group).
    # None -> no rejection sampling for this group's datasets.
    group_weight_map: dict[int, float] | None = None
    # Per-dataset group label histograms {group_label: step_count} for acceptance rate computation.
    group_histograms: list[dict[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grouping strategy (ABC)
# ---------------------------------------------------------------------------


class DatasetGrouper(ABC):
    """Abstract grouping strategy.

    Subclasses decide:

    - How many groups there are and what they are called.
    - Which datasets belong to which group.
    - What target budget each group receives.

    The factory passes *build_dataset_fn* so the grouper can construct actual
    dataset objects without depending on factory internals.
    """

    @abstractmethod
    def build_groups(
        self,
        dataset_specs: list,
        build_dataset_fn: Callable,
        data_filter: object,
        hydrator_client: object,
        data_config: DataConfig,
    ) -> list[DatasetGroup]:
        """Scan data, classify, compute budgets, and return groups.

        Args:
            dataset_specs: ``data_config.datasets`` - a list of
                ``SingleDatasetConfig`` objects.
            build_dataset_fn: Callback with signature
                ``(dataset_path, dataset_spec, data_filter,
                step_indices_filter=None, hydrator_client=None) -> Dataset``.
            data_filter: Pre-built data filter (or ``None``).
            hydrator_client: Hydrator IPC client (or ``None``).
            data_config: Data config.

        Returns:
            List of :class:`DatasetGroup`.  The sum of ``target_budget`` across
            all returned groups should be approximately 1.0.
        """


# ---------------------------------------------------------------------------
# Factory (composes grouper + weighted sampling)
# ---------------------------------------------------------------------------


class DatasetFactoryGroupedSampling:
    """Compose a :class:`DatasetGrouper` with :meth:`compute_sampling_weights`.

    Usage::

        grouper = HumanPolicyGrouper()  # or any DatasetGrouper subclass
        fac = DatasetFactoryGroupedSampling(data_config, grouper)
        train_ds, _ = fac.build(processor)
    """

    def __init__(
        self,
        data_config: DataConfig,
        grouper: DatasetGrouper,
        experiment_name: str | None = None,
        use_relative_action: bool | None = None,
        eval_strategy: str = "no",
    ):
        self.data_config = data_config
        self.grouper = grouper
        from groot.core.data.dataset.factory import _derive_experiment_name

        self.experiment_name = _derive_experiment_name(experiment_name)
        self.use_relative_action = use_relative_action
        self.eval_strategy = eval_strategy

    # -- Weighted sampling --------------------------------------------------

    def compute_sampling_weights(
        self,
        groups: list[DatasetGroup],
    ) -> tuple[
        list, list[float], list[int], list[dict[int, float] | None], list[dict[int, int] | None]
    ]:
        """Compute per-dataset sampling weights from arbitrary groups.

        For each group the budget is distributed among its datasets
        proportionally to ``steps * mix_ratio``::

            weighted_total = sum(steps_i * ratio_i)  for i in group
            w_i = group.target_budget * (steps_i * ratio_i / weighted_total)

        Returns:
            ``(all_datasets, all_weights, all_group_ids, all_group_weight_maps, all_group_histograms)``
            flattened across all groups.  Weight maps and histograms are ``None``
            per dataset when speedup is off (no rejection sampling).
        """
        all_datasets: list = []
        all_weights: list[float] = []
        all_group_ids: list[int] = []
        all_group_weight_maps: list[dict[int, float] | None] = []
        all_group_histograms: list[dict[int, int] | None] = []

        for group_id, group in enumerate(groups):
            weighted_total = sum(sum(ds.shard_lengths) * ratio for ds, ratio in group.datasets)
            for ds_i, (ds, ratio) in enumerate(group.datasets):
                ds_steps = sum(ds.shard_lengths)
                if weighted_total > 0:
                    w = group.target_budget * (ds_steps * ratio / weighted_total)
                else:
                    w = 0.0
                all_datasets.append(ds)
                all_weights.append(w)
                all_group_ids.append(group_id)
                all_group_weight_maps.append(group.group_weight_map)
                hist = group.group_histograms[ds_i] if ds_i < len(group.group_histograms) else None
                all_group_histograms.append(hist)

        return all_datasets, all_weights, all_group_ids, all_group_weight_maps, all_group_histograms

    # -- Main entry point ---------------------------------------------------

    def build(
        self,
        processor: "types.BaseProcessor",
    ) -> tuple[
        sharded_mixture_dataset.ShardedMixtureDataset,
        sharded_mixture_dataset.ShardedMixtureDataset | None,
    ]:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        assert self.eval_strategy == "no", "Sharded dataset does not support evaluation sets"

        hydrator_enabled = any(
            factory._resolve_one_path(str(p)).startswith(  # noqa: SLF001 - shared factory helper
                "s3://"
            )
            for spec in self.data_config.datasets
            for p in spec.dataset_paths
            if not factory._is_packds_path(str(p))  # noqa: SLF001 - shared factory helper
        )
        if hydrator_enabled:
            get_unified_logger().info(
                "[RANK %s] DatasetFactoryGroupedSampling.build: creating hydrator ...",
                rank,
            )
            hydrator = self._create_hydrator()
            self._hydrator = hydrator
            hydrator_client = hydrator.get_ipc_client()
        else:
            hydrator = None
            hydrator_client = None

        factory._generate_all_stats(  # noqa: SLF001 - shared factory helper
            self.data_config, hydrator, self.use_relative_action
        )
        data_filter = self._build_data_filter()

        groups = self.grouper.build_groups(
            dataset_specs=self.data_config.datasets,
            build_dataset_fn=self._build_single_turn_dataset,
            data_filter=data_filter,
            hydrator_client=hydrator_client,
            data_config=self.data_config,
        )

        all_datasets, all_weights, all_group_ids, all_group_weight_maps, all_group_histograms = (
            self.compute_sampling_weights(groups)
        )

        # -- Logging --------------------------------------------------------
        speedup = getattr(self.data_config, "quality_weighted_speedup", False)
        presample = getattr(self.data_config, "quality_weighted_presample", False)
        if speedup and presample:
            mode_str = "SPEEDUP+PRESAMPLE (merge+subsample, zero I/O waste)"
        elif speedup:
            mode_str = "SPEEDUP (merge+rejection)"
        else:
            mode_str = "DEFAULT (per-priority datasets)"
        print(f"[V2_SAMPLING] mode={mode_str}")  # noqa: T201 - live sampling diagnostics
        print(  # noqa: T201 - live sampling diagnostics
            f"[V2_SAMPLING] n_groups={len(groups)} n_datasets={len(all_datasets)}"
        )
        for group in groups:
            total_steps = sum(sum(ds.shard_lengths) for ds, _ in group.datasets)
            print(  # noqa: T201 - live sampling diagnostics
                f"[V2_SAMPLING] group='{group.name}' "
                f"n_datasets={len(group.datasets)} total_steps={total_steps} "
                f"budget={group.target_budget:.6f} "
                f"weight_map={group.group_weight_map}"
            )
        for i, (ds, w) in enumerate(zip(all_datasets, all_weights, strict=False)):
            ds_path = getattr(ds, "dataset_path", f"idx={i}")
            ds_steps = sum(ds.shard_lengths)
            n_shards = len(ds)
            print(  # noqa: T201 - live sampling diagnostics
                f"[V2_SAMPLING] ds[{i}] path={ds_path} "
                f"steps={ds_steps} n_shards={n_shards} "
                f"weight={w:.6f} group_id={all_group_ids[i]} "
                f"weight_map={all_group_weight_maps[i]} "
                f"histogram={all_group_histograms[i]}"
            )
        print(  # noqa: T201 - live sampling diagnostics
            f"[V2_SAMPLING] weights_sum={sum(all_weights):.6f}"
        )

        self.sampling_summary: dict[str, float | int] = {}
        for group in groups:
            total_steps = sum(sum(ds.shard_lengths) for ds, _ in group.datasets)
            self.sampling_summary[f"grouped_sampling/{group.name}_total_steps"] = total_steps
            self.sampling_summary[f"grouped_sampling/{group.name}_budget"] = group.target_budget

        # Use grouped mixture subclass when rejection sampling is active (speedup path)
        has_rejection = any(m is not None for m in all_group_weight_maps)

        if has_rejection:
            mixture_ds = sharded_mixture_dataset_grouped.ShardedMixtureDatasetGrouped(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.data_config.seed,
                training=True,
                num_shards_per_epoch=self.data_config.num_shards_per_epoch,
                override_pretraining_statistics=self.data_config.override_pretraining_statistics,
                group_ids=all_group_ids,
                per_dataset_group_weight_maps=all_group_weight_maps,
                per_dataset_group_histograms=all_group_histograms,
            )
        else:
            mixture_ds = sharded_mixture_dataset.ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.data_config.seed,
                training=True,
                num_shards_per_epoch=self.data_config.num_shards_per_epoch,
                override_pretraining_statistics=self.data_config.override_pretraining_statistics,
            )

        self._train_dataset = mixture_ds
        return mixture_ds, None

    # -- Shared infrastructure (same as old factory) ------------------------

    def _create_hydrator(self) -> service.Hydrator:
        worker_id = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        job_name = self.experiment_name
        hydrator_config = getattr(self.data_config, "hydrator_config", None) or HydratorConfig()
        return service.Hydrator(job_name=job_name, worker_id=worker_id, config=hydrator_config)

    def _build_single_turn_dataset(
        self,
        dataset_path,
        dataset_spec,
        data_filter,
        step_indices_filter=None,
        step_group_labels=None,
        hydrator_client=None,
    ) -> sharded_single_step_dataset.ShardedSingleStepDataset:
        embodiment_tag = dataset_spec.embodiment_tag
        assert embodiment_tag is not None, "embodiment_tag is required"
        _raise_unsupported_packds_path(
            dataset_path,
            factory_name="Grouped sampling factory",
        )

        # Choose grouped subclass when step_group_labels is provided (speedup path),
        # otherwise use the base class (identical to postTrain_main behavior).
        if step_group_labels is not None:
            if self.data_config.mode == "single_turn":
                dataset_class = sharded_single_step_dataset_grouped.ShardedSingleStepDatasetGrouped
            elif self.data_config.mode == "single_turn_rl":
                dataset_class = (
                    sharded_single_step_rl_dataset_grouped.ShardedSingleStepRLDatasetGrouped
                )
            else:
                raise ValueError(
                    f"Grouped sampling factory does not support mode: {self.data_config.mode}"
                )
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
                step_group_labels=step_group_labels,
            )
        else:
            if self.data_config.mode == "single_turn":
                dataset_class = sharded_single_step_dataset.ShardedSingleStepDataset
            elif self.data_config.mode == "single_turn_rl":
                dataset_class = sharded_single_step_rl_dataset.ShardedSingleStepRLDataset
            else:
                raise ValueError(
                    f"Grouped sampling factory does not support mode: {self.data_config.mode}"
                )
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
        """Shutdown background dataset workers and the hydrator."""
        train_dataset = getattr(self, "_train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "shutdown"):
            train_dataset.shutdown(wait=True)
            self._train_dataset = None

        hydrator = getattr(self, "_hydrator", None)
        if hydrator is not None:
            hydrator.shutdown()


# ---------------------------------------------------------------------------
# Helper functions for action-source classification
# ---------------------------------------------------------------------------


def _classifier_progress_step(total: int) -> int:
    """Pick a sensible cadence for progress logs (~20 lines per dataset, min 1)."""
    return max(1, total // 20)


def _log_classifier_progress(
    grouper: str,
    dataset_path: str | Path,
    processed: int,
    total: int,
    started_at: float,
) -> None:
    """Emit a single progress line: ``[grouper] dataset N/M (PCT%) elapsed=Xs eta=Ys``."""
    elapsed = time.perf_counter() - started_at
    pct = (100.0 * processed / total) if total else 100.0
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 else float("inf")
    print(  # noqa: T201 - live sampling diagnostics
        f"[{grouper}] {dataset_path} {processed}/{total} ({pct:.1f}%) "
        f"elapsed={elapsed:.1f}s rate={rate:.2f} eps/s eta={eta:.1f}s",
        flush=True,
    )


_ACTION_SOURCE_CANDIDATES: tuple[str, ...] = ("rl_info.action_source_mask", "action.source")
_QUALITY_CANDIDATES: tuple[str, ...] = (
    "rl_info.predicted_action_quality_mask",
    "rl_info.action_quality_mask",
)


def _coerce_column_to_int64(values: np.ndarray) -> np.ndarray:
    """Apply the same reshaping/casting both column readers used to do per-call."""
    if values.ndim > 1:
        values = values[:, 0]
    elif values.dtype == object:
        values = np.array([int(v) if np.isscalar(v) else int(v[0]) for v in values])
    return values.astype(np.int64)


def _read_classifier_columns(
    parquet_path: str | Path,
    hydrator: object | None = None,
    *,
    want_source: bool = True,
    want_quality: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (action_source, quality) arrays via column-projected parquet reads.

    For ``s3://`` paths, opens the parquet footer via the gear-configured pyarrow
    S3 filesystem, intersects ``schema.names`` with the candidate columns, and
    reads only the present columns in one ``pq.read_table`` call. This skips the
    whole-file fetch ``Hydrator.hydrate`` would do — small projected byte ranges
    instead of the full parquet body.

    For local paths, uses ``pd.read_parquet(path, columns=[...])`` which already
    pushes column projection into the parquet reader.

    Either column may be missing in the schema; that candidate simply returns
    ``None`` instead of raising.
    """
    parquet_path_str = str(parquet_path)
    candidates: list[str] = []
    if want_source:
        candidates.extend(_ACTION_SOURCE_CANDIDATES)
    if want_quality:
        candidates.extend(_QUALITY_CANDIDATES)
    if not candidates:
        return None, None

    if parquet_path_str.startswith("s3://"):
        if hydrator is None:
            return None, None
        from groot.core.data.hydrator import data_source_resolver as _dsr
        from groot.core.data.hydrator.utils import aws as _aws
        from groot.core.data.hydrator.utils import io as _io

        try:
            profile = _dsr.default_resolver().get_profile_for_uri(parquet_path_str) or "default"
            bucket, key = _io.parse_s3_uri(parquet_path_str)
            s3_fs = _aws.get_pyarrow_s3_fs(profile=profile, bucket=bucket)
            s3_path = f"{bucket}/{key}"
            schema = pq.read_schema(s3_path, filesystem=s3_fs)
            present = [c for c in candidates if c in schema.names]
            if not present:
                return None, None
            table = pq.read_table(s3_path, filesystem=s3_fs, columns=present)
        except Exception:  # noqa: BLE001
            return None, None
        present_set = set(table.column_names)
    else:
        try:
            schema = pq.read_schema(str(parquet_path))
            present = [c for c in candidates if c in schema.names]
            if not present:
                return None, None
            raw_df = pd.read_parquet(parquet_path, columns=present)
            table = None  # use raw_df path below
        except Exception:  # noqa: BLE001
            return None, None
        present_set = set(raw_df.columns)

    def _extract(col_candidates: tuple[str, ...]) -> np.ndarray | None:
        for col_name in col_candidates:
            if col_name not in present_set:
                continue
            if table is not None:
                raw = table[col_name].to_pandas().to_numpy()
            else:
                raw = raw_df[col_name].to_numpy()
            return _coerce_column_to_int64(raw)
        return None

    action_source = _extract(_ACTION_SOURCE_CANDIDATES) if want_source else None
    quality = _extract(_QUALITY_CANDIDATES) if want_quality else None
    return action_source, quality


def _read_population_classifier_columns(dataset_path, info, episodes_meta, hydrator):
    """Scan episode labels together, retaining file boundaries and frame order."""
    if not episodes_meta:
        return {}
    total = len(episodes_meta)
    progress_every = _classifier_progress_step(total)

    def report(stage, processed, started):
        if processed % progress_every == 0 or processed == total:
            _log_classifier_progress(
                f"QualityPriorityGrouper:{stage}", dataset_path, processed, total, started
            )

    started = time.perf_counter()
    report("metadata", 0, started)
    paths = [
        str(
            _resolve_parquet_path(
                dataset_path,
                info["data_path"].format(
                    episode_chunk=episode["episode_index"] // info["chunks_size"],
                    episode_index=episode["episode_index"],
                ),
            )
        )
        for episode in episodes_meta
    ]
    filesystem = pafs.LocalFileSystem()
    if paths[0].startswith("s3://"):
        if hydrator is None:
            raise ValueError("Hydrator required to resolve S3 classifier storage")
        from groot.core.data.hydrator import data_source_resolver as _dsr
        from groot.core.data.hydrator.utils import aws as _aws
        from groot.core.data.hydrator.utils import io as _io

        profile = _dsr.default_resolver().get_profile_for_uri(paths[0]) or "default"
        bucket, _ = _io.parse_s3_uri(paths[0])
        filesystem = _aws.get_pyarrow_s3_fs(profile=profile, bucket=bucket)
        paths = [path.removeprefix("s3://") for path in paths]

    parquet = pads.ParquetFileFormat()
    fragments = [parquet.make_fragment(path, filesystem=filesystem) for path in paths]
    columns = (*_ACTION_SOURCE_CANDIDATES, *_QUALITY_CANDIDATES, "episode_index", "frame_index")

    def label_schema(fragment):
        # Retain each fragment's footer for the scan; later files may add labels.
        return pa.schema([field for field in fragment.physical_schema if field.name in columns])

    with ThreadPoolExecutor(max_workers=min(32, len(fragments))) as executor:
        futures = {
            executor.submit(label_schema, fragment): index
            for index, fragment in enumerate(fragments)
        }
        schemas = [None] * total
        for processed, future in enumerate(as_completed(futures), start=1):
            schemas[futures[future]] = future.result()
            report("metadata", processed, started)
    if not any(schema.names for schema in schemas):
        return {episode["episode_index"]: (None, None) for episode in episodes_meta}
    try:
        scan_groups = {pa.unify_schemas(schemas, promote_options="permissive"): fragments}
    except pa.ArrowTypeError:
        # Arrow cannot unify scalar and list labels. Batch matching schemas and
        # normalize each episode's labels with the same coercion as the file reader.
        scan_groups = {}
        for schema, fragment in zip(schemas, fragments, strict=True):
            scan_groups.setdefault(schema, []).append(fragment)
    # Old datasets can omit episode/frame columns; their file membership and
    # physical row order remain authoritative. Do not filter lookahead context.
    batches = {}
    remaining_rows = {
        path: episode["length"] for path, episode in zip(paths, episodes_meta, strict=True)
    }
    completed = set()
    started = time.perf_counter()
    report("labels", 0, started)
    for schema, group in scan_groups.items():
        population = pads.FileSystemDataset(group, schema, parquet, filesystem)
        for fragment in group:
            batches[fragment.path] = (schema, [])
        scanner = population.scanner(columns=schema.names, use_threads=True)
        for tagged in scanner.scan_batches():
            path = tagged.fragment.path
            batches[path][1].append(tagged.record_batch)
            remaining_rows[path] -= tagged.record_batch.num_rows
            if remaining_rows[path] <= 0 and path not in completed:
                completed.add(path)
                report("labels", len(completed), started)

    def extract(table, physical_schema, candidates):
        for name in candidates:
            if name in physical_schema.names:
                return _coerce_column_to_int64(table[name].to_numpy())
        return None

    result = {}
    for path, episode, physical_schema in zip(paths, episodes_meta, schemas, strict=True):
        scan_schema, episode_batches = batches[path]
        table = pa.Table.from_batches(episode_batches, schema=scan_schema)
        if len(table) != episode["length"]:
            raise ValueError(f"Classifier row count differs from episode metadata: {path}")
        if "episode_index" in physical_schema.names and not np.all(
            table["episode_index"].to_numpy() == episode["episode_index"]
        ):
            raise ValueError(f"Classifier episode identity differs from metadata: {path}")
        if "frame_index" in physical_schema.names:
            table = table.sort_by("frame_index")
            frames = table["frame_index"].to_numpy()
            # An episode cut from a longer recording keeps the recording's frame numbers
            # (e.g. 6000..9219), so require consecutive frames from the first one.
            if not np.array_equal(frames - frames[:1], np.arange(episode["length"])):
                raise ValueError(f"Classifier frame indices differ from episode metadata: {path}")

        result[episode["episode_index"]] = (
            extract(table, physical_schema, _ACTION_SOURCE_CANDIDATES),
            extract(table, physical_schema, _QUALITY_CANDIDATES),
        )
    return result


def _load_dataset_meta(
    dataset_path: str | Path,
    hydrator: object | None,
) -> tuple[dict, list[dict]]:
    """Load ``meta/info.json`` and ``meta/episodes.jsonl``, S3-aware.

    For ``s3://`` datasets, joins via :meth:`service.Hydrator.join_path` and reads
    via ``hydrator.hydrate`` (mirrors the LeRobotEpisodeLoader pattern). For local
    paths, falls back to ``Path`` + ``open`` so callers without a hydrator (tests,
    pure-local runs) still work.
    """
    dataset_path_str = str(dataset_path)
    if dataset_path_str.startswith("s3://"):
        if hydrator is None:
            raise ValueError(f"Hydrator required to read S3 dataset metadata: {dataset_path_str}")
        info = hydrator.hydrate(service.Hydrator.join_path(dataset_path_str, "meta/info.json"))[0]
        episodes_meta = hydrator.hydrate(
            service.Hydrator.join_path(dataset_path_str, "meta/episodes.jsonl")
        )
        return info, episodes_meta
    meta_dir = Path(dataset_path_str) / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    with open(meta_dir / "episodes.jsonl") as f:
        episodes_meta = [json.loads(line) for line in f]
    return info, episodes_meta


def _resolve_parquet_path(dataset_path: str | Path, parquet_filename: str) -> str | Path:
    """Join a parquet filename to the dataset root, preserving ``s3://`` when present."""
    dataset_path_str = str(dataset_path)
    if dataset_path_str.startswith("s3://"):
        return service.Hydrator.join_path(dataset_path_str, parquet_filename)
    return Path(dataset_path_str) / parquet_filename


def _classify_steps_for_dataset(
    dataset_path: str | Path,
    min_delta_index: int,
    max_delta_index: int,
    lookahead: int,
    allow_padding: bool = False,
    hydrator: object | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Pre-scan parquet files to classify each valid step as human-chunk or policy-chunk.

    A step is a "human chunk" if any of the first ``lookahead`` steps in its
    action horizon window has ``action source == HUMAN``.  Everything else
    is a "policy chunk".

    Returns two dicts mapping episode_index -> np.ndarray of valid step indices
    for human chunks and policy chunks respectively.
    """
    info, episodes_meta = _load_dataset_meta(dataset_path, hydrator)
    data_path_pattern = info["data_path"]
    chunk_size = info["chunks_size"]

    human_indices: dict[int, np.ndarray] = {}
    policy_indices: dict[int, np.ndarray] = {}

    total = len(episodes_meta)
    progress_every = _classifier_progress_step(total)
    started = time.perf_counter()
    for processed, ep_meta in enumerate(episodes_meta, start=1):
        ep_idx = ep_meta["episode_index"]
        ep_len = int(ep_meta["length"])
        valid_steps = _compute_valid_anchor_indices(
            episode_len=ep_len,
            min_delta_index=min_delta_index,
            max_delta_index=max_delta_index,
            allow_padding=allow_padding,
        )
        if len(valid_steps) == 0:
            if processed % progress_every == 0 or processed == total:
                _log_classifier_progress(
                    "HumanPolicyGrouper", dataset_path, processed, total, started
                )
            continue

        chunk_idx = ep_idx // chunk_size
        parquet_filename = data_path_pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        parquet_path = _resolve_parquet_path(dataset_path, parquet_filename)

        action_source, _ = _read_classifier_columns(
            parquet_path, hydrator=hydrator, want_quality=False
        )
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

        if processed % progress_every == 0 or processed == total:
            _log_classifier_progress("HumanPolicyGrouper", dataset_path, processed, total, started)

    return human_indices, policy_indices


# ---------------------------------------------------------------------------
# Concrete grouper: human-policy split by action_source
# ---------------------------------------------------------------------------


class HumanPolicyGrouper(DatasetGrouper):
    """Groups rollout datasets by action source into demo / human-chunk / policy-chunk.

    Behaviour is identical to the legacy ``DatasetFactoryWeightedSampling``.

    Config fields used:
        - ``data_config.weighted_sample_batch_demo_ratio_cap``
        - ``data_config.weighted_sample_batch_rollout_human_target_prob``
        - ``data_config.weighted_sample_rollout_human_chunk_lookahead``
    """

    def build_groups(
        self,
        dataset_specs: list,
        build_dataset_fn: Callable,
        data_filter: object,
        hydrator_client: object,
        data_config: DataConfig,
    ) -> list[DatasetGroup]:
        demo_cap = data_config.weighted_sample_batch_demo_ratio_cap
        human_target = data_config.weighted_sample_batch_rollout_human_target_prob
        lookahead = data_config.weighted_sample_rollout_human_chunk_lookahead

        if demo_cap + human_target > 1.0:
            raise ValueError(
                f"weighted_sample_batch_demo_ratio_cap ({demo_cap}) + "
                f"weighted_sample_batch_rollout_human_target_prob ({human_target}) "
                f"exceeds 1.0 -- no budget left for policy chunks"
            )

        demo_datasets: list[tuple] = []
        human_datasets: list[tuple] = []
        policy_datasets: list[tuple] = []

        for dataset_spec in tqdm(
            dataset_specs,
            total=len(dataset_specs),
            desc="[HumanPolicyGrouper] Initializing datasets",
        ):
            role = getattr(dataset_spec, "dataset_role", "default")
            mix_ratio = getattr(dataset_spec, "mix_ratio", 1.0)

            if role in ("default", "clean_demo"):
                for dataset_path in (
                    factory._resolve_one_path(str(p))  # noqa: SLF001 - shared factory helper
                    for p in dataset_spec.dataset_paths
                ):
                    _raise_unsupported_packds_path(
                        dataset_path,
                        factory_name="HumanPolicyGrouper",
                    )
                    ds = build_dataset_fn(
                        dataset_path,
                        dataset_spec,
                        data_filter,
                        hydrator_client=hydrator_client,
                    )
                    demo_datasets.append((ds, mix_ratio))

            elif role == "rollout":
                for dataset_path in (
                    factory._resolve_one_path(str(p))  # noqa: SLF001 - shared factory helper
                    for p in dataset_spec.dataset_paths
                ):
                    _raise_unsupported_packds_path(
                        dataset_path,
                        factory_name="HumanPolicyGrouper",
                    )
                    embodiment_tag = dataset_spec.embodiment_tag
                    assert embodiment_tag is not None, "Rollout datasets require embodiment_tag"
                    modality_configs = data_config.modality_configs[embodiment_tag]
                    min_delta = min(min(c.delta_indices) for c in modality_configs.values())
                    max_delta = max(max(c.delta_indices) for c in modality_configs.values())

                    print(  # noqa: T201 - live sampling diagnostics
                        f"[HumanPolicyGrouper] Pre-scanning {dataset_path} "
                        f"for action source classification..."
                    )
                    human_step_filter, policy_step_filter = _classify_steps_for_dataset(
                        dataset_path,
                        min_delta,
                        max_delta,
                        lookahead,
                        allow_padding=data_config.allow_padding,
                        hydrator=hydrator_client,
                    )

                    human_count = sum(len(v) for v in human_step_filter.values())
                    policy_count = sum(len(v) for v in policy_step_filter.values())
                    print(  # noqa: T201 - live sampling diagnostics
                        f"[HumanPolicyGrouper]   -> human chunks: {human_count}, "
                        f"policy chunks: {policy_count}"
                    )

                    if human_count > 0:
                        ds_human = build_dataset_fn(
                            dataset_path,
                            dataset_spec,
                            data_filter,
                            step_indices_filter=human_step_filter,
                            hydrator_client=hydrator_client,
                        )
                        human_datasets.append((ds_human, mix_ratio))

                    if policy_count > 0:
                        ds_policy = build_dataset_fn(
                            dataset_path,
                            dataset_spec,
                            data_filter,
                            step_indices_filter=policy_step_filter,
                            hydrator_client=hydrator_client,
                        )
                        policy_datasets.append((ds_policy, mix_ratio))
            else:
                raise ValueError(f"Unknown dataset_role: {role}")

        # -- Compute budgets ------------------------------------------------
        total_demo = sum(sum(ds.shard_lengths) for ds, _ in demo_datasets)
        total_human = sum(sum(ds.shard_lengths) for ds, _ in human_datasets)
        total_policy = sum(sum(ds.shard_lengths) for ds, _ in policy_datasets)
        total_all = total_demo + total_human + total_policy

        if total_all == 0:
            raise ValueError("No data found across all datasets")

        initial_demo_ratio = total_demo / total_all if total_demo > 0 else 0.0
        demo_budget = min(initial_demo_ratio, demo_cap)
        human_budget = human_target
        policy_budget = 1.0 - human_budget - demo_budget

        if policy_budget < 0:
            print(  # noqa: T201 - live sampling diagnostics
                f"[HumanPolicyGrouper] WARNING: policy budget is negative "
                f"({policy_budget:.4f}). Clamping to 0. "
                f"demo_budget={demo_budget:.4f}, human_budget={human_budget:.4f}"
            )
            policy_budget = 0.0

        return [
            DatasetGroup("demo", demo_datasets, demo_budget),
            DatasetGroup("human", human_datasets, human_budget),
            DatasetGroup("policy", policy_datasets, policy_budget),
        ]


# ---------------------------------------------------------------------------
# Helper functions for quality-priority classification
# ---------------------------------------------------------------------------

# Mapping from rl_info quality mask value to base priority contribution.
# quality=0 (clear_positive / normal) -> +2
# quality=1 (slowly_positive / suboptimal) -> +1
# quality=2 (clear_bad / failure) -> +0
_QUALITY_TO_PRIORITY = {0: 2, 1: 1, 2: 0}

# Integer value for "failure" quality (used to gate human upweight).
_FAILURE_QUALITY = 2


def _compute_step_priorities(
    quality: np.ndarray,
    source: np.ndarray,
    allow_upweight_human: bool,
) -> np.ndarray:
    """Compute per-step priority scores (vectorized).

    Priority formula::

        priority = QUALITY_TO_PRIORITY[quality]
        if allow_upweight_human and source == HUMAN and quality != failure:
            priority += 1

    Args:
        quality: int array of quality mask values (0/1/2).
        source: int array of action source values.
        allow_upweight_human: whether to add +1 for human-source steps
            (only when quality is not failure).

    Returns:
        int64 array of priority values.
    """
    priority = np.vectorize(_QUALITY_TO_PRIORITY.get)(quality).astype(np.int64)
    if allow_upweight_human:
        human_non_failure = (source == HUMAN_ACTION_SOURCE) & (quality != _FAILURE_QUALITY)
        priority = priority + human_non_failure.astype(np.int64)
    return priority


def _majority_vote_quality_for_steps(
    quality: np.ndarray,
    valid_steps: np.ndarray,
    lookahead: int,
) -> np.ndarray:
    """Compute majority-vote quality for explicit anchor steps.

    For anchor ``t``, the window is ``quality[t : t + lookahead]`` (clamped to
    array length).  The quality value with the highest count wins.  On ties the
    **worse** quality (higher int value) wins (pessimistic tie-break).

    Args:
        quality: Full-episode quality array (length ``ep_len``).
        valid_steps: Explicit valid anchor step indices.
        lookahead: Window size.  ``<= 1`` means no lookahead (return
            ``quality[valid_steps]`` as-is).

    Returns:
        int64 array of length ``len(valid_steps)`` with voted quality per anchor.
    """
    if lookahead <= 1:
        return quality[valid_steps].copy()

    ep_len = len(quality)
    voted = np.empty(len(valid_steps), dtype=np.int64)

    for i, step in enumerate(valid_steps):
        window_end = min(step + lookahead, ep_len)
        window = quality[step:window_end]
        values, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        # Among values with max count, pick worst (highest value = pessimistic)
        tied_values = values[counts == max_count]
        voted[i] = tied_values.max()

    return voted


def _majority_vote_quality(
    quality: np.ndarray,
    num_anchors: int,
    lookahead: int,
) -> np.ndarray:
    """Compute majority-vote quality for anchors ``0..num_anchors-1``."""
    valid_steps = np.arange(num_anchors, dtype=np.int64)
    return _majority_vote_quality_for_steps(quality, valid_steps, lookahead)


def _classify_steps_by_priority(
    dataset_path: str | Path,
    min_delta_index: int,
    max_delta_index: int,
    allow_upweight_human: bool,
    lookahead: int = 0,
    allow_padding: bool = False,
    hydrator: object | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """Pre-scan parquet files and group valid steps by priority level.

    When ``lookahead > 1``, each anchor's quality is determined by majority
    vote over ``quality[t : t + lookahead]`` (pessimistic tie-break).

    Returns ``{priority: {episode_index: np.ndarray_of_step_indices}}``.
    """
    info, episodes_meta = _load_dataset_meta(dataset_path, hydrator)
    priority_indices: dict[int, dict[int, np.ndarray]] = {}

    total = len(episodes_meta)
    progress_every = _classifier_progress_step(total)
    started = time.perf_counter()
    eligible_episodes = [
        episode
        for episode in episodes_meta
        if len(
            _compute_valid_anchor_indices(
                int(episode["length"]), min_delta_index, max_delta_index, allow_padding
            )
        )
    ]
    labels = _read_population_classifier_columns(dataset_path, info, eligible_episodes, hydrator)
    for processed, ep_meta in enumerate(episodes_meta, start=1):
        ep_idx = ep_meta["episode_index"]
        ep_len = int(ep_meta["length"])
        valid_steps = _compute_valid_anchor_indices(
            episode_len=ep_len,
            min_delta_index=min_delta_index,
            max_delta_index=max_delta_index,
            allow_padding=allow_padding,
        )
        if len(valid_steps) == 0:
            if processed % progress_every == 0 or processed == total:
                _log_classifier_progress(
                    "QualityPriorityGrouper", dataset_path, processed, total, started
                )
            continue

        source, quality = labels.pop(ep_idx)
        if quality is None:
            quality = np.zeros(ep_len, dtype=np.int64)
        if source is None:
            source = np.full(ep_len, HUMAN_ACTION_SOURCE, dtype=np.int64)

        voted_quality = _majority_vote_quality_for_steps(quality, valid_steps, lookahead)
        priorities = _compute_step_priorities(
            voted_quality,
            source[valid_steps],
            allow_upweight_human,
        )

        for p in np.unique(priorities):
            p_int = int(p)
            mask = priorities == p
            steps = valid_steps[mask]
            if len(steps) > 0:
                if p_int not in priority_indices:
                    priority_indices[p_int] = {}
                priority_indices[p_int][ep_idx] = steps

        if processed % progress_every == 0 or processed == total:
            _log_classifier_progress(
                "QualityPriorityGrouper", dataset_path, processed, total, started
            )

    return priority_indices


# ---------------------------------------------------------------------------
# Concrete grouper: quality-priority based
# ---------------------------------------------------------------------------


class QualityPriorityGrouper(DatasetGrouper):
    """Groups datasets by predicted action quality + optional human-source bonus.

    Splits datasets into two families (clean_demo vs rollout/default), then
    within each family groups steps by priority level.  Each
    ``(family, priority)`` pair becomes a ``DatasetGroup`` with budget
    derived from a configurable mapping dict.

    Config fields used:
        - ``data_config.quality_weighted_clean_demo_budget``
        - ``data_config.quality_weighted_allow_upweight_human``
        - ``data_config.quality_weighted_demo_priority_map``
        - ``data_config.quality_weighted_rollout_priority_map``
        - ``data_config.quality_weighted_lookahead``
    """

    def build_groups(
        self,
        dataset_specs: list,
        build_dataset_fn: Callable,
        data_filter: object,
        hydrator_client: object,
        data_config: DataConfig,
    ) -> list[DatasetGroup]:
        speedup = getattr(data_config, "quality_weighted_speedup", False)
        presample = getattr(data_config, "quality_weighted_presample", False)
        if presample and not speedup:
            raise ValueError(
                "quality_weighted_presample=True requires quality_weighted_speedup=True. "
                "Pre-sampling is a sub-option of the speedup path."
            )
        if speedup:
            return self._build_groups_speedup(
                dataset_specs, build_dataset_fn, data_filter, hydrator_client, data_config
            )
        else:
            return self._build_groups_default(
                dataset_specs, build_dataset_fn, data_filter, hydrator_client, data_config
            )

    # -- Default path: one dataset per (family, path, priority) -------------
    # Identical to postTrain_main behavior.

    def _build_groups_default(
        self,
        dataset_specs: list,
        build_dataset_fn: Callable,
        data_filter: object,
        hydrator_client: object,
        data_config: DataConfig,
    ) -> list[DatasetGroup]:
        demo_budget_total = data_config.quality_weighted_clean_demo_budget
        other_budget_total = 1.0 - demo_budget_total
        allow_upweight = data_config.quality_weighted_allow_upweight_human
        demo_map = data_config.quality_weighted_demo_priority_map
        rollout_map = data_config.quality_weighted_rollout_priority_map
        lookahead = data_config.quality_weighted_lookahead
        proportional_budget = getattr(data_config, "quality_weighted_proportional_budget", False)

        # Collect (dataset, mix_ratio) per (family, priority).
        buckets: dict[tuple[str, int], list[tuple]] = {}

        for dataset_spec in tqdm(
            dataset_specs,
            total=len(dataset_specs),
            desc="[QualityPriorityGrouper] Initializing datasets",
        ):
            role = getattr(dataset_spec, "dataset_role", "default")
            mix_ratio = getattr(dataset_spec, "mix_ratio", 1.0)
            family = "demo" if role == "clean_demo" else "rollout"

            for dataset_path in (
                factory._resolve_one_path(str(p))  # noqa: SLF001 - shared factory helper
                for p in dataset_spec.dataset_paths
            ):
                _raise_unsupported_packds_path(
                    dataset_path,
                    factory_name="QualityPriorityGrouper",
                )
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "embodiment_tag is required"
                modality_configs = data_config.modality_configs[embodiment_tag]
                min_delta = min(min(c.delta_indices) for c in modality_configs.values())
                max_delta = max(max(c.delta_indices) for c in modality_configs.values())

                print(  # noqa: T201 - live sampling diagnostics
                    f"[QualityPriorityGrouper] Pre-scanning {dataset_path} "
                    f"(family={family}) for quality-priority classification..."
                )
                priority_step_filters = _classify_steps_by_priority(
                    dataset_path,
                    min_delta,
                    max_delta,
                    allow_upweight,
                    lookahead=lookahead,
                    allow_padding=data_config.allow_padding,
                    hydrator=hydrator_client,
                )

                for priority, ep_step_map in priority_step_filters.items():
                    family_map = demo_map if family == "demo" else rollout_map
                    if family_map.get(priority, 0) <= 0:
                        skip_count = sum(len(v) for v in ep_step_map.values())
                        print(  # noqa: T201 - live sampling diagnostics
                            f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                            f"{skip_count} steps (SKIPPED, weight=0)"
                        )
                        continue
                    step_count = sum(len(v) for v in ep_step_map.values())
                    if step_count == 0:
                        continue
                    ds = build_dataset_fn(
                        dataset_path,
                        dataset_spec,
                        data_filter,
                        step_indices_filter=ep_step_map,
                        hydrator_client=hydrator_client,
                    )
                    key = (family, priority)
                    if key not in buckets:
                        buckets[key] = []
                    buckets[key].append((ds, mix_ratio))

                    print(  # noqa: T201 - live sampling diagnostics
                        f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                        f"{step_count} steps"
                    )

        # Build DatasetGroup list with budgets from mapping dicts.
        groups: list[DatasetGroup] = []

        budget_mode = "proportional" if proportional_budget else "fixed"
        print(  # noqa: T201 - live sampling diagnostics
            f"[QualityPriorityGrouper] Budget mode: {budget_mode}"
        )

        for family, family_budget, priority_map in [
            ("demo", demo_budget_total, demo_map),
            ("rollout", other_budget_total, rollout_map),
        ]:
            if proportional_budget:
                weighted_step_total = 0.0
                for priority, weight in sorted(priority_map.items()):
                    if weight <= 0:
                        continue
                    datasets_in_bucket = buckets.get((family, priority), [])
                    if not datasets_in_bucket:
                        continue
                    steps = sum(sum(ds.shard_lengths) for ds, _ in datasets_in_bucket)
                    weighted_step_total += weight * steps
                if weighted_step_total <= 0:
                    continue
                for priority, weight in sorted(priority_map.items()):
                    key = (family, priority)
                    datasets_in_bucket = buckets.get(key, [])
                    if not datasets_in_bucket or weight <= 0:
                        continue
                    steps = sum(sum(ds.shard_lengths) for ds, _ in datasets_in_bucket)
                    budget = family_budget * (weight * steps / weighted_step_total)
                    groups.append(
                        DatasetGroup(
                            name=f"{family}_P{priority}",
                            datasets=datasets_in_bucket,
                            target_budget=budget,
                        )
                    )
            else:
                map_total = sum(priority_map.values())
                if map_total <= 0:
                    continue
                for priority, weight in sorted(priority_map.items()):
                    key = (family, priority)
                    datasets_in_bucket = buckets.get(key, [])
                    if not datasets_in_bucket or weight <= 0:
                        continue
                    budget = family_budget * (weight / map_total)
                    groups.append(
                        DatasetGroup(
                            name=f"{family}_P{priority}",
                            datasets=datasets_in_bucket,
                            target_budget=budget,
                        )
                    )

        if not groups:
            raise ValueError(
                "[QualityPriorityGrouper] No data found across all datasets. "
                "Check that dataset_paths exist and parquet files are accessible."
            )

        total_budget = sum(g.target_budget for g in groups)
        if total_budget > 0 and abs(total_budget - 1.0) > 1e-9:
            print(  # noqa: T201 - live sampling diagnostics
                f"[QualityPriorityGrouper] Renormalizing budgets: "
                f"{total_budget:.4f} -> 1.0 "
                f"(some families/priorities had no eligible data)"
            )
            for g in groups:
                g.target_budget /= total_budget

        return groups

    # -- Speedup path: merge-then-build with rejection or pre-sampling ------
    # One dataset per (family, path). Priority weighting via rejection sampling
    # or pre-sampling (when quality_weighted_presample=True).

    def _build_groups_speedup(
        self,
        dataset_specs: list,
        build_dataset_fn: Callable,
        data_filter: object,
        hydrator_client: object,
        data_config: DataConfig,
    ) -> list[DatasetGroup]:
        demo_budget_total = data_config.quality_weighted_clean_demo_budget
        other_budget_total = 1.0 - demo_budget_total
        allow_upweight = data_config.quality_weighted_allow_upweight_human
        demo_map = data_config.quality_weighted_demo_priority_map
        rollout_map = data_config.quality_weighted_rollout_priority_map
        lookahead = data_config.quality_weighted_lookahead
        proportional_budget = getattr(data_config, "quality_weighted_proportional_budget", False)
        presample = getattr(data_config, "quality_weighted_presample", False)

        family_datasets: dict[str, list[tuple]] = {"demo": [], "rollout": []}

        strategy_label = "presample" if presample else "rejection"
        for dataset_spec in tqdm(
            dataset_specs,
            total=len(dataset_specs),
            desc=f"[QualityPriorityGrouper] Initializing datasets (speedup/{strategy_label})",
        ):
            role = getattr(dataset_spec, "dataset_role", "default")
            mix_ratio = getattr(dataset_spec, "mix_ratio", 1.0)
            family = "demo" if role == "clean_demo" else "rollout"

            for dataset_path in (
                factory._resolve_one_path(str(p))  # noqa: SLF001 - shared factory helper
                for p in dataset_spec.dataset_paths
            ):
                _raise_unsupported_packds_path(
                    dataset_path,
                    factory_name="QualityPriorityGrouper",
                )
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "embodiment_tag is required"
                modality_configs = data_config.modality_configs[embodiment_tag]
                min_delta = min(min(c.delta_indices) for c in modality_configs.values())
                max_delta = max(max(c.delta_indices) for c in modality_configs.values())

                print(  # noqa: T201 - live sampling diagnostics
                    f"[QualityPriorityGrouper] Pre-scanning {dataset_path} "
                    f"(family={family}) for quality-priority classification..."
                )
                priority_step_filters = _classify_steps_by_priority(
                    dataset_path,
                    min_delta,
                    max_delta,
                    allow_upweight,
                    lookahead=lookahead,
                    allow_padding=data_config.allow_padding,
                    hydrator=hydrator_client,
                )

                family_map = demo_map if family == "demo" else rollout_map

                if presample:
                    # -- Pre-sampling: subsample steps at build time --------
                    # Compute accept_prob per priority, subsample, merge.
                    # No step_group_labels needed - every loaded sample is yielded.
                    active_weights = {
                        int(p): float(family_map[p])
                        for p in priority_step_filters
                        if family_map.get(p, 0) > 0
                    }
                    if not active_weights:
                        for priority, ep_step_map in priority_step_filters.items():
                            skip_count = sum(len(v) for v in ep_step_map.values())
                            print(  # noqa: T201 - live sampling diagnostics
                                f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                                f"{skip_count} steps (SKIPPED, weight=0)"
                            )
                        continue

                    # Compute accept probabilities
                    if proportional_budget:
                        # accept_prob(g) = map[g] / max(map)
                        max_w = max(active_weights.values())
                        accept_probs = {g: w / max_w for g, w in active_weights.items()}
                    else:
                        # accept_prob(g) proportional to map[g] / steps[g] to cancel natural distribution
                        steps_per_priority = {}
                        for p, ep_map in priority_step_filters.items():
                            if p in active_weights:
                                steps_per_priority[p] = sum(len(v) for v in ep_map.values())
                        rates = {
                            g: active_weights[g] / steps_per_priority[g]
                            for g in active_weights
                            if steps_per_priority.get(g, 0) > 0
                        }
                        max_rate = max(rates.values()) if rates else 1.0
                        accept_probs = {g: r / max_rate for g, r in rates.items()}

                    merged_filter: dict[int, np.ndarray] = {}
                    merged_labels: dict[int, np.ndarray] = {}
                    original_histogram: dict[int, int] = {}
                    presample_histogram: dict[int, int] = {}

                    for priority, ep_step_map in priority_step_filters.items():
                        if priority not in accept_probs:
                            skip_count = sum(len(v) for v in ep_step_map.values())
                            print(  # noqa: T201 - live sampling diagnostics
                                f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                                f"{skip_count} steps (SKIPPED, weight=0)"
                            )
                            continue

                        ap = accept_probs[priority]
                        original_count = sum(len(v) for v in ep_step_map.values())
                        original_histogram[priority] = original_count
                        kept = 0

                        for ep_idx, steps in ep_step_map.items():
                            if ap >= 1.0:
                                selected = steps
                            else:
                                rng = np.random.default_rng(
                                    seed=data_config.seed + ep_idx * 100 + priority
                                )
                                n_keep = max(1, round(len(steps) * ap))
                                if n_keep >= len(steps):
                                    selected = steps
                                else:
                                    selected = rng.choice(steps, size=n_keep, replace=False)
                                    selected = np.sort(selected)

                            if len(selected) > 0:
                                priority_labels = np.full(len(selected), priority, dtype=np.int8)
                                if ep_idx in merged_filter:
                                    merged_filter[ep_idx] = np.concatenate(
                                        [merged_filter[ep_idx], selected]
                                    )
                                    merged_labels[ep_idx] = np.concatenate(
                                        [merged_labels[ep_idx], priority_labels]
                                    )
                                else:
                                    merged_filter[ep_idx] = selected
                                    merged_labels[ep_idx] = priority_labels
                                kept += len(selected)

                        presample_histogram[priority] = kept
                        print(  # noqa: T201 - live sampling diagnostics
                            f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                            f"{original_count} steps, kept {kept} "
                            f"(accept_prob={ap:.3f})"
                        )

                    if not merged_filter:
                        continue

                    # Sort steps within each episode for sequential I/O; keep
                    # priority labels aligned with their step indices.
                    for ep_idx, steps in merged_filter.items():
                        order = np.argsort(steps)
                        merged_filter[ep_idx] = steps[order]
                        merged_labels[ep_idx] = merged_labels[ep_idx][order]

                    total_merged = sum(len(v) for v in merged_filter.values())
                    print(  # noqa: T201 - live sampling diagnostics
                        f"[PRESAMPLE_BUILD] family={family} path={dataset_path} "
                        f"total_steps={total_merged} "
                        f"original={dict(sorted(original_histogram.items()))} "
                        f"kept={dict(sorted(presample_histogram.items()))}"
                    )

                    log = structlog.get_logger()
                    for ep_idx in sorted(merged_filter.keys()):
                        log.debug(
                            "step_priority_mapping",
                            dataset_path=dataset_path,
                            dataset_role=role,
                            family=family,
                            episode_id=int(ep_idx),
                            kept_step_ids=merged_filter[ep_idx].tolist(),
                            kept_priorities=merged_labels[ep_idx].tolist(),
                            n_kept=len(merged_filter[ep_idx]),
                        )

                    ds = build_dataset_fn(
                        dataset_path,
                        dataset_spec,
                        data_filter,
                        step_indices_filter=merged_filter,
                        hydrator_client=hydrator_client,
                    )
                    family_datasets[family].append((ds, mix_ratio, presample_histogram))

                else:
                    # -- Rejection sampling: merge all steps + parallel labels -
                    merged_filter: dict[int, np.ndarray] = {}
                    merged_labels: dict[int, np.ndarray] = {}
                    group_histogram: dict[int, int] = {}

                    for priority, ep_step_map in priority_step_filters.items():
                        if family_map.get(priority, 0) <= 0:
                            skip_count = sum(len(v) for v in ep_step_map.values())
                            print(  # noqa: T201 - live sampling diagnostics
                                f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                                f"{skip_count} steps (SKIPPED, weight=0)"
                            )
                            continue
                        step_count = sum(len(v) for v in ep_step_map.values())
                        if step_count == 0:
                            continue
                        group_histogram[priority] = step_count

                        for ep_idx, steps in ep_step_map.items():
                            if ep_idx not in merged_filter:
                                merged_filter[ep_idx] = steps
                                merged_labels[ep_idx] = np.full(len(steps), priority, dtype=np.int8)
                            else:
                                merged_filter[ep_idx] = np.concatenate(
                                    [merged_filter[ep_idx], steps]
                                )
                                merged_labels[ep_idx] = np.concatenate(
                                    [
                                        merged_labels[ep_idx],
                                        np.full(len(steps), priority, dtype=np.int8),
                                    ]
                                )

                        print(  # noqa: T201 - live sampling diagnostics
                            f"[QualityPriorityGrouper]   -> {family}/P{priority}: "
                            f"{step_count} steps"
                        )

                    if not merged_filter:
                        continue

                    # Sort steps within each episode for sequential I/O
                    for ep_idx in merged_filter:
                        order = np.argsort(merged_filter[ep_idx])
                        merged_filter[ep_idx] = merged_filter[ep_idx][order]
                        merged_labels[ep_idx] = merged_labels[ep_idx][order]

                    total_merged = sum(len(v) for v in merged_filter.values())
                    print(  # noqa: T201 - live sampling diagnostics
                        f"[MERGED_BUILD] family={family} path={dataset_path} "
                        f"total_steps={total_merged} "
                        f"group_dist={dict(sorted(group_histogram.items()))}"
                    )

                    log = structlog.get_logger()
                    for ep_idx in sorted(merged_filter.keys()):
                        log.debug(
                            "step_priority_mapping",
                            dataset_path=dataset_path,
                            dataset_role=role,
                            family=family,
                            episode_id=int(ep_idx),
                            kept_step_ids=merged_filter[ep_idx].tolist(),
                            kept_priorities=merged_labels[ep_idx].tolist(),
                            n_kept=len(merged_filter[ep_idx]),
                        )

                    ds = build_dataset_fn(
                        dataset_path,
                        dataset_spec,
                        data_filter,
                        step_indices_filter=merged_filter,
                        step_group_labels=merged_labels,
                        hydrator_client=hydrator_client,
                    )
                    family_datasets[family].append((ds, mix_ratio, group_histogram))

        # Build groups
        groups: list[DatasetGroup] = []
        budget_mode = "proportional" if proportional_budget else "fixed"

        if presample:
            print(  # noqa: T201 - live sampling diagnostics
                f"[QualityPriorityGrouper] Speedup mode: merge-then-build "
                f"(pre-sampling active, zero I/O waste, budget_mode={budget_mode})"
            )
            for family, family_budget in [
                ("demo", demo_budget_total),
                ("rollout", other_budget_total),
            ]:
                ds_entries = family_datasets[family]
                if not ds_entries:
                    continue
                datasets_with_ratios = [(ds, ratio) for ds, ratio, _ in ds_entries]
                groups.append(
                    DatasetGroup(
                        name=f"{family}_presampled",
                        datasets=datasets_with_ratios,
                        target_budget=family_budget,
                    )
                )
        else:
            print(  # noqa: T201 - live sampling diagnostics
                f"[QualityPriorityGrouper] Speedup mode: merge-then-build "
                f"(rejection sampling active, budget_mode={budget_mode})"
            )
            for family, family_budget, priority_map in [
                ("demo", demo_budget_total, demo_map),
                ("rollout", other_budget_total, rollout_map),
            ]:
                ds_entries = family_datasets[family]
                if not ds_entries:
                    continue
                datasets_with_ratios = [(ds, ratio) for ds, ratio, _ in ds_entries]
                histograms = [hist for _, _, hist in ds_entries]

                # Compute the rejection weight map.
                active_map = {int(k): float(v) for k, v in priority_map.items() if v > 0}

                if proportional_budget:
                    group_weight_map = active_map
                else:
                    aggregated_steps: dict[int, int] = {}
                    for hist in histograms:
                        for g, count in hist.items():
                            aggregated_steps[g] = aggregated_steps.get(g, 0) + count
                    group_weight_map = {}
                    for g, w in active_map.items():
                        steps_g = aggregated_steps.get(g, 0)
                        if steps_g > 0:
                            group_weight_map[g] = w / steps_g

                    print(  # noqa: T201 - live sampling diagnostics
                        f"[QualityPriorityGrouper] {family}: proportional_budget=False correction "
                        f"raw_map={active_map} "
                        f"step_counts={dict(sorted(aggregated_steps.items()))} "
                        f"corrected_map={group_weight_map}"
                    )

                groups.append(
                    DatasetGroup(
                        name=f"{family}_merged",
                        datasets=datasets_with_ratios,
                        target_budget=family_budget,
                        group_weight_map=group_weight_map,
                        group_histograms=histograms,
                    )
                )

        if not groups:
            raise ValueError(
                "[QualityPriorityGrouper] No data found across all datasets. "
                "Check that dataset_paths exist and parquet files are accessible."
            )

        total_budget = sum(g.target_budget for g in groups)
        if total_budget > 0 and abs(total_budget - 1.0) > 1e-9:
            print(  # noqa: T201 - live sampling diagnostics
                f"[QualityPriorityGrouper] Renormalizing budgets: "
                f"{total_budget:.4f} -> 1.0 "
                f"(some families/priorities had no eligible data)"
            )
            for g in groups:
                g.target_budget /= total_budget

        return groups
