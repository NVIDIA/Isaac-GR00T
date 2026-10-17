# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Simplified Configuration for ICRT v2 Training.

Flat configuration structure with sensible defaults.
Focus on DeepSpeed for distributed training.
"""

import copy
from dataclasses import dataclass, field
import time
from typing import Annotated, Any, cast

import tyro

from groot.core.data.dataset.recipe_fields import expand_filter_sets
from groot.core.data.hydrator.config import HydratorConfig
from groot.core.data.types import ModalityConfig
from groot.core.training.configs.data.embodiment_configs import MODALITY_CONFIGS
from groot.core.training.configs.data.filter_config import FilterConfig, FilterType

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def _to_plain_container(value: Any) -> Any:
    """Convert OmegaConf-style containers when available, otherwise use dict()."""
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return dict(value)
    return dict(value)


@dataclass
class DeltaCatDatasetConfig:
    """DeltaCAT PackDS loader options attached to one dataset entry."""

    episode_query: str | None = None
    steps_query: str | None = None
    # UTC Unix timestamp in nanoseconds, passed to DeltaCAT as `as_of`. When
    # left unset for a dc:// dataset, training materializes the current
    # timestamp before the frozen config is saved so the run records its exact
    # table snapshot point.
    as_of_timestamp: int | None = None
    # Canonical modality key -> physical DeltaCAT table key. This lets a dataset
    # use a canonical embodiment config while reading tables whose source
    # columns were written with older or table-specific names.
    key_remap: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.as_of_timestamp is not None:
            self.as_of_timestamp = int(self.as_of_timestamp)
        self.key_remap = {str(key): str(value) for key, value in (self.key_remap or {}).items()}


@dataclass
class DeltaCatStatsEmbodimentSamplingConfig:
    """Per-embodiment overrides for DeltaCAT sampled statistics."""

    sample_anchor_steps_per_embodiment: int | None = None
    min_anchor_steps_per_stratum: int | None = None


@dataclass
class DeltaCatStatsSamplingConfig:
    """Training-start sampled statistics controls for DeltaCAT PackDS datasets."""

    sample_anchor_steps_per_embodiment: int = 50_000
    min_anchor_steps_per_stratum: int = 1_024
    seed: int | None = None
    embodiments: dict[str, DeltaCatStatsEmbodimentSamplingConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Convert YAML/OmegaConf dict overrides into nested dataclasses."""
        converted: dict[str, DeltaCatStatsEmbodimentSamplingConfig] = {}
        for embodiment, override in self.embodiments.items():
            if isinstance(override, DeltaCatStatsEmbodimentSamplingConfig):
                converted[str(embodiment)] = override
                continue
            raw = _to_plain_container(override)
            if isinstance(raw, dict):
                converted[str(embodiment)] = DeltaCatStatsEmbodimentSamplingConfig(
                    **cast(dict[str, Any], raw)
                )
            else:
                converted[str(embodiment)] = DeltaCatStatsEmbodimentSamplingConfig()
        self.embodiments = converted


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios.  For convenience the *legacy*
    single-dataset fields still exist; if ``datasets`` is non-empty they take
    precedence.
    """

    # Path to the dataset root directory (can be strings or dicts for complex configs)
    dataset_paths: list[Any]

    # Robot embodiment identifier (e.g. "gr1", "franka")
    embodiment_tag: str | None = None

    # Relative sampling probability (will be normalised across the list)
    mix_ratio: float = 1.0

    dataset_type: str = "physical_embodiment"

    # Role for weighted BC factory v2: "default" | "clean_demo" | "rollout"
    dataset_role: str = "default"

    # Optional validation dataset path for open-loop evaluation
    # If not provided, falls back to dataset_paths for evaluation
    val_dataset_path: str | None = None

    # DeltaCAT PackDS-specific SQL filters. This is
    # ignored for non-dc:// datasets and keeps training YAML self-contained
    # without encoding loader settings into the dataset URI.
    deltacat: DeltaCatDatasetConfig | None = None

    # Which LanceDS view an entry reads, and at which version (views are materialized offline by
    # `gear lance ingest`). Each field is one value for the entry or a {dataset_path: value}
    # mapping. An s3:// path is read as a LanceDS store only when its entry gives it one of these
    # fields (path_detection.is_lance_source); a local store is found by its layout.
    # lance_view_id takes legacy (spec.json) view ids only: a tagged view is named by its filters,
    # in lance_step_filter, and the loader refuses its id, with or without them.
    # The lance_* entry fields are typed Any: experiment.py dumps conf.yaml as an OmegaConf
    # structured config for every run, and OmegaConf refuses unions of containers (as tyro does).
    lance_view_id: Annotated[Any, tyro.conf.Suppress] = None  # str | {dataset_path: str}
    # The pin: which versions of step.lance, media.lance and episodes.lance a LanceDS training path
    # reads. Either lance_as_of, a quoted ISO-8601 UTC timestamp (yaml reads an unquoted one as a
    # datetime): each table at its newest version committed at or before it; or
    # lance_step_version, lance_media_version and lance_episodes_version, all three: exactly those
    # versions. Training refuses a layout that does not describe the step table it reads (pin a
    # time after both committed), a path left unpinned and a pin that does not pin
    # (lance_pins.check_lance_training_pins); a launch stamps every path given neither with its
    # launch time (lance_pins.stamp_lance_pins). A store val_dataset_path reads its full view,
    # unpinned (the step table's latest version at startup): the entry's lance_view_id and
    # lance_step_filter do not reach it.
    lance_as_of: Annotated[Any, tyro.conf.Suppress] = None  # str | {dataset_path: str}
    lance_step_version: Annotated[Any, tyro.conf.Suppress] = None  # int | {dataset_path: int}
    lance_media_version: Annotated[Any, tyro.conf.Suppress] = None  # int | {dataset_path: int}
    lance_episodes_version: Annotated[Any, tyro.conf.Suppress] = None  # int | {dataset_path: int}
    # The view by definition: a filter list `[{fn, args}]`, or the name of a data.lance_step_filters
    # entry; its view id is derived at load, so an edited filter is a missing view, never another.
    # str | [{fn, args}] | {dataset_path: either}
    lance_step_filter: Annotated[Any, tyro.conf.Suppress] = None

    def __post_init__(self):
        """Convert dict/OmegaConf DictConfig values to nested configs when needed."""
        # Plain containers: the filter list is hashed (json) into the view id downstream.
        if OmegaConf is not None and OmegaConf.is_config(self.lance_step_filter):
            self.lance_step_filter = OmegaConf.to_container(self.lance_step_filter, resolve=True)
        if self.deltacat is None or isinstance(self.deltacat, DeltaCatDatasetConfig):
            return
        raw = _to_plain_container(self.deltacat)
        if isinstance(raw, dict):
            self.deltacat = DeltaCatDatasetConfig(**cast(dict[str, Any], raw))
        else:
            self.deltacat = DeltaCatDatasetConfig()


@dataclass
class DataConfig:
    """Dataset configuration (supports single or multiple datasets)."""

    # Leave empty by default for backwards-compatibility with the original
    # single-dataset workflow.  Users can supply one or more configs via CLI or
    # YAML when they need mixing.
    datasets: list[SingleDatasetConfig] = field(default_factory=list)

    # Modality configs
    # There are three sources of modality configs:
    # 1. Default modality configs in code: groot/vla/omni/configs/data/embodiment_configs.py
    # 2. Modality configs supplied through command line: --data.modality_configs (although rare and inconvenient)
    # 1 and 2 are unified through `config.data.modality_configs`.
    # 3. modality configs saved in the pretrained checkpoint.
    # Both 1, 2 and 3 will be "merged" in N1.5 finetuning. See logic in `GrootN1d5Processor.from_pretrained`.
    # NOTE: N1.6 does not yet have equivalent modality-config merge logic.
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        # Config.load merges YAML overrides into this mapping, so each DataConfig
        # needs its own copy instead of mutating the shared default template.
        default_factory=lambda: copy.deepcopy(MODALITY_CONFIGS)
    )

    # Sharded dataset configuration
    download_cache: bool = False
    # Full hydrator config (cache dir, workers, etc.). Parsed from YAML under data.hydrator_config.
    # Converted from dict to HydratorConfig in Config.load_dict before DataConfig is built (OmegaConf-friendly).
    # If None, Hydrator uses default HydratorConfig().
    hydrator_config: HydratorConfig | None = None
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)
    # Override statistics from the pretrained checkpoint
    override_pretraining_statistics: bool = False
    # DeltaCAT PackDS stats are computed at training start by sampling valid
    # low-dimensional anchor steps from dc:// datasets. Non-DeltaCAT datasets
    # still use their existing meta/stats.json and meta/relative_stats.json paths.
    deltacat_stats: DeltaCatStatsSamplingConfig = field(default_factory=DeltaCatStatsSamplingConfig)

    # General task / mode config (shared across datasets)
    # Options: "icrt", "trajectory", "single_turn", "single_turn_rl".
    # NOTE: This should become a Literal when downstream config parsing supports it.
    mode: str = "single_turn"
    random_chop: float = 0.0
    # Cache and repeatedly return the first datapoint to simulate best-case data loading.
    mock_dataset_mode: bool = False

    # ICRT specific
    num_prompt_trajectories: int = 2
    variable_num_demos: bool = False  # Support variable number of demonstrations
    max_prompt_trajectories: int = 5  # Maximum demos when using variable

    # Data loading
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str = "fork"  # Options: "fork", "spawn", and "forkserver"
    enable_struct_logs: bool = False
    data_loading_logging_level: str = "INFO"
    data_loading_log_dir: str | None = None
    dataloader_prefetch_factor: int | None = None
    allow_padding: bool = False

    # Subsample ratio for the dataset
    subsample_ratio: float = 1.0

    # DP Image Config
    image_crop_size: list[int] = field(default_factory=lambda: [244, 244])
    image_target_size: list[int] = field(default_factory=lambda: [224, 224])
    video_backend: str = "torchcodec"
    # When True, on invalid video frame indices clamp to [0, total_frames-1] and retry; when False, fail training
    resample_on_invalid_frames: bool = False

    # Filter configuration (global, applies to all datasets)
    filter: FilterConfig = field(default_factory=FilterConfig)

    # Maximum number of steps per trajectory subsequence (trajectory mode only).
    # Long episodes are split into windows of this length; short
    # episodes are returned whole.  None = no limit (full episodes).
    max_len: int | None = None

    # Step interval between sampled trajectory timesteps (trajectory mode only).
    # None preserves the existing behavior of spacing trajectory timesteps by
    # the action horizon.
    trajectory_step_stride: int | None = None

    # Chunk configuration (Currently used by DreamZero)
    num_chunks: int = 1

    # Relative stats: reservoir sampling is streaming/lower-memory; False computes exact
    # q01/q99 values with higher memory use.
    use_reservoir_sampling: bool = True

    # Weighted BC factory v2 settings (used when dataset_factory_type == "weighted_sampling")
    dataset_factory_type: str = "default"  # "default" | "weighted_sampling" | "quality_weighted"
    # Legacy weighted_sampling factory flag; the default and quality_weighted factories auto-detect S3.
    hydrator_enabled: bool = False
    weighted_sample_batch_demo_ratio_cap: float = 0.4
    weighted_sample_batch_rollout_human_target_prob: float = 0.5
    weighted_sample_rollout_human_chunk_lookahead: int = 10
    weighted_sample_soft_threshold: bool = False

    # Quality-priority weighted sampling (dataset_factory_type == "quality_weighted")
    # Level 1: clean_demo vs everything else budget split.
    quality_weighted_clean_demo_budget: float = 0.4
    # Whether to add +1 priority for human-source steps, except failure quality.
    quality_weighted_allow_upweight_human: bool = False
    # Mapping from priority level -> sampling weight, normalized within each family.
    quality_weighted_demo_priority_map: dict[int, float] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 4}
    )
    quality_weighted_rollout_priority_map: dict[int, float] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 4}
    )
    # Lookahead window for quality majority vote; 0/1 uses the anchor step only.
    quality_weighted_lookahead: int = 10
    # When True, budget allocation considers natural step counts per priority level.
    quality_weighted_proportional_budget: bool = False
    # Merge priority levels into one dataset per path and use rejection or presampling.
    quality_weighted_speedup: bool = False
    # Requires quality_weighted_speedup=True. Subsample at build time instead of rejection.
    quality_weighted_presample: bool = False

    # When False, missing or corrupt stats files cause a hard failure instead of
    # being silently regenerated.  Set to False for production training to catch
    # data preparation issues early.
    auto_generate_stats: bool = True

    # Honor ``meta/step_filter.jsonl`` exclusion metadata in single-step datasets.
    # Kept append-only and opt-in so existing positional construction and datasets
    # carrying legacy metadata retain their prior behavior.
    load_step_filter: bool = False

    # The datasets root: its _registry/ is keyed by store path relative to it; unset = each store's
    # meta/.
    lance_datasets_root: str | None = None
    # {name: [{fn, args}]} filter sets that entries' lance_step_filter may name; expanded at load.
    lance_step_filters: Annotated[dict[str, list[dict[str, Any]]] | None, tyro.conf.Suppress] = None
    # The registry as of this ISO-8601 UTC timestamp: every loader reads its row (modality, stats)
    # at the registry version committed at or before it; needs lance_datasets_root. When unset, a
    # launch stamps its launch time and a resume reuses its checkpoint's
    # (lance_pins.stamp_lance_pins).
    lance_registry_as_of: str | None = None

    def _resolve_filter_sets(self) -> None:
        """Replace ``lance_step_filter`` names with the lists ``lance_step_filters`` defines."""
        if self.lance_step_filters is not None:
            self.lance_step_filters = _to_plain_container(self.lance_step_filters)
        sets = self.lance_step_filters or {}
        for entry in self.datasets:
            entry.lance_step_filter = expand_filter_sets(
                entry.lance_step_filter, sets, entry.dataset_paths
            )

    def __post_init__(self):
        """Convert dict/OmegaConf DictConfig values to nested configs when needed."""
        converted_datasets: list[SingleDatasetConfig] = []
        for dataset in self.datasets:
            if isinstance(dataset, SingleDatasetConfig):
                converted_datasets.append(dataset)
                continue
            raw_dataset = _to_plain_container(dataset)
            if isinstance(raw_dataset, dict):
                converted_datasets.append(SingleDatasetConfig(**cast(dict[str, Any], raw_dataset)))
            else:
                converted_datasets.append(dataset)
        self.datasets = converted_datasets
        self._resolve_filter_sets()

        if self.hydrator_config is not None and not isinstance(
            self.hydrator_config, HydratorConfig
        ):
            raw = _to_plain_container(self.hydrator_config)
            if isinstance(raw, dict):
                self.hydrator_config = HydratorConfig(**cast(dict[str, Any], raw))
            else:
                self.hydrator_config = HydratorConfig()

        if not isinstance(self.filter, FilterConfig):
            raw_filter = _to_plain_container(self.filter)
            if isinstance(raw_filter, dict):
                filter_type = raw_filter.get("filter_type")
                if filter_type is not None and not isinstance(filter_type, FilterType):
                    try:
                        raw_filter["filter_type"] = FilterType(filter_type)
                    except ValueError:
                        raw_filter["filter_type"] = FilterType[str(filter_type)]
                self.filter = FilterConfig(**cast(dict[str, Any], raw_filter))
            else:
                self.filter = FilterConfig()

        if not isinstance(self.deltacat_stats, DeltaCatStatsSamplingConfig):
            raw_stats = _to_plain_container(self.deltacat_stats)
            if isinstance(raw_stats, dict):
                self.deltacat_stats = DeltaCatStatsSamplingConfig(**cast(dict[str, Any], raw_stats))
            else:
                self.deltacat_stats = DeltaCatStatsSamplingConfig()
        if self.dataloader_prefetch_factor is not None:
            self.dataloader_prefetch_factor = int(self.dataloader_prefetch_factor)

        # YAML / OmegaConf serializes dict keys as strings; coerce to int->float.
        for attr in ("quality_weighted_demo_priority_map", "quality_weighted_rollout_priority_map"):
            raw = getattr(self, attr, None)
            if raw is not None and isinstance(raw, dict):
                raw = _to_plain_container(raw)
                setattr(self, attr, {int(k): float(v) for k, v in raw.items()})

    def materialize_deltacat_as_of_timestamps(self, now: int | None = None) -> None:
        """Fill missing DeltaCAT as-of timestamps for reproducible saved configs."""
        timestamp = int(time.time_ns() if now is None else now)
        for dataset in self.datasets:
            if not _dataset_has_deltacat_source(dataset):
                continue
            if dataset.deltacat is None:
                dataset.deltacat = DeltaCatDatasetConfig()
            if dataset.deltacat.as_of_timestamp is None:
                dataset.deltacat.as_of_timestamp = timestamp

    def has_deltacat_sources(self) -> bool:
        """Return whether any configured dataset source is DeltaCAT-backed."""
        return any(_dataset_has_deltacat_source(dataset) for dataset in self.datasets)

    def get_max_trajectory_sequence_length(self) -> int:
        """Return the fixed trajectory length implied by the active data configuration."""
        if self.mode != "trajectory":
            raise ValueError("Fixed trajectory sequence length requires data.mode='trajectory'")
        if self.max_len is None or self.max_len <= 0:
            raise ValueError("Fixed trajectory sequence length requires data.max_len > 0")

        active_embodiments = {
            dataset.embodiment_tag
            for dataset in self.datasets
            if dataset.embodiment_tag is not None
        }
        if not active_embodiments:
            raise ValueError("Fixed trajectory sequence length requires an embodiment_tag")

        max_sequence_length = 0
        for embodiment_tag in active_embodiments:
            modalities = self.modality_configs.get(embodiment_tag)
            if modalities is None or "action" not in modalities:
                raise ValueError(f"Missing action modality config for {embodiment_tag!r}")
            action_delta_indices = modalities["action"].delta_indices
            if not action_delta_indices:
                raise ValueError(f"Action delta indices must not be empty for {embodiment_tag!r}")

            step_stride = (
                len(action_delta_indices)
                if self.trajectory_step_stride is None
                else self.trajectory_step_stride
            )
            if step_stride <= 0:
                raise ValueError("data.trajectory_step_stride must be positive")

            action_horizon = len(action_delta_indices)
            effective_length = self.max_len - action_horizon + 1
            if effective_length <= 0:
                raise ValueError(
                    f"data.max_len={self.max_len} is shorter than the action horizon "
                    f"{action_horizon} for {embodiment_tag!r}"
                )
            sequence_length = (effective_length + step_stride - 1) // step_stride
            max_sequence_length = max(max_sequence_length, sequence_length)

        return max_sequence_length


def _dataset_has_deltacat_source(dataset: SingleDatasetConfig) -> bool:
    for source in dataset.dataset_paths:
        if isinstance(source, str) and source.startswith("dc://"):
            return True
        if isinstance(source, dict):
            path = source.get("path") or source.get("uri")
            if path is not None and str(path).startswith("dc://"):
                return True
            nested = source.get("deltacat")
            if isinstance(nested, dict):
                return True
            source_type = source.get("type") or source.get("source")
            if source_type == "deltacat":
                return True
    return False
