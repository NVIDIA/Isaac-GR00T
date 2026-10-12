# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Fixed data, budget, and base-model fields inside an editable training config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

# data.datasets is locked whole, so per-dataset mix ratios are part of the fixed data split.
# resume_from_checkpoint loads weights and trainer state after start_from_checkpoint.
# The model.* paths are loaded separately from the checkpoint; tokenizers and processors
# are never overwritten by checkpoint weights.
DEFAULT_LOCKED_FIELDS = (
    "data.datasets",
    "training.start_from_checkpoint",
    "training.resume_from_checkpoint",
    "training.max_steps",
    "training.num_gpus",
    "model.tokenizer_path",
    "model.text_encoder_cfg.text_encoder_pretrained_path",
    "model.diffusion_model_cfg.diffusion_model_pretrained_path",
    "model.vae_cfg.vae_pretrained_path",
)


class _Missing:
    """Marks a dotted field that is absent from a config."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class TrainConfigLock:
    """Compare a candidate training config against the pinned baseline config.

    The training config stays editable so agents can tune the recipe, but the
    training data, step budget, accelerator count, effective global batch, and
    base checkpoint must render to the baseline values. Training datasets must
    also stay disjoint from the evaluation datasets.
    """

    path: str
    fields: tuple[str, ...] = DEFAULT_LOCKED_FIELDS
    eval_dataset_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the lock before any candidate is compared."""
        if not isinstance(self.path, str) or not self.path.strip("/"):
            raise TypeError("locked_train_config.path must be a non-empty string")
        if Path(self.path).is_absolute():
            raise ValueError("locked_train_config.path must be relative to the workspace")
        for label, values in (
            ("fields", self.fields),
            ("eval_dataset_paths", self.eval_dataset_paths),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise TypeError(f"locked_train_config.{label} entries must be non-empty strings")

    @classmethod
    def from_mapping(cls, raw: Any, source: Path) -> TrainConfigLock:
        """Load the already environment-expanded `locked_train_config` YAML section."""
        if not isinstance(raw, Mapping):
            raise TypeError(f"{source}: locked_train_config must be a mapping")
        fields = raw.get("fields", list(DEFAULT_LOCKED_FIELDS))
        eval_dataset_paths = raw.get("eval_dataset_paths", [])
        for label, value in (("fields", fields), ("eval_dataset_paths", eval_dataset_paths)):
            if not isinstance(value, list):
                raise TypeError(f"{source}: locked_train_config.{label} must be a list")
        return cls(
            path=raw.get("path"),
            fields=tuple(fields),
            eval_dataset_paths=tuple(eval_dataset_paths),
        )

    def violations(
        self,
        *,
        baseline: Any,
        candidate: Any,
        extra_eval_dataset_paths: tuple[str, ...] = (),
    ) -> list[str]:
        """Compare rendered configs; `candidate=MISSING` means the file was deleted."""
        if candidate is MISSING:
            return [f"{self.path}: locked training config was deleted"]
        violations = [
            f"{self.path}: locked field {name} changed from "
            f"{_lookup(baseline, name)!r} to {_lookup(candidate, name)!r}"
            for name in self.fields
            if _lookup(baseline, name) != _lookup(candidate, name)
        ]
        baseline_batch = _effective_global_batch(baseline)
        candidate_batch = _effective_global_batch(candidate)
        if baseline_batch != candidate_batch:
            violations.append(
                f"{self.path}: effective global batch changed from "
                f"{baseline_batch!r} to {candidate_batch!r}"
            )
        eval_paths = (*self.eval_dataset_paths, *extra_eval_dataset_paths)
        violations.extend(
            f"{self.path}: training dataset {train} overlaps evaluation dataset {held_out}"
            for train in _train_dataset_paths(candidate)
            for held_out in eval_paths
            if _paths_overlap(train, held_out)
        )
        return violations


def _lookup(config: Any, dotted: str) -> Any:
    """Return one dotted mapping field, or MISSING when any segment is absent."""
    value = config
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return MISSING
        value = value[key]
    return value


def _effective_global_batch(config: Any) -> Any:
    """Return the per-step sample count, or the raw factors when one is not an integer."""
    gradient_accumulation = _lookup(config, "training.gradient_accumulation_steps")
    factors = (
        _lookup(config, "training.per_device_batch_size"),
        _lookup(config, "training.num_gpus"),
        1 if gradient_accumulation is MISSING else gradient_accumulation,
    )
    if all(isinstance(factor, int) and not isinstance(factor, bool) for factor in factors):
        return factors[0] * factors[1] * factors[2]
    return factors


def _train_dataset_paths(config: Any) -> list[str]:
    """Collect every `data.datasets[].dataset_paths` entry."""
    datasets = _lookup(config, "data.datasets")
    if not isinstance(datasets, list):
        return []
    paths: list[str] = []
    for dataset in datasets:
        if isinstance(dataset, Mapping) and isinstance(dataset.get("dataset_paths"), list):
            paths.extend(path for path in dataset["dataset_paths"] if isinstance(path, str))
    return paths


def _paths_overlap(first: str, second: str) -> bool:
    """Return whether two dataset locations are equal or nested inside each other."""
    left, right = _normalize_location(first), _normalize_location(second)
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def _normalize_location(location: str) -> str:
    """Canonicalize local paths through symlinks; compare URIs textually."""
    if "://" in location:
        return location.rstrip("/")
    return os.path.realpath(os.path.expanduser(location))
