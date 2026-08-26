# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Registration and compatibility helpers for polymorphic data configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from groot.core.training.configs.data.data_config import (
    DataConfig,
    SingleDatasetConfig,
    _to_plain_container,
)

DATA_CONFIG_TYPES: dict[str, type[DataConfig]] = {}


def register_data_config(shortname: str, config_type: type[DataConfig]) -> None:
    """Register a concrete data configuration for YAML reconstruction."""
    existing = DATA_CONFIG_TYPES.get(shortname)
    if existing is not None and existing is not config_type:
        raise ValueError(f"Data config {shortname!r} is already registered")
    DATA_CONFIG_TYPES[shortname] = config_type


def create_data_config(payload: Any) -> DataConfig:
    """Construct the data-config subtype selected by an optional ``_config`` key."""
    raw = _to_plain_container(payload)
    if not isinstance(raw, dict):
        raise TypeError("Data configuration must be a mapping")
    shortname = str(raw.pop("_config", "Data"))
    try:
        config_type = DATA_CONFIG_TYPES[shortname]
    except KeyError as error:
        available = ", ".join(sorted(DATA_CONFIG_TYPES))
        raise ValueError(
            f"Unknown data config {shortname!r}; registered configs: {available}"
        ) from error
    return config_type(**cast(dict[str, Any], raw))


@dataclass(frozen=True)
class DataValidationInputs:
    """Inputs a data subtype exposes to the unchanged base validation rules."""

    datasets: list[SingleDatasetConfig]
    embodiment_tags: set[str]
    permits_empty_dataset_mix: bool = False


def validate_data_training_contract(data: DataConfig, model: Any, training: Any) -> None:
    """Validate subtype-specific data/model/training compatibility."""
    validate = getattr(data, "validate_training_contract", None)
    if callable(validate):
        validate(model, training)
        return
    validate = getattr(model, "validate_training_contract", None)
    if callable(validate):
        validate(data, training)


def resolve_data_validation_inputs(data: DataConfig) -> DataValidationInputs:
    """Resolve the inputs consumed by shared validation without validating them."""
    build_inputs = getattr(data, "build_validation_inputs", None)
    if callable(build_inputs):
        return cast(DataValidationInputs, build_inputs())
    return DataValidationInputs(datasets=list(data.datasets), embodiment_tags=set())


register_data_config("Data", DataConfig)
