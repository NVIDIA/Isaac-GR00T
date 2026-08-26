# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Registration helpers for polymorphic training configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from groot.core.training.configs.training.training_config import TrainingConfig

TRAINING_CONFIG_TYPES: dict[str, type[TrainingConfig]] = {}


def register_training_config(shortname: str, config_type: type[TrainingConfig]) -> None:
    """Register a concrete training configuration for YAML reconstruction."""
    existing = TRAINING_CONFIG_TYPES.get(shortname)
    if existing is not None and existing is not config_type:
        raise ValueError(f"Training config {shortname!r} is already registered")
    TRAINING_CONFIG_TYPES[shortname] = config_type


def create_training_config(payload: Any) -> TrainingConfig:
    """Construct the training-config subtype selected by an optional ``_config`` key."""
    if not isinstance(payload, Mapping):
        raise TypeError("Training configuration must be a mapping")
    raw = dict(payload)
    shortname = str(raw.pop("_config", "Training"))
    try:
        config_type = TRAINING_CONFIG_TYPES[shortname]
    except KeyError as error:
        available = ", ".join(sorted(TRAINING_CONFIG_TYPES))
        raise ValueError(
            f"Unknown training config {shortname!r}; registered configs: {available}"
        ) from error
    return config_type(**cast(dict[str, Any], raw))


def training_config_shortname(config: TrainingConfig) -> str:
    """Return the registered discriminator for a concrete training config."""
    for shortname, config_type in TRAINING_CONFIG_TYPES.items():
        if type(config) is config_type:
            return shortname
    raise ValueError(f"Unregistered training config type {type(config).__name__!r}")


register_training_config("Training", TrainingConfig)
