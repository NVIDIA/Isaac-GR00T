# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Compute-family overrides for one AutoModel experiment."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from typing import Any

from groot.infra.launcher.one_launcher.automodel_topology import AutoModelTopology


@dataclass(frozen=True)
class AutoModelComputeVariant:
    """Topology, recipe, and environment deltas for one compute family."""

    topology: AutoModelTopology
    recipe_overrides: Mapping[str, Any] = field(default_factory=dict)
    recipe_delete_paths: tuple[str, ...] = ()
    env_vars: Mapping[str, str | None] = field(default_factory=dict)
    launch_blocker: str = ""

    def recipe_from(self, baseline: Mapping[str, Any]) -> dict[str, Any]:
        recipe = copy.deepcopy(dict(baseline))
        self.merge(recipe, self.recipe_overrides)
        for path in self.recipe_delete_paths:
            self.delete_path(recipe, path)
        return recipe

    def environment_from(self, baseline: Mapping[str, str]) -> dict[str, str]:
        environment = dict(baseline)
        for name, value in self.env_vars.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[str(name)] = str(value)
        return environment

    @classmethod
    def merge(cls, target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            current = target.get(key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                cls.merge(current, value)
            else:
                target[key] = copy.deepcopy(value)

    @classmethod
    def delete_path(cls, target: dict[str, Any], path: str) -> None:
        parts = path.split(".")
        if not path or any(not part for part in parts):
            raise ValueError(f"Invalid recipe delete path {path!r}")
        current = target
        for part in parts[:-1]:
            nested = current.get(part)
            if not isinstance(nested, dict):
                raise TypeError(f"Recipe delete path {path!r} does not resolve to a mapping")
            current = nested
        if parts[-1] not in current:
            raise ValueError(f"Recipe delete path {path!r} does not exist")
        del current[parts[-1]]
