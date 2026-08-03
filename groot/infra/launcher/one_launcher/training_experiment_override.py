# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A typed dotted-path override for a resolved training experiment."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence
import copy
from dataclasses import dataclass
import math
from typing import Any

import yaml

_UNSET = object()


@dataclass(frozen=True)
class TrainingExperimentOverride:
    expression: str
    path: tuple[str, ...]
    value: Any

    @classmethod
    def init_from_expression(cls, expression: str) -> TrainingExperimentOverride:
        raw_path, separator, raw_value = expression.partition("=")
        if not separator:
            raise ValueError(f"--set expects PATH=VALUE, got {expression!r}")
        path = tuple(part.strip() for part in raw_path.strip().split("."))
        if not path or any(not part for part in path):
            raise ValueError(f"--set path must contain non-empty dotted segments: {expression!r}")
        if path[0] == "training":
            normalized = tuple(part.replace("-", "_") for part in path[1:])
            if normalized and normalized[0] == "model":
                path = ("training", "model", *path[2:])
            else:
                path = ("training", *normalized)
        elif path[0] == "compute":
            normalized = tuple(part.replace("-", "_") for part in path[1:])
            if normalized and normalized[0] == "env_vars":
                path = ("compute", "env_vars", *path[2:])
            else:
                path = ("compute", *normalized)
        elif path[0] == "data" and len(path) > 1:
            if path[1].replace("-", "_") == "launcher_data":
                path = ("data", "launcher_data", *path[2:])
        elif path[0] == "evaluation":
            path = ("evaluation", *(part.replace("-", "_") for part in path[1:]))
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value in --set {expression!r}: {exc}") from exc
        override = cls(expression=expression, path=path, value=value)
        override.validate_value(value)
        return override

    def apply(
        self,
        target: MutableMapping[str, Any],
        *,
        alias_aware: bool = False,
        path: tuple[str, ...] | None = None,
    ) -> None:
        override_path = path or self.path
        current: Any = target
        for position, segment in enumerate(override_path):
            last = position == len(override_path) - 1
            if isinstance(current, MutableMapping):
                key = self.mapping_key(
                    current,
                    segment,
                    alias_aware=alias_aware,
                    requires_container=not last,
                )
                if last:
                    current[key] = copy.deepcopy(self.value)
                    return
                if key not in current:
                    current[key] = {}
                current = current[key]
                continue
            if isinstance(current, MutableSequence):
                index = self.sequence_index(segment, len(current))
                if last:
                    current[index] = copy.deepcopy(self.value)
                    return
                current = current[index]
                continue
            parent = ".".join(override_path[:position])
            raise ValueError(
                f"Cannot apply --set {self.expression!r}: {parent!r} is not a mapping or list"
            )

    def mapping_key(
        self,
        target: MutableMapping[Any, Any],
        segment: str,
        *,
        alias_aware: bool,
        requires_container: bool,
    ) -> str | int:
        exact_match = segment in target
        if exact_match and (
            not alias_aware
            or not requires_container
            or isinstance(target[segment], MutableMapping | MutableSequence)
        ):
            return segment
        if alias_aware:
            normalized = segment.replace("-", "_")
            aliases = [
                key
                for key in target
                if isinstance(key, str) and key.replace("-", "_") == normalized
            ]
            if requires_container:
                container_aliases = [
                    key
                    for key in aliases
                    if isinstance(target[key], MutableMapping | MutableSequence)
                ]
                if len(container_aliases) == 1:
                    return container_aliases[0]
                if len(container_aliases) > 1:
                    raise ValueError(
                        f"Cannot apply --set {self.expression!r}: path segment {segment!r} "
                        f"matches multiple config keys {container_aliases}"
                    )
            if exact_match:
                return segment
            if len(aliases) == 1:
                return aliases[0]
            if len(aliases) > 1:
                raise ValueError(
                    f"Cannot apply --set {self.expression!r}: path segment {segment!r} "
                    f"matches multiple config keys {aliases}"
                )
        try:
            integer_key = int(segment)
        except ValueError:
            return segment
        if integer_key in target:
            return integer_key
        return segment

    def sequence_index(self, segment: str, length: int) -> int:
        try:
            index = int(segment)
        except ValueError as exc:
            raise ValueError(
                f"Cannot apply --set {self.expression!r}: list index {segment!r} is not an integer"
            ) from exc
        if index < 0 or index >= length:
            raise ValueError(
                f"Cannot apply --set {self.expression!r}: list index {index} is outside "
                f"the valid range 0..{length - 1}"
            )
        return index

    def validate_value(self, value: Any) -> None:
        if value is None or isinstance(value, str | bool | int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"--set {self.expression!r} must not contain NaN or infinity")
            return
        if isinstance(value, list):
            for item in value:
                self.validate_value(item)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not (key is None or isinstance(key, str | bool | int | float)):
                    raise ValueError(
                        f"--set {self.expression!r} contains unsupported mapping key {key!r}"
                    )
                if isinstance(key, float) and not math.isfinite(key):
                    raise ValueError(f"--set {self.expression!r} must not contain NaN or infinity")
                self.validate_value(item)
            return
        raise ValueError(
            f"--set {self.expression!r} contains unsupported YAML value type "
            f"{type(value).__name__}"
        )

    def validate_cli_value(self, value: Any = _UNSET) -> None:
        current = self.value if value is _UNSET else value
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        f"--set {self.expression!r} contains a training CLI mapping key that "
                        "is not a non-empty string"
                    )
                self.validate_cli_value(item)
            return
        if isinstance(current, list) and any(isinstance(item, Mapping | list) for item in current):
            raise ValueError(
                f"--set {self.expression!r} contains a nested list or mapping that cannot be "
                "represented by the training CLI"
            )
