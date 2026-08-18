#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Render environment variables in an AutoResearch YAML config."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml

_ENVIRONMENT_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any, source: Path) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"{source}: environment variable {name} is not set")
            return os.environ[name]

        return _ENVIRONMENT_VARIABLE_RE.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, source) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, source) for key, item in value.items()}
    return value


def main() -> None:
    source, destination = map(Path, sys.argv[1:])
    config = _expand(yaml.safe_load(source.read_text(encoding="utf-8")), source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
