# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime artifacts produced before dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetBuildArtifacts:
    """Computed dataset-build inputs that are not user configuration."""

    deltacat_stats_by_embodiment: dict[str, dict[str, Any]] = field(default_factory=dict)
