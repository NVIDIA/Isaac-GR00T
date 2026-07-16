# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-compute-family experiment sizing facts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentComputeConfig:
    per_device_batch_size: int
