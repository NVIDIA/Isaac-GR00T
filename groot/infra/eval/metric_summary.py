# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Metric summary produced by open-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    per_trajectory: list[dict[str, float | int]]
    average: dict[str, float]

    @classmethod
    def init_from_trajectory_metrics(
        cls,
        rows: list[dict[str, float | int]],
        metrics: Mapping[str, list[float]],
    ) -> MetricSummary:
        average = {key: float(sum(values) / len(values)) for key, values in metrics.items()}
        return cls(per_trajectory=rows, average=average)
