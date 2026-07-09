# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
