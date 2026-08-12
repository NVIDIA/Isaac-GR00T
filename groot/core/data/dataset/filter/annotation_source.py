# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

import polars as pl

from groot.core.data.types import EpisodeData, OnlineDataFilter


class AnnotationSourceFilter(OnlineDataFilter):
    """Filter the data to only include subtasks from a specific source."""

    def __init__(self, source: str):
        self.source = source

    def __call__(self, df: EpisodeData, meta: dict | None = None) -> EpisodeData:
        if "language.sub_task" in df.columns:
            sub_tasks_all_steps: list[str] = df["language.sub_task"].to_list()
            sub_tasks_filtered: list[str] = []
            for sub_tasks in sub_tasks_all_steps:
                sub_tasks_per_step = json.loads(sub_tasks)
                sub_tasks_per_step = [
                    sub_task
                    for sub_task in sub_tasks_per_step
                    if self.source in sub_task.get("source", "")
                ]
                sub_tasks_filtered.append(json.dumps(sub_tasks_per_step))
            df = df.with_columns(pl.Series("language.sub_task", sub_tasks_filtered))
        return df
