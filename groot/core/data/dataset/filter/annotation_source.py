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
