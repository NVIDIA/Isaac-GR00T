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

import random

import polars as pl

from groot.core.data.types import EpisodeData, OnlineDataFilter


class Sys2Filter(OnlineDataFilter):
    """Filter the data to only include subtasks from a specific source and handle language modes."""

    def __init__(
        self,
        source: str,
        language_mode: str = "default",
        subtask_frac: float = 1.0,
    ):

        self.source = source.split("|") if source is not None else None
        self.language_mode = language_mode
        self.subtask_frac = subtask_frac

    def create_language_from_meta(
        self, episode_meta: dict, modality_configs: dict, nframes: int, lang_key: str
    ) -> list[str]:
        task_language = random.choice(episode_meta["tasks"])

        if self.language_mode == "task_only":
            return [task_language] * nframes

        # For all other modes, we need to compute subtask languages
        sub_tasks = episode_meta.get("sub_tasks", [])

        # Filter subtasks by source if filter is specified
        if self.source is not None:
            sub_tasks = [
                st for st in sub_tasks if any(src in st.get("source", "") for src in self.source)
            ]

        # extract subtask languages
        action_delta_indices = modality_configs["action"].delta_indices
        action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        subtask_languages = [[] for _ in range(nframes)]
        for sub_task in sub_tasks:
            start_idx, end_idx, sub_text = sub_task["start"], sub_task["end"], sub_task["text"]
            horizon = action_horizon // 2
            for i in range(max(0, start_idx - horizon), end_idx):
                if i < 0 or i >= nframes:
                    continue
                subtask_languages[i].append(sub_text)

        # Pick one subtask per frame (or empty string if none)
        subtask_languages = [
            random.choice(lst) if lst and random.random() < self.subtask_frac else ""
            for lst in subtask_languages
        ]

        if self.language_mode == "subtask_only":
            # Use subtask only, empty string if no subtask available
            return subtask_languages

        elif self.language_mode == "subtask_fallback":
            # Use subtask if available, fallback to task language
            new_languages = []
            for subtask_lang in subtask_languages:
                if not subtask_lang:
                    new_languages.append(task_language)
                else:
                    new_languages.append(subtask_lang)
            return new_languages

        elif self.language_mode == "subtask_concat":
            # Concatenate task and subtask (if available)
            new_languages = []
            for subtask_lang in subtask_languages:
                if subtask_lang:
                    new_languages.append(f"{task_language}; {subtask_lang}")
                else:
                    new_languages.append(task_language)
            return new_languages

        else:
            raise ValueError(
                f"Language mode '{self.language_mode}' not supported. Use: task_only, subtask_only, subtask_fallback, subtask_concat"
            )

    def __call__(self, df: EpisodeData, meta: dict | None = None) -> EpisodeData:
        assert meta is not None, "Sys2Filter requires meta dict with episode info"

        modality_configs = meta.get("modality_configs")
        assert modality_configs is not None, "Sys2Filter requires modality_configs in meta"
        assert "language" in modality_configs, "Sys2Filter requires 'language' in modality_configs"

        if self.language_mode != "default":
            lang_key = modality_configs["language"].modality_keys[0]
            new_languages = self.create_language_from_meta(
                meta, modality_configs, df.height, lang_key
            )
            df = df.with_columns(pl.Series("language." + lang_key, new_languages))

        return df
