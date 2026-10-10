#!/usr/bin/env python

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

"""
Faster LeRobot episode loader.

This loader keeps the original episode loading behavior available through
``__getitem__`` and adds a sparse video loading entrypoint through ``__call__``.
The sparse path decodes only the video frames needed by the requested shard step
indices while leaving the rest of the data loading pipeline unchanged.
"""

from typing import Any

import numpy as np
import pandas as pd

from .lerobot_episode_loader import LANG_KEYS, LeRobotEpisodeLoader


class LeRobotEpisodeLoaderFaster(LeRobotEpisodeLoader):
    """Episode loader that supports sparse video decoding for shard step indices."""

    def _get_video_indices_from_steps(
        self,
        step_indices: np.ndarray,
        video_delta_indices: list[int],
        episode_length: int,
        allow_padding: bool = False,
    ) -> np.ndarray:
        """
        Convert shard step indices into the exact video frame indices to decode.

        Args:
            step_indices: Step indices assigned to the current shard for one episode
            video_delta_indices: Temporal offsets configured for the video modality
            episode_length: Actual episode length after parquet trimming
            allow_padding: Whether to clamp frame indices into valid bounds

        Returns:
            Sorted unique frame indices needed by the requested steps
        """
        requested_indices = set()
        for step_index in step_indices:
            for delta_index in video_delta_indices:
                frame_index = int(step_index + delta_index)
                if allow_padding:
                    frame_index = max(0, min(frame_index, episode_length - 1))
                requested_indices.add(frame_index)
        return np.array(sorted(requested_indices), dtype=int)

    def __call__(
        self,
        idx: int,
        step_indices: np.ndarray,
        allow_padding: bool = False,
    ) -> pd.DataFrame:
        """
        Load one episode while decoding video only for the requested shard steps.

        Args:
            idx: Episode index to load
            step_indices: Step indices assigned to the current shard for this episode
            allow_padding: Whether to clamp temporal indices into valid bounds

        Returns:
            DataFrame with all non-video modalities loaded for the full episode and
            video columns populated only at frame indices needed by ``step_indices``
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        if "video" in self.modality_configs:
            # [speed up] Decode only the sparse video frames needed by this shard.
            video_indices = self._get_video_indices_from_steps(
                np.asarray(step_indices, dtype=int),
                self.modality_configs["video"].delta_indices,
                actual_length,
                allow_padding=allow_padding,
            )
            video_data = self._load_video_data(episode_id, video_indices)

            for key in video_data.keys():
                sparse_frames: list[Any | None] = [None] * len(df)
                for frame_index, frame in zip(video_indices, video_data[key]):
                    sparse_frames[int(frame_index)] = frame
                df[f"video.{key}"] = sparse_frames

        # [speed up] Keep mask loading unchanged to keep the optimization narrowly
        # focused on the video decoding bottleneck.
        mask_data = self._load_mask_data(episode_id, np.arange(actual_length))
        for key in mask_data.keys():
            assert len(mask_data[key]) == len(df), (
                f"Mask data for {key} has length {len(mask_data[key])} but dataframe has length {len(df)}"
            )
            df[f"mask.{key}"] = [mask for mask in mask_data[key]]

        return df
