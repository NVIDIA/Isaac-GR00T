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

"""Factory for creating the correct episode loader based on dataset path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from groot.core.data import types as data_types
from groot.core.data.dataset.path_detection import is_packds_path


def create_episode_loader(
    dataset_path: str | Path,
    modality_configs: dict[str, data_types.ModalityConfig],
    *,
    video_backend: str = "torchcodec",
    video_backend_kwargs: dict[str, Any] | None = None,
    resample_on_invalid_frames: bool = False,
    data_filter: data_types.OnlineDataFilter | None = None,
    hydrator: Any | None = None,
    episode_query: str | None = None,
    steps_query: str | None = None,
    as_of_timestamp: int | None = None,
    key_remap: dict[str, str] | None = None,
):
    """Create an episode loader, auto-selecting the backend from the path.

    - PackDS paths use PackDSEpisodeLoader.
    - Otherwise uses LeRobotEpisodeLoader.
    """
    path_str = str(dataset_path).rstrip("/")

    if path_str.startswith("dc://"):
        from groot.core.data.dataset.packds_episode_loader import DeltaCatPackDSEpisodeLoader

        return DeltaCatPackDSEpisodeLoader(
            dataset_path=path_str,
            modality_configs=modality_configs,
            data_filter=data_filter,
            episode_query=episode_query,
            steps_query=steps_query,
            as_of_timestamp=as_of_timestamp,
            key_remap=key_remap,
        )

    if is_packds_path(path_str):
        from groot.core.data.dataset.packds_episode_loader import PackDSEpisodeLoader

        return PackDSEpisodeLoader(
            dataset_path=path_str,
            modality_configs=modality_configs,
            data_filter=data_filter,
        )

    from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    return LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
        video_backend=video_backend,
        video_backend_kwargs=video_backend_kwargs,
        resample_on_invalid_frames=resample_on_invalid_frames,
        data_filter=data_filter,
        hydrator=hydrator,
    )
