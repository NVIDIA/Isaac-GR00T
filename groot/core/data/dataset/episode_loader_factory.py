# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Factory for creating the correct episode loader based on dataset path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from groot.core.data import types as data_types
from groot.core.data.dataset.path_detection import is_lance_source, is_packds_path


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
    view_id: str | None = None,
    datasets_root: str | None = None,
    dataset_name: str | None = None,
    lance_as_of: str | None = None,
    lance_step_version: int | None = None,
    lance_media_version: int | None = None,
    lance_episodes_version: int | None = None,
    lance_registry_as_of: str | None = None,
    lance_step_filter: list[dict[str, Any]] | None = None,
):
    """Create an episode loader, auto-selecting the backend from the path.

    - LanceDS stores (a directory containing step.lance, or an s3:// path given any of the entry
      fields below: path_detection.is_lance_source) use LanceEpisodeLoader, whatever a ``.packds``
      or ``yam_packds*`` name says, as the factory and the pin check read them; with the kwargs
      factory._lance_episode_loader_kwargs takes from the store's entry and the run's DataConfig:

      - view_id: the entry's lance_view_id;
      - lance_as_of, lance_step_version, lance_media_version, lance_episodes_version,
        lance_step_filter: the entry's fields of those names;
      - datasets_root: data.lance_datasets_root;
      - dataset_name: the store's path relative to it, its registry key;
      - lance_registry_as_of: data.lance_registry_as_of.
    - Other PackDS paths use PackDSEpisodeLoader.
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

    lance_fields = {
        "lance_view_id": view_id,
        "lance_as_of": lance_as_of,
        "lance_step_version": lance_step_version,
        "lance_media_version": lance_media_version,
        "lance_episodes_version": lance_episodes_version,
        "lance_step_filter": lance_step_filter,
    }
    if is_lance_source(path_str, lance_fields):
        from groot.core.data.dataset.lance.read.episode_loader import LanceEpisodeLoader

        return LanceEpisodeLoader(
            dataset_path=path_str,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            resample_on_invalid_frames=resample_on_invalid_frames,
            data_filter=data_filter,
            hydrator=hydrator,
            view_id=view_id,
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            lance_as_of=lance_as_of,
            lance_step_version=lance_step_version,
            lance_media_version=lance_media_version,
            lance_episodes_version=lance_episodes_version,
            lance_registry_as_of=lance_registry_as_of,
            lance_step_filter=lance_step_filter,
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
