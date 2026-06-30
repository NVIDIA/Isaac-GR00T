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
Video Backend Codec Validation Tests

Verifies that the torchcodec video backend decodes non-identical frames across
a representative set of robotics datasets. On failure, the original video and a
re-encoded debug copy are written to ``debug_video_decoding/<dataset_name>/``
for offline inspection.

Datasets are resolved in order: shared drive, in-repo path, then downloaded
from HuggingFace Hub using ``hf_hub_download`` (avoids full repo enumeration).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig
from gr00t.utils import video_utils
import huggingface_hub
import numpy as np
import pytest
from test_support.runtime import TEST_CACHE_PATH, get_root, resolve_demo_dataset


REPO_ROOT = get_root()


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Catalog entry describing a robotics dataset and how to obtain it.

    Datasets are resolved in order: shared drive path, in-repo path, then
    downloaded from HuggingFace Hub if neither exists locally. The
    hf_files field pins the download to a single representative video
    to avoid fetching the full dataset during CI.
    """

    name: str
    rel_path: str
    hf_repo_id: str | None = None
    hf_files: tuple[str, ...] | None = None

    _VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    @staticmethod
    def _scan_videos(directory: Path) -> list[Path]:
        """Return all video files under directory, sorted by path."""
        return sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in DatasetCatalogEntry._VIDEO_SUFFIXES
        )

    def list_videos(self) -> list[Path]:
        """Resolve the local dataset directory and return all video files within it."""
        return self._scan_videos(self.get_local_directory())

    def download(self, dest: Path) -> None:
        """Download the dataset from HuggingFace Hub into dest.

        Uses hf_hub_download for each file in hf_files to avoid enumerating
        the entire repo index, which can be very slow for large datasets.
        Does nothing if hf_repo_id or hf_files is not set.
        """
        if self.hf_repo_id is None or not self.hf_files:
            return
        dest.mkdir(parents=True, exist_ok=True)
        for file_path in self.hf_files:
            huggingface_hub.hf_hub_download(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                filename=file_path,
                local_dir=str(dest),
            )

    def get_local_directory(self) -> Path:
        """Return the local directory containing this dataset's videos.

        Checks the shared drive path then the in-repo path. If neither exists,
        downloads the dataset via HuggingFace Hub. Raises FileNotFoundError if
        no videos are found after downloading.
        """
        shared_path = SHARED_DATASETS_ROOT / self.rel_path
        repo_path = REPO_ROOT / self.rel_path

        for candidate in (shared_path, repo_path):
            if candidate.exists() and self._scan_videos(candidate):
                return candidate

        self.download(shared_path)

        if shared_path.exists() and self._scan_videos(shared_path):
            return shared_path

        raise FileNotFoundError(
            f"Failed to find or download dataset videos for entry {self.name}. "
            f"Checked shared path: {shared_path} and repo path: {repo_path}"
        )


SHARED_DATASETS_ROOT = TEST_CACHE_PATH / "datasets"


DATASET_CATALOG: tuple[DatasetCatalogEntry, ...] = (
    DatasetCatalogEntry(
        "so100_finish_sandwich",
        "examples/SO100/finish_sandwich_lerobot",
        hf_repo_id="izuluaga/finish_sandwich",
        hf_files=("videos/observation.images.front/chunk-000/file-000.mp4",),
    ),
    DatasetCatalogEntry(
        "libero_10_lerobot",
        "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot",
        hf_repo_id="IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        hf_files=("videos/chunk-000/observation.images.image/episode_000000.mp4",),
    ),
    DatasetCatalogEntry(
        "libero_goal_lerobot",
        "examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot",
        hf_repo_id="IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        hf_files=("videos/chunk-000/observation.images.image/episode_000000.mp4",),
    ),
    DatasetCatalogEntry(
        "libero_object_lerobot",
        "examples/LIBERO/libero_object_no_noops_1.0.0_lerobot",
        hf_repo_id="IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        hf_files=("videos/chunk-000/observation.images.image/episode_000000.mp4",),
    ),
    DatasetCatalogEntry(
        "libero_spatial_lerobot",
        "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot",
        hf_repo_id="IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        hf_files=("videos/chunk-000/observation.images.image/episode_000000.mp4",),
    ),
    DatasetCatalogEntry(
        "simplerenv_bridge_lerobot",
        "examples/SimplerEnv/bridge_orig_lerobot",
        hf_repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
        hf_files=("videos/chunk-000/observation.images.image_0/episode_000000.mp4",),
    ),
    DatasetCatalogEntry(
        "simplerenv_fractal_lerobot",
        "examples/SimplerEnv/fractal20220817_data_lerobot",
        hf_repo_id="IPEC-COMMUNITY/fractal20220817_data_lerobot",
        hf_files=("videos/chunk-000/observation.images.image/episode_000000.mp4",),
    ),
)


SO100_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)), modality_keys=["single_arm", "gripper"]
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


SO101_MASK_MODALITY_CONFIG = {
    **SO100_MODALITY_CONFIG,
    "mask": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
}


@dataclass(frozen=True)
class ReadmeDemoDatasetEntry:
    """Concrete README training dataset that ships under demo_data/."""

    name: str
    dataset_name: str
    modality_configs: dict[str, ModalityConfig]
    env_var: str | None = None

    @property
    def video_keys(self) -> list[str]:
        return self.modality_configs["video"].modality_keys

    @property
    def mask_keys(self) -> list[str]:
        mask_config = self.modality_configs.get("mask")
        return [] if mask_config is None else mask_config.modality_keys


README_DEMO_DATASET_CATALOG: tuple[ReadmeDemoDatasetEntry, ...] = (
    ReadmeDemoDatasetEntry(
        name="readme_droid_sample",
        dataset_name="droid_sample",
        modality_configs=MODALITY_CONFIGS["oxe_droid_relative_eef_relative_joint"],
        env_var="DROID_DEMO_DATASET_PATH",
    ),
    ReadmeDemoDatasetEntry(
        name="readme_libero_demo",
        dataset_name="libero_demo",
        modality_configs=MODALITY_CONFIGS["libero_sim"],
        env_var="LIBERO_DEMO_DATASET_PATH",
    ),
    ReadmeDemoDatasetEntry(
        name="readme_simplerenv_bridge_sample",
        dataset_name="simplerenv_bridge_sample",
        modality_configs=MODALITY_CONFIGS["simpler_env_widowx"],
    ),
    ReadmeDemoDatasetEntry(
        name="readme_simplerenv_fractal_sample",
        dataset_name="simplerenv_fractal_sample",
        modality_configs=MODALITY_CONFIGS["simpler_env_google"],
    ),
    ReadmeDemoDatasetEntry(
        name="readme_cube_to_bowl_5",
        dataset_name="cube_to_bowl_5",
        modality_configs=SO100_MODALITY_CONFIG,
    ),
    ReadmeDemoDatasetEntry(
        name="readme_cube_to_bowl_5_with_mask",
        dataset_name="cube_to_bowl_5_with_mask",
        modality_configs=SO101_MASK_MODALITY_CONFIG,
    ),
)


@pytest.fixture(scope="module")
def video_decoder_cls():
    try:
        return video_utils._get_video_decoder_cls()
    except ImportError as exc:
        pytest.skip(str(exc))


@pytest.mark.edge_device
@pytest.mark.parametrize("entry", DATASET_CATALOG, ids=lambda e: e.name)
def test_dataset_backend_policy_on_sample_video(
    entry: DatasetCatalogEntry, video_decoder_cls
) -> None:
    """Verify that torchcodec decodes non-identical frames for each dataset video."""
    video_paths = entry.list_videos()
    assert len(video_paths) > 0, (
        f"No videos found for dataset entry {entry.name} in {entry.get_local_directory()}"
    )

    for video_path in video_paths:
        video_path_str = str(video_path)

        decoder = video_decoder_cls(video_path_str)
        nb_frames = len(decoder)
        fps = float(decoder.metadata.average_fps or 1.0)

        if nb_frames < 10:
            raise ValueError(f"Video has too few frames: {video_path_str}")

        nb_frames = min(nb_frames, 60)

        frames = video_utils.get_frames_by_indices(
            video_path=video_path_str,
            indices=list(range(nb_frames)),
            decoder_kwargs={},
        )
        assert len(frames) == nb_frames
        all_identical = all((frames[i] == frames[0]).all() for i in range(1, nb_frames))
        if all_identical:
            debug_dir = REPO_ROOT / "debug_video_decoding" / entry.name
            debug_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(video_path, debug_dir / video_path.name)

            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(
                str(debug_dir / f"decoded_{video_path.stem}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            pytest.fail(
                f"All {nb_frames} decoded frames are identical in {video_path_str}. "
                f"Original video and decoded frames saved to {debug_dir}"
            )


@pytest.mark.edge_device
@pytest.mark.parametrize("entry", README_DEMO_DATASET_CATALOG, ids=lambda e: e.name)
def test_readme_demo_training_dataset_loads_frames_with_torchcodec(
    entry: ReadmeDemoDatasetEntry,
    video_decoder_cls,
) -> None:
    """Load concrete README demo training datasets through the real episode loader."""
    assert video_decoder_cls is not None

    dataset_path = resolve_demo_dataset(
        dataset_name=entry.dataset_name,
        path_override_env=entry.env_var,
        repo_root=REPO_ROOT,
    )
    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=copy.deepcopy(entry.modality_configs),
        decoder_kwargs={},
    )
    assert len(loader) > 0, f"{entry.name} has no episodes: {dataset_path}"

    episode = loader[0]
    assert len(episode) > 0, f"{entry.name} episode 0 loaded no rows: {dataset_path}"

    for video_key in entry.video_keys:
        column = f"video.{video_key}"
        assert column in episode.columns, (
            f"{entry.name} did not load expected video column {column}. "
            f"Available columns: {list(episode.columns)}"
        )
        frame = episode[column].iloc[0]
        assert isinstance(frame, np.ndarray), (
            f"{entry.name} {column} loaded {type(frame).__name__}, expected np.ndarray"
        )
        assert frame.ndim == 3 and frame.shape[-1] == 3, (
            f"{entry.name} {column} loaded frame with unexpected shape {frame.shape}"
        )
        assert frame.dtype == np.uint8, (
            f"{entry.name} {column} loaded frame with unexpected dtype {frame.dtype}"
        )

    for mask_key in entry.mask_keys:
        column = f"mask.{mask_key}"
        assert column in episode.columns, (
            f"{entry.name} did not load expected mask column {column}. "
            f"Available columns: {list(episode.columns)}"
        )
        mask = episode[column].iloc[0]
        assert isinstance(mask, np.ndarray), (
            f"{entry.name} {column} loaded {type(mask).__name__}, expected np.ndarray"
        )
        assert mask.ndim >= 2, f"{entry.name} {column} loaded mask shape {mask.shape}"
