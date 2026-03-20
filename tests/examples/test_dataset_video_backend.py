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

from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
from gr00t.utils import video_utils
import huggingface_hub
import pytest
from test_support.runtime import SHARED_DRIVE_ROOT, get_root


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


SHARED_DATASETS_ROOT = SHARED_DRIVE_ROOT / "datasets"


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
    DatasetCatalogEntry(
        "g1_locomanip_embodiment_sim",
        "examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        hf_repo_id="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        hf_files=(
            "unitree_g1.LMPnPAppleToPlateDC/videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        ),
    ),
)


@pytest.mark.parametrize("entry", DATASET_CATALOG, ids=lambda e: e.name)
def test_dataset_backend_policy_on_sample_video(entry: DatasetCatalogEntry) -> None:
    """Verify that torchcodec decodes non-identical frames for each dataset video."""
    video_paths = entry.list_videos()
    assert len(video_paths) > 0, (
        f"No videos found for dataset entry {entry.name} in {entry.get_local_directory()}"
    )

    for video_path in video_paths:
        video_path_str = str(video_path)

        info = video_utils._get_video_info_ffmpeg(video_path_str)
        nb_frames = int(info["nb_frames"])
        if nb_frames < 10:
            raise ValueError(f"Video has too few frames: {video_path_str}")

        nb_frames = min(nb_frames, 60)

        frames = video_utils.get_frames_by_indices(
            video_path=video_path_str,
            indices=list(range(nb_frames)),
            video_backend="torchcodec",
            video_backend_kwargs={},
        )
        assert len(frames) == nb_frames
        all_identical = all((frames[i] == frames[0]).all() for i in range(1, nb_frames))
        if all_identical:
            debug_dir = REPO_ROOT / "debug_video_decoding" / entry.name
            debug_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(video_path, debug_dir / video_path.name)

            h, w = frames[0].shape[:2]
            fps = float(info.get("fps", 1.0)) or 1.0
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
