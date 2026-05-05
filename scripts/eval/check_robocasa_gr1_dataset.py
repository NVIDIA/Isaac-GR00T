#!/usr/bin/env python3
"""Check RoboCasa GR1 LeRobot dataset completeness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/home/d024/DiT4DiT/playground/Datasets/nvidia/"
    "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
)
DEFAULT_PATTERN = "gr1_unified.*GR1ArmsAndWaistFourierHands_1000"
EXPECTED_EPISODES = 1000
REQUIRED_META_FILES = (
    "episodes.jsonl",
    "info.json",
    "modality.json",
    "relative_stats.json",
    "stats.json",
    "tasks.jsonl",
)


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    meta_lines: int
    data_count: int
    video_count: int
    zero_data_count: int
    zero_video_count: int
    missing_meta: list[str]
    missing_data: list[str]
    missing_videos: list[str]

    @property
    def ok(self) -> bool:
        return (
            self.meta_lines == EXPECTED_EPISODES
            and self.data_count == EXPECTED_EPISODES
            and self.video_count == EXPECTED_EPISODES
            and self.zero_data_count == 0
            and self.zero_video_count == 0
            and not self.missing_meta
            and not self.missing_data
            and not self.missing_videos
        )


def count_episode_lines(dataset_dir: Path) -> int:
    episodes_file = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        return 0
    with episodes_file.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def check_dataset(dataset_dir: Path, expected_episodes: int) -> DatasetReport:
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data" / "chunk-000"
    video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.ego_view"

    data_files = list((dataset_dir / "data").rglob("episode_*.parquet"))
    video_files = list((dataset_dir / "videos").rglob("episode_*.mp4"))

    missing_meta = [name for name in REQUIRED_META_FILES if not (meta_dir / name).is_file()]
    missing_data = []
    missing_videos = []

    for episode_idx in range(expected_episodes):
        episode_name = f"episode_{episode_idx:06d}"
        if not (data_dir / f"{episode_name}.parquet").is_file():
            missing_data.append(f"{episode_name}.parquet")
        if not (video_dir / f"{episode_name}.mp4").is_file():
            missing_videos.append(f"{episode_name}.mp4")

    return DatasetReport(
        path=dataset_dir,
        meta_lines=count_episode_lines(dataset_dir),
        data_count=len(data_files),
        video_count=len(video_files),
        zero_data_count=sum(1 for path in data_files if path.stat().st_size == 0),
        zero_video_count=sum(1 for path in video_files if path.stat().st_size == 0),
        missing_meta=missing_meta,
        missing_data=missing_data,
        missing_videos=missing_videos,
    )


def print_missing(label: str, missing: list[str], max_items: int) -> None:
    if not missing:
        return
    shown = missing[:max_items]
    print(f"  missing {label} ({len(missing)}):")
    for item in shown:
        print(f"    {item}")
    if len(missing) > max_items:
        print(f"    ... {len(missing) - max_items} more")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check meta/data/videos completeness for RoboCasa GR1 datasets."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--expected-episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--max-missing-to-print", type=int, default=30)
    args = parser.parse_args()

    dataset_dirs = sorted(path for path in args.data_root.glob(args.pattern) if path.is_dir())
    if not dataset_dirs:
        print(f"No datasets matched: {args.data_root / args.pattern}")
        return 2

    reports = [check_dataset(path, args.expected_episodes) for path in dataset_dirs]

    print(f"Data root: {args.data_root}")
    print(f"Matched datasets: {len(reports)}")
    print(f"Expected episodes per dataset: {args.expected_episodes}")
    print()

    for report in reports:
        name = report.path.name
        if report.ok:
            print(f"{name} meta=OK data=OK videos=OK")
            continue

        print(
            f"{name} CHECK "
            f"meta_lines={report.meta_lines} "
            f"data={report.data_count} "
            f"videos={report.video_count} "
            f"zero_data={report.zero_data_count} "
            f"zero_video={report.zero_video_count}"
        )
        print_missing("meta", report.missing_meta, args.max_missing_to_print)
        print_missing("data", report.missing_data, args.max_missing_to_print)
        print_missing("videos", report.missing_videos, args.max_missing_to_print)

    bad_reports = [report for report in reports if not report.ok]
    print()
    if bad_reports:
        print(f"FAILED: {len(bad_reports)} / {len(reports)} datasets have missing or empty files.")
        return 1

    total_episodes = len(reports) * args.expected_episodes
    print(f"OK: all {len(reports)} datasets are complete ({total_episodes} episodes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
