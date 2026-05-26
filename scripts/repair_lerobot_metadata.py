#!/usr/bin/env python3

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

"""Repair LeRobot metadata after downloading a partial or imperfect dataset.

Some Hugging Face dataset snapshots can contain episode metadata for files that
are not present in the remote repo. This script drops those broken episodes from
``meta/episodes.jsonl``, updates summary fields in ``meta/info.json``, and
regenerates stats for changed datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class RepairResult:
    dataset_path: Path
    total_episodes: int
    kept_episodes: int
    dropped_episodes: int
    missing_examples: tuple[str, ...]
    changed: bool


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            tmp = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(record) + "\n" for record in records))


def _format_episode_path(
    dataset_path: Path,
    pattern: str,
    *,
    chunks_size: int,
    episode_index: int,
    video_key: str | None = None,
    mask_key: str | None = None,
) -> Path:
    episode_chunk = episode_index // chunks_size
    return dataset_path / pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key or "",
        mask_key=mask_key or "",
    )


def _original_keys(modality: dict[str, Any], modality_name: str) -> list[str]:
    entries = modality.get(modality_name) or {}
    if not isinstance(entries, dict):
        return []
    return [
        str(value.get("original_key") or key)
        for key, value in entries.items()
        if isinstance(value, dict)
    ]


def _required_episode_files(
    dataset_path: Path,
    info: dict[str, Any],
    modality: dict[str, Any],
    episode_index: int,
) -> list[Path]:
    chunks_size = int(info["chunks_size"])
    paths = [
        _format_episode_path(
            dataset_path,
            info["data_path"],
            chunks_size=chunks_size,
            episode_index=episode_index,
        )
    ]

    video_path = info.get("video_path")
    if video_path:
        paths.extend(
            _format_episode_path(
                dataset_path,
                video_path,
                chunks_size=chunks_size,
                episode_index=episode_index,
                video_key=video_key,
            )
            for video_key in _original_keys(modality, "video")
        )

    mask_path = info.get("mask_path")
    if mask_path:
        paths.extend(
            _format_episode_path(
                dataset_path,
                mask_path,
                chunks_size=chunks_size,
                episode_index=episode_index,
                mask_key=mask_key,
                video_key=mask_key,
            )
            for mask_key in _original_keys(modality, "mask")
        )

    return paths


def _file_index(
    dataset_path: Path, roots: tuple[str, ...] = ("data", "videos", "masks")
) -> set[str]:
    files: set[str] = set()
    for root in roots:
        root_path = dataset_path / root
        if not root_path.is_dir():
            continue
        for path in root_path.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(dataset_path).as_posix())
    return files


def _missing_required_files(
    dataset_path: Path,
    info: dict[str, Any],
    modality: dict[str, Any],
    episode: dict[str, Any],
    existing_files: set[str],
) -> list[str]:
    episode_index = int(episode["episode_index"])
    missing = []
    for path in _required_episode_files(dataset_path, info, modality, episode_index):
        rel = path.relative_to(dataset_path).as_posix()
        if rel not in existing_files:
            missing.append(rel)
    return missing


def _update_info(
    info: dict[str, Any], episodes: list[dict[str, Any]], modality: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(info)
    updated["total_episodes"] = len(episodes)
    updated["total_frames"] = sum(int(episode.get("length", 0)) for episode in episodes)
    if "total_videos" in updated:
        updated["total_videos"] = len(episodes) * len(_original_keys(modality, "video"))
    if "total_chunks" in updated:
        chunks_size = int(updated["chunks_size"])
        updated["total_chunks"] = len(
            {int(episode["episode_index"]) // chunks_size for episode in episodes}
        )
    if isinstance(updated.get("splits"), dict) and "train" in updated["splits"]:
        updated["splits"] = dict(updated["splits"])
        updated["splits"]["train"] = f"0:{len(episodes)}"
    return updated


def _regenerate_stats(dataset_path: Path, embodiment_tag: str | None) -> None:
    from gr00t.data.stats import generate_rel_stats, generate_stats

    stats_path = dataset_path / "meta" / "stats.json"
    rel_stats_path = dataset_path / "meta" / "relative_stats.json"
    stats_path.unlink(missing_ok=True)
    rel_stats_path.unlink(missing_ok=True)
    generate_stats(dataset_path)

    if embodiment_tag:
        from gr00t.data.embodiment_tags import EmbodimentTag

        generate_rel_stats(dataset_path, EmbodimentTag.resolve(embodiment_tag))


def repair_dataset(
    dataset_path: Path,
    *,
    embodiment_tag: str | None = None,
    regenerate_stats: bool = True,
    dry_run: bool = False,
) -> RepairResult:
    dataset_path = dataset_path.expanduser().resolve()
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    modality_path = meta_dir / "modality.json"

    info = _read_json(info_path)
    episodes = _read_jsonl(episodes_path)
    modality = _read_json(modality_path)
    existing_files = _file_index(dataset_path)

    kept: list[dict[str, Any]] = []
    dropped_episodes: list[dict[str, Any]] = []
    missing_examples: list[str] = []
    for episode in episodes:
        missing = _missing_required_files(dataset_path, info, modality, episode, existing_files)
        if missing:
            dropped_episodes.append(episode)
            if len(missing_examples) < 20:
                missing_examples.append(
                    f"episode {episode['episode_index']}: " + ", ".join(missing[:4])
                )
            continue
        kept.append(episode)

    dropped = len(episodes) - len(kept)
    if dropped == 0:
        return RepairResult(dataset_path, len(episodes), len(kept), 0, tuple(), False)
    if not kept:
        raise RuntimeError(
            f"All {len(episodes)} episode(s) in {dataset_path} reference missing files; "
            "refusing to write an empty dataset."
        )

    if not dry_run:
        updated_info = _update_info(info, kept, modality)
        _write_json(info_path, updated_info)
        _write_jsonl(episodes_path, kept)
        for episode in dropped_episodes:
            episode_index = int(episode["episode_index"])
            for path in _required_episode_files(dataset_path, info, modality, episode_index):
                path.unlink(missing_ok=True)
        if regenerate_stats:
            _regenerate_stats(dataset_path, embodiment_tag)

    return RepairResult(
        dataset_path, len(episodes), len(kept), dropped, tuple(missing_examples), True
    )


def _expand_dataset_args(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for part in value.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            matches = glob.glob(part)
            if matches:
                paths.extend(Path(match) for match in sorted(matches))
            else:
                paths.append(Path(part))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help="LeRobot dataset path(s). Each arg may also be an os.pathsep-separated list.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default=None,
        help="Optional embodiment tag used to regenerate relative stats after repair.",
    )
    parser.add_argument(
        "--no-regenerate-stats",
        action="store_true",
        help="Only rewrite metadata; leave stats regeneration to the training pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report broken episodes only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _expand_dataset_args(args.dataset_paths)
    if not paths:
        raise SystemExit("No dataset paths provided.")

    had_changes = False
    for path in paths:
        result = repair_dataset(
            path,
            embodiment_tag=args.embodiment_tag,
            regenerate_stats=not args.no_regenerate_stats,
            dry_run=args.dry_run,
        )
        had_changes = had_changes or result.changed
        if result.changed:
            action = "would drop" if args.dry_run else "dropped"
            print(
                f"{result.dataset_path}: {action} {result.dropped_episodes}/"
                f"{result.total_episodes} broken episode(s)"
            )
            for missing in result.missing_examples:
                print(f"  {missing}")
        else:
            print(f"{result.dataset_path}: metadata already matches available files")

    if args.dry_run and had_changes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
