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

from __future__ import annotations

import json
from pathlib import Path

from scripts.repair_lerobot_metadata import repair_dataset


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    _write_json(
        dataset / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "test",
            "total_episodes": 3,
            "total_frames": 60,
            "total_videos": 6,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 20,
            "splits": {"train": "0:3"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32"},
                "action": {"dtype": "float32"},
            },
        },
    )
    _write_json(
        dataset / "meta/modality.json",
        {
            "state": {},
            "action": {},
            "video": {
                "front": {"original_key": "observation.images.front"},
                "wrist": {"original_key": "observation.images.wrist"},
            },
        },
    )
    _write_jsonl(
        dataset / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["ok"], "length": 10},
            {"episode_index": 1, "tasks": ["missing"], "length": 20},
            {"episode_index": 2, "tasks": ["ok"], "length": 30},
        ],
    )
    _write_jsonl(dataset / "meta/tasks.jsonl", [{"task_index": 0, "task": "ok"}])
    _write_json(dataset / "meta/stats.json", {"observation.state": {"mean": [0.0]}})

    for episode_index in (0, 2):
        _touch(dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet")
        _touch(
            dataset / f"videos/chunk-000/observation.images.front/episode_{episode_index:06d}.mp4"
        )
        _touch(
            dataset / f"videos/chunk-000/observation.images.wrist/episode_{episode_index:06d}.mp4"
        )
    _touch(dataset / "data/chunk-000/episode_000001.parquet")
    return dataset


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_repair_dataset_drops_episodes_with_missing_required_files(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    result = repair_dataset(dataset, regenerate_stats=False)

    assert result.changed
    assert result.dropped_episodes == 1
    assert "episode 1" in result.missing_examples[0]
    episodes = _read_jsonl(dataset / "meta/episodes.jsonl")
    assert [episode["episode_index"] for episode in episodes] == [0, 2]
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 40
    assert info["total_videos"] == 4
    assert info["total_chunks"] == 1
    assert info["splits"]["train"] == "0:2"
    assert not (dataset / "data/chunk-000/episode_000001.parquet").exists()


def test_repair_dataset_dry_run_does_not_rewrite_metadata(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    before = (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8")

    result = repair_dataset(dataset, regenerate_stats=False, dry_run=True)

    assert result.changed
    assert result.dropped_episodes == 1
    assert (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8") == before


def test_repair_dataset_noops_when_metadata_matches_files(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    _touch(dataset / "data/chunk-000/episode_000001.parquet")
    _touch(dataset / "videos/chunk-000/observation.images.front/episode_000001.mp4")
    _touch(dataset / "videos/chunk-000/observation.images.wrist/episode_000001.mp4")

    result = repair_dataset(dataset, regenerate_stats=False)

    assert not result.changed
    assert result.dropped_episodes == 0
    assert len(_read_jsonl(dataset / "meta/episodes.jsonl")) == 3
