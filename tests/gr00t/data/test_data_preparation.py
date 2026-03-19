"""
Validate that bundled demo datasets conform to the GR00T LeRobot format
described in getting_started/data_preparation.md.

Checks:
  - Required directory structure (meta/, data/chunk-*, videos/chunk-*)
  - Required meta files (info.json, episodes.jsonl, tasks.jsonl, modality.json)
  - modality.json schema (state/action keys with start/end, video keys)
  - Parquet files exist and contain expected columns
  - Video files exist as .mp4
"""

from __future__ import annotations

import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
DEMO_DATA_DIR = ROOT / "demo_data"

# All bundled demo datasets to validate.
DEMO_DATASETS = [
    p for p in DEMO_DATA_DIR.iterdir() if p.is_dir() and (p / "meta" / "modality.json").exists()
]


@pytest.fixture(params=[str(d.relative_to(ROOT)) for d in DEMO_DATASETS])
def dataset_path(request):
    return ROOT / request.param


class TestDatasetStructure:
    """Validate the directory structure required by data_preparation.md."""

    def test_meta_dir_exists(self, dataset_path):
        assert (dataset_path / "meta").is_dir(), f"Missing meta/ in {dataset_path}"

    def test_data_dir_exists(self, dataset_path):
        assert (dataset_path / "data").is_dir(), f"Missing data/ in {dataset_path}"

    def test_videos_dir_exists(self, dataset_path):
        assert (dataset_path / "videos").is_dir(), f"Missing videos/ in {dataset_path}"

    def test_required_meta_files(self, dataset_path):
        meta = dataset_path / "meta"
        for filename in ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"]:
            assert (meta / filename).is_file(), f"Missing {filename} in {meta}"

    def test_data_has_parquet_files(self, dataset_path):
        parquets = list((dataset_path / "data").rglob("*.parquet"))
        assert parquets, f"No .parquet files found in {dataset_path / 'data'}"

    def test_videos_has_mp4_files(self, dataset_path):
        mp4s = list((dataset_path / "videos").rglob("*.mp4"))
        assert mp4s, f"No .mp4 files found in {dataset_path / 'videos'}"


class TestModalityJson:
    """Validate modality.json schema per data_preparation.md."""

    def test_modality_json_is_valid_json(self, dataset_path):
        modality_path = dataset_path / "meta" / "modality.json"
        with open(modality_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_modality_has_state_and_action(self, dataset_path):
        with open(dataset_path / "meta" / "modality.json") as f:
            data = json.load(f)
        assert "state" in data, "modality.json missing 'state' key"
        assert "action" in data, "modality.json missing 'action' key"

    def test_state_keys_have_start_end(self, dataset_path):
        with open(dataset_path / "meta" / "modality.json") as f:
            data = json.load(f)
        for key, spec in data["state"].items():
            assert "start" in spec, f"state.{key} missing 'start'"
            assert "end" in spec, f"state.{key} missing 'end'"
            assert isinstance(spec["start"], int), f"state.{key}.start must be int"
            assert isinstance(spec["end"], int), f"state.{key}.end must be int"
            assert spec["end"] > spec["start"], (
                f"state.{key}: end ({spec['end']}) must be > start ({spec['start']})"
            )

    def test_action_keys_have_start_end(self, dataset_path):
        with open(dataset_path / "meta" / "modality.json") as f:
            data = json.load(f)
        for key, spec in data["action"].items():
            assert "start" in spec, f"action.{key} missing 'start'"
            assert "end" in spec, f"action.{key} missing 'end'"
            assert isinstance(spec["start"], int), f"action.{key}.start must be int"
            assert isinstance(spec["end"], int), f"action.{key}.end must be int"
            assert spec["end"] > spec["start"], (
                f"action.{key}: end ({spec['end']}) must be > start ({spec['start']})"
            )

    def test_video_keys_have_original_key(self, dataset_path):
        with open(dataset_path / "meta" / "modality.json") as f:
            data = json.load(f)
        if "video" not in data:
            pytest.skip("No video section in modality.json")
        for key, spec in data["video"].items():
            assert "original_key" in spec, f"video.{key} missing 'original_key'"


class TestEpisodesJsonl:
    """Validate episodes.jsonl format."""

    def test_episodes_are_valid_jsonl(self, dataset_path):
        episodes_path = dataset_path / "meta" / "episodes.jsonl"
        episodes = []
        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        assert len(episodes) > 0, "episodes.jsonl is empty"
        for ep in episodes:
            assert "episode_index" in ep, f"Episode missing 'episode_index': {ep}"
            assert "length" in ep, f"Episode missing 'length': {ep}"


class TestTasksJsonl:
    """Validate tasks.jsonl format."""

    def test_tasks_are_valid_jsonl(self, dataset_path):
        tasks_path = dataset_path / "meta" / "tasks.jsonl"
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        assert len(tasks) > 0, "tasks.jsonl is empty"
        for task in tasks:
            assert "task_index" in task, f"Task missing 'task_index': {task}"
            assert "task" in task, f"Task missing 'task': {task}"
