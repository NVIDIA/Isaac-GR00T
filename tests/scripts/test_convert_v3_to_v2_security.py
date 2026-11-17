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

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def converter(monkeypatch):
    # LeRobot is installed in a separate conversion environment. Stub only its
    # imports; the path construction, validation and ffmpeg invocation stay real.
    utils = ModuleType("lerobot.datasets.utils")
    constants = {
        "DEFAULT_CHUNK_SIZE": 1000,
        "DEFAULT_DATA_PATH": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "DEFAULT_VIDEO_PATH": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "EPISODES_DIR": "meta/episodes",
        "LEGACY_EPISODES_PATH": "meta/episodes.jsonl",
        "LEGACY_EPISODES_STATS_PATH": "meta/episodes_stats.jsonl",
        "LEGACY_TASKS_PATH": "meta/tasks.jsonl",
    }
    for name, value in constants.items():
        setattr(utils, name, value)
    for name in ("load_info", "load_tasks", "serialize_dict", "unflatten_dict", "write_info"):
        setattr(utils, name, Mock(side_effect=AssertionError(f"Unexpected call: {name}")))
    constants_module = ModuleType("lerobot.utils.constants")
    constants_module.HF_LEROBOT_HOME = Path("unused-hf-home")
    logging_utils = ModuleType("lerobot.utils.utils")
    logging_utils.init_logging = Mock()
    for name, module in {
        "lerobot": ModuleType("lerobot"),
        "lerobot.datasets": ModuleType("lerobot.datasets"),
        "lerobot.datasets.utils": utils,
        "lerobot.utils": ModuleType("lerobot.utils"),
        "lerobot.utils.constants": constants_module,
        "lerobot.utils.utils": logging_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "scripts/lerobot_conversion/convert_v3_to_v2.py"
    spec = importlib.util.spec_from_file_location("converter_security_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(key):
    return {
        "episode_index": 0,
        f"videos/{key}/chunk_index": 0,
        f"videos/{key}/file_index": 0,
        f"videos/{key}/from_timestamp": 0.0,
        f"videos/{key}/to_timestamp": 0.2,
    }


def _source(root, key="camera"):
    path = root / f"videos/{key}/chunk-000/file-000.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"source video")
    return path


@pytest.mark.parametrize(
    "key",
    [
        "../../../controlled/repository/videos/camera",
        "../camera",
        "/camera",
        "camera/../../outside",
        r"..\camera",
        "C:camera",
        "",
        ".",
        "..",
        "cam\x00era",
        "cam\nera",
    ],
)
def test_rejects_video_keys_before_writes(converter, tmp_path, monkeypatch, key):
    root = tmp_path / "controlled/repository"
    new_root = root.with_name("repository_v2.1")
    _source(root)
    canary = tmp_path / "controlled/controlled/repository/videos/camera/episode_000000.mp4"
    canary.parent.mkdir(parents=True)
    canary.write_bytes(b"canary")
    run = Mock(side_effect=AssertionError("ffmpeg must not run"))
    monkeypatch.setattr(converter.subprocess, "run", run)
    with pytest.raises(ValueError, match="Invalid video feature key"):
        converter.convert_videos(root, new_root, [_record(key)], [key], 1000)
    assert canary.read_bytes() == b"canary"
    assert not new_root.exists()
    run.assert_not_called()


def test_dataset_rejects_key_before_removing_conversion_dirs(converter, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    output = tmp_path / "repository_v2.1"
    output.mkdir()
    canary = output / "keep"
    canary.write_bytes(b"existing conversion")
    monkeypatch.setattr(converter, "validate_local_dataset_version", lambda root: None)
    monkeypatch.setattr(converter, "load_episode_records", lambda root: [])
    monkeypatch.setattr(
        converter, "load_info", lambda root: {"features": {"../camera": {"dtype": "video"}}}
    )
    with pytest.raises(ValueError, match="Invalid video feature key"):
        converter.convert_dataset("repository", root=tmp_path)
    assert canary.read_bytes() == b"existing conversion"


@pytest.mark.parametrize("escape", ["source", "destination", "destination_file", "sibling_prefix"])
def test_rejects_resolved_path_escapes(converter, tmp_path, monkeypatch, escape):
    root = tmp_path / "dataset"
    new_root = tmp_path / "converted"
    src = _source(root)
    dst = new_root / "videos/chunk-000/camera/episode_000000.mp4"
    outside = tmp_path / "converted-sibling"
    outside.mkdir()
    canary = outside / "episode_000000.mp4"
    canary.write_bytes(b"canary")
    if escape == "source":
        src.unlink()
        src.symlink_to(canary)
    elif escape == "destination":
        new_root.mkdir()
        (new_root / "videos").symlink_to(outside, target_is_directory=True)
    elif escape == "destination_file":
        dst.parent.mkdir(parents=True)
        dst.symlink_to(canary)
    else:
        dst = canary
    run = Mock(side_effect=AssertionError("ffmpeg must not run"))
    monkeypatch.setattr(converter.subprocess, "run", run)
    with pytest.raises(ValueError, match="escapes its root"):
        converter._extract_video_segment(src, dst, 0.0, 0.2, root=root, new_root=new_root)
    assert canary.read_bytes() == b"canary"
    run.assert_not_called()


def test_normal_video_conversion(converter, tmp_path, monkeypatch):
    root, new_root = tmp_path / "dataset", tmp_path / "converted"
    src = _source(root, "observation.images.front")
    run = Mock()
    monkeypatch.setattr(converter.subprocess, "run", run)
    key = "observation.images.front"
    converter.convert_videos(root, new_root, [_record(key)], [key], 1000)
    command = run.call_args.args[0]
    assert command[command.index("-i") + 1] == str(src.resolve())
    assert command[-1] == str(new_root / f"videos/chunk-000/{key}/episode_000000.mp4")
    assert Path(command[-1]).parent.is_dir()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("malicious", [False, True])
def test_real_ffmpeg_conversion(converter, tmp_path, malicious):
    root = tmp_path / "controlled/repository"
    new_root = root.with_name("repository_v2.1")
    src = _source(root)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=10",
            "-t",
            "0.5",
            "-c:v",
            "mpeg4",
            "-y",
            str(src),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    if malicious:
        key = "../../../controlled/repository/videos/camera"
        canary = tmp_path / "controlled/controlled/repository/videos/camera/episode_000000.mp4"
        canary.parent.mkdir(parents=True)
        canary.write_bytes(b"existing canary")
        with pytest.raises(ValueError, match="Invalid video feature key"):
            converter.convert_videos(root, new_root, [_record(key)], [key], 1000)
        assert canary.read_bytes() == b"existing canary"
        assert not new_root.exists()
        return
    converter.convert_videos(root, new_root, [_record("camera")], ["camera"], 1000)
    dst = new_root / "videos/chunk-000/camera/episode_000000.mp4"
    assert dst.stat().st_size > 0
