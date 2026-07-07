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

"""Tests for the non-destructive output handling of ``convert_v3_to_v2.py``."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


pytest.importorskip("pyarrow")


def _install_lerobot_stubs() -> list[str]:
    """Install just enough fake ``lerobot`` modules to import the conversion script.

    The conversion script lives in its own subproject with a real ``lerobot``
    dependency; the destination-resolution logic under test does not need it.
    """

    stub_attrs: dict[str, dict[str, object]] = {
        "lerobot": {},
        "lerobot.datasets": {},
        "lerobot.datasets.utils": {
            "DEFAULT_CHUNK_SIZE": 1000,
            "DEFAULT_DATA_PATH": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "DEFAULT_VIDEO_PATH": (
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            ),
            "EPISODES_DIR": "meta/episodes",
            "LEGACY_EPISODES_PATH": "meta/episodes.jsonl",
            "LEGACY_EPISODES_STATS_PATH": "meta/episodes_stats.jsonl",
            "LEGACY_TASKS_PATH": "meta/tasks.jsonl",
            "load_info": lambda *args, **kwargs: None,
            "load_tasks": lambda *args, **kwargs: None,
            "serialize_dict": lambda *args, **kwargs: None,
            "unflatten_dict": lambda *args, **kwargs: None,
            "write_info": lambda *args, **kwargs: None,
        },
        "lerobot.utils": {},
        "lerobot.utils.constants": {"HF_LEROBOT_HOME": Path("unused")},
        "lerobot.utils.utils": {"init_logging": lambda *args, **kwargs: None},
    }

    installed: list[str] = []
    for name, attrs in stub_attrs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        installed.append(name)
    return installed


_stub_names = _install_lerobot_stubs()
try:
    from scripts.lerobot_conversion.convert_v3_to_v2 import resolve_output_roots
finally:
    for _name in _stub_names:
        sys.modules.pop(_name, None)


def test_default_output_is_sibling_with_v21_suffix(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    new_root, backup_root = resolve_output_roots(root, None, in_place=False)

    assert new_root == tmp_path / "dataset_v2.1"
    assert backup_root is None


def test_refuses_output_equal_to_source(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(ValueError, match="overwrite the source"):
        resolve_output_roots(root, root, in_place=False)


def test_refuses_output_inside_source(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(ValueError, match="overwrite the source"):
        resolve_output_roots(root, root / "v2", in_place=False)


def test_refuses_existing_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (tmp_path / "dataset_v2.1").mkdir()

    with pytest.raises(FileExistsError, match="Output directory already exists"):
        resolve_output_roots(root, None, in_place=False)


def test_in_place_returns_backup_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()

    new_root, backup_root = resolve_output_roots(root, None, in_place=True)

    assert new_root == tmp_path / "dataset_v2.1"
    assert backup_root == tmp_path / "dataset_backup_v3.0"


def test_in_place_refuses_existing_backup(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (tmp_path / "dataset_backup_v3.0").mkdir()

    with pytest.raises(FileExistsError, match="Backup directory already exists"):
        resolve_output_roots(root, None, in_place=True)


def test_output_and_in_place_are_mutually_exclusive(tmp_path: Path) -> None:
    root = tmp_path / "dataset"

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_output_roots(root, tmp_path / "out", in_place=True)
