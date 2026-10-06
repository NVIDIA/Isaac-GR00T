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

"""Unit tests for ``fast_copy_tree``.

The helper is exercised against ``tmp_path`` so the tests are hermetic
and don't depend on a real ``/shared`` NFS mount. Each test pins one
behavioural property: bulk parity with ``shutil.copytree``, symlink
dereference vs preserve, dirs-exist merging, ``tar``-missing fallback,
and the not-a-directory contract.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest
from test_support import runtime


def _materialise_tree(root: pathlib.Path) -> dict[str, bytes]:
    """Create a small mixed tree under *root*; return path → content map."""
    files: dict[str, bytes] = {
        "top.txt": b"hello\n",
        "pkg/__init__.py": b"",
        "pkg/sub/data.bin": b"\x00\x01\x02" * 100,
        "pkg/sub/empty": b"",
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def _read_tree(root: pathlib.Path) -> dict[str, bytes]:
    """Return path → content map of all regular files under *root*."""
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def test_copies_files_and_subdirs(tmp_path):
    """Round-trip: bytes and tree shape match the source after copy."""
    src = tmp_path / "src"
    expected = _materialise_tree(src)
    dst = tmp_path / "dst"

    runtime.fast_copy_tree(src, dst)

    assert _read_tree(dst) == expected


def test_dereferences_symlinks_by_default(tmp_path):
    """``symlinks=False`` (default) must copy real bytes, not symlinks.

    Load-bearing for staged ``/shared`` trees: leaving a symlink in the
    staged copy would route subsequent reads back over NFS and defeat
    the purpose of staging.
    """
    src = tmp_path / "src"
    src.mkdir()
    target = src / "real.bin"
    target.write_bytes(b"payload")
    (src / "link.bin").symlink_to(target)
    dst = tmp_path / "dst"

    runtime.fast_copy_tree(src, dst)

    copied_link = dst / "link.bin"
    assert copied_link.exists()
    assert not copied_link.is_symlink()
    assert copied_link.read_bytes() == b"payload"


def test_preserves_symlinks_when_requested(tmp_path):
    """``symlinks=True`` must keep symlinks verbatim (libero venv pattern)."""
    src = tmp_path / "src"
    src.mkdir()
    target = src / "real.bin"
    target.write_bytes(b"payload")
    (src / "link.bin").symlink_to(target)
    dst = tmp_path / "dst"

    runtime.fast_copy_tree(src, dst, symlinks=True)

    copied_link = dst / "link.bin"
    assert copied_link.is_symlink()


def test_merges_into_existing_destination(tmp_path):
    """Existing dst entries are preserved; same-name entries overwritten.

    Matches ``shutil.copytree(dirs_exist_ok=True)`` semantics so the new
    helper is a drop-in replacement for the prior call sites.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "shared.txt").write_bytes(b"new")
    (src / "src_only.txt").write_bytes(b"src")

    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "shared.txt").write_bytes(b"old")
    (dst / "dst_only.txt").write_bytes(b"dst")

    runtime.fast_copy_tree(src, dst)

    assert (dst / "shared.txt").read_bytes() == b"new"
    assert (dst / "src_only.txt").read_bytes() == b"src"
    assert (dst / "dst_only.txt").read_bytes() == b"dst"


def test_falls_back_to_copytree_when_tar_missing(monkeypatch, tmp_path):
    """When ``tar`` is unavailable we fall back to ``shutil.copytree``.

    Asserted via a monkeypatched ``shutil.which`` so the test is
    hermetic regardless of host PATH.
    """
    src = tmp_path / "src"
    expected = _materialise_tree(src)
    dst = tmp_path / "dst"

    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    called: list[tuple[str, str]] = []
    real_copytree = shutil.copytree

    def _spy_copytree(s, d, *args, **kwargs):
        # Only the top-level call records — copytree() recurses into
        # itself for subdirs, so filter to the call we initiated.
        if str(s) == str(src):
            called.append((str(s), str(d)))
        return real_copytree(s, d, *args, **kwargs)

    monkeypatch.setattr(runtime.shutil, "copytree", _spy_copytree)

    runtime.fast_copy_tree(src, dst)

    assert called == [(str(src), str(dst))]
    assert _read_tree(dst) == expected


def test_rejects_non_directory_source(tmp_path):
    """Helper rejects file/non-existent sources up-front rather than mid-pipe.

    Prevents the more confusing failure mode where ``tar`` would error
    deep inside the pipe and we'd surface a generic non-zero rc.
    """
    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        runtime.fast_copy_tree(not_a_dir, tmp_path / "dst")


def test_reaps_src_proc_when_dst_popen_fails(monkeypatch, tmp_path):
    """If the second Popen raises, src_proc must not be left as a zombie.

    Pre-fix, the only cleanup was closing the pipe fd; src_proc kept
    running with no reader and was never `wait()`-ed. Under repeated
    failures (CI memory pressure) this accumulated zombies.
    """
    src = tmp_path / "src"
    _materialise_tree(src)
    dst = tmp_path / "dst"

    real_popen = runtime.subprocess.Popen
    started: list[runtime.subprocess.Popen] = []
    call_count = {"n": 0}

    def _failing_popen(cmd, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            proc = real_popen(cmd, *args, **kwargs)
            started.append(proc)
            return proc
        raise OSError("simulated dst Popen failure")

    monkeypatch.setattr(runtime.subprocess, "Popen", _failing_popen)

    with pytest.raises(OSError, match="simulated dst Popen failure"):
        runtime.fast_copy_tree(src, dst)

    assert len(started) == 1
    src_proc = started[0]
    assert src_proc.returncode is not None, "src_proc was not reaped after dst Popen failure"
