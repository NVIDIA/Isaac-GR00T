# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pinned Git checkout enforcement for AutoDebug isolation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class GitCheckout:
    path: Path
    root: Path
    prefix: str
    common_dir: Path
    head: str
    tracked_status: str
    worktree_status: str
    tracked_tree_sha256: str

    @classmethod
    def init_from_path(cls, path: Path) -> GitCheckout:
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Git checkout is not a directory: {path}")

        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        common_result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        prefix_result = subprocess.run(
            ["git", "rev-parse", "--show-prefix"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        worktree_status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        diff_result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            cwd=path,
            capture_output=True,
            check=False,
        )
        results = (
            head_result,
            common_result,
            root_result,
            prefix_result,
            status_result,
            worktree_status_result,
            diff_result,
        )
        if any(result.returncode != 0 for result in results):
            failed = next(result for result in results if result.returncode != 0)
            stderr = failed.stderr
            detail = (
                stderr.decode(errors="replace").strip()
                if isinstance(stderr, bytes)
                else stderr.strip()
            )
            raise ValueError(f"Cannot inspect Git checkout {path}: {detail or 'unknown error'}")

        common_dir = Path(common_result.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = path / common_dir
        return cls(
            path=path,
            root=Path(root_result.stdout.strip()).resolve(),
            prefix=prefix_result.stdout.strip().rstrip("/"),
            common_dir=common_dir.resolve(),
            head=head_result.stdout.strip(),
            tracked_status=status_result.stdout.strip(),
            worktree_status=worktree_status_result.stdout.strip(),
            tracked_tree_sha256=hashlib.sha256(diff_result.stdout).hexdigest(),
        )

    def get_text(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
            raise ValueError(f"Git command failed in {self.path}: {detail}")
        return result.stdout.strip()

    def get_bytes(self, *arguments: str) -> bytes:
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.path,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip() or "unknown Git error"
            raise ValueError(f"Git command failed in {self.path}: {detail}")
        return result.stdout

    def contains_revision(self, revision: str) -> bool:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, self.head],
            cwd=self.path,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
