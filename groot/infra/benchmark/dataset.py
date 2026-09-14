# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolve recipe-owned data without modifying the source dataset."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils.sha import git_hash

from groot.core.policy.benchmark_contract import artifact_identity
from groot.core.utils.storage.msc import hf_token
from groot.infra.eval.hf_artifact_cache import HfArtifactCache
from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig


def dataset_files(root: Path) -> list[Path]:
    """Enumerate all input files, excluding download/VCS bookkeeping."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    paths = []
    visited = set()
    for directory, subdirs, files in os.walk(root, followlinks=True):
        parent = Path(directory)
        if parent.resolve() in visited:
            raise ValueError(f"Dataset contains repeated or cyclic directories: {parent}")
        visited.add(parent.resolve())
        if parent == root:
            subdirs[:] = [name for name in subdirs if name not in {".git", ".cache"}]
        paths.extend(parent / name for name in files if parent != root or name != ".git")
    return sorted(paths)


def _manifest(root: Path, files: list[Path]) -> dict[str, Any]:
    return {
        path.relative_to(root).as_posix(): {
            key: value for key, value in artifact_identity(path).items() if key != "path"
        }
        for path in files
    }


def _fingerprint(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dataset_fingerprint(root: Path) -> str:
    """Hash relative filenames, sizes, and file contents for a local recipe."""
    root = root.expanduser().resolve()
    return _fingerprint(_manifest(root, dataset_files(root)))


def resolve_dataset(
    dataset: Mapping[str, Any], *, recipe_path: Path, destination: Path
) -> dict[str, Any]:
    """Verify recipe inputs and prepare a writable view with independent metadata."""
    if "path" in dataset:
        root = Path(dataset["path"]).expanduser()
        if not root.is_absolute():
            root = recipe_path.parent / root
        root = root.resolve()
        manifest = _manifest(root, dataset_files(root))
        if _fingerprint(manifest) != dataset["sha256"]:
            raise ValueError("Local dataset fingerprint does not match the recipe")
    else:
        cache = HfArtifactCache.init_from_config(OssOpenLoopEvalConfig())
        root = cache.download(
            repo_id=dataset["repo_id"], repo_type="dataset", revision=dataset["revision"]
        )
        entries = [
            entry
            for entry in HfApi(token=hf_token() or None).list_repo_tree(
                dataset["repo_id"],
                repo_type="dataset",
                revision=dataset["revision"],
                recursive=True,
            )
            if isinstance(entry, RepoFile)
        ]
        for entry in entries:
            if Path(entry.path).is_absolute() or ".." in Path(entry.path).parts:
                raise ValueError("Dataset file escapes its snapshot")
        files = [root / entry.path for entry in entries]
        manifest = _manifest(root, files)
        for entry in entries:
            actual = (
                manifest[entry.path]["sha256"]
                if entry.lfs
                else git_hash((root / entry.path).read_bytes())
            )
            expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
            if actual != expected:
                raise ValueError(f"Dataset file does not match the pinned revision: {entry.path}")
    if "meta/info.json" not in manifest or not any(name.startswith("data/") for name in manifest):
        raise ValueError("Dataset must contain LeRobot metadata and data files")
    if destination.resolve().is_relative_to(root):
        raise ValueError("Benchmark output must be outside the source dataset")
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("meta/"):
            shutil.copyfile(root / name, target)
        else:
            target.symlink_to((root / name).resolve())
    return {**dataset, "sha256": _fingerprint(manifest), "verified": True, "files": len(manifest)}
