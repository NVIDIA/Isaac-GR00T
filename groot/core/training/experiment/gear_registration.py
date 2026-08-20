# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standalone experiment-registration helpers for the OSS package.

The internal module at this package path integrates with GEAR Dashboard. Public
training keeps the same import surface and local path/resume behavior, while all
service operations are explicit no-ops.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

API_URL_ENV = "GEAR_DASHBOARD_API_URL"
PRESERVE_CHECKPOINT_ROOT_ENV = "GEAR_PRESERVE_CHECKPOINT_ROOT"
PRESERVE_OUTPUT_DIR_ENV = "GEAR_PRESERVE_OUTPUT_DIR"
TRAINING_OUTPUT_DIR_ENV = "TRAINING_OUTPUT_DIR"
DASHBOARD_LFS_CHECKPOINT_PATH_ENV = "GEAR_LFS_CHECKPOINT_PATH"
RESUME_CHECKPOINT_PATH_ENV = "GROOT_OL_RESUME_CHECKPOINT_PATH"
RESUME_DECIDED_ENV = "GROOT_OL_RESUME_DECIDED"
RESUME_REQUIRED_ENV = "GROOT_OL_RESUME_REQUIRED"


def preserve_checkpoint_root_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether launcher/worker should keep user-provided checkpoint roots."""
    env = os.environ if env is None else env
    return any(
        env.get(key, "").lower() in {"1", "true", "yes"}
        for key in (PRESERVE_CHECKPOINT_ROOT_ENV, PRESERVE_OUTPUT_DIR_ENV)
    )


def resolve_training_output_dir(
    configured_output_dir: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the output root used by local training and resume state."""
    env = os.environ if env is None else env
    configured_output_dir = env.get(TRAINING_OUTPUT_DIR_ENV, "").strip() or configured_output_dir
    if preserve_checkpoint_root_enabled(env):
        return configured_output_dir
    return env.get(DASHBOARD_LFS_CHECKPOINT_PATH_ENV, "").strip() or configured_output_dir


def one_launcher_resume_checkpoint() -> str | bool:
    """Return an exact checkpoint, or enable local latest-checkpoint discovery."""
    resume_decided = os.environ.get(RESUME_DECIDED_ENV, "").lower() in {"1", "true", "yes"}
    if not resume_decided:
        return True
    return os.environ.get(RESUME_CHECKPOINT_PATH_ENV, "").strip() or False


def dashboard_checkpoint_path_to_msc_url(checkpoint_path: str | None) -> str:
    """Convert a configured checkpoint root to the URL expected by MSC uploads."""
    if not checkpoint_path:
        return ""
    checkpoint_path = checkpoint_path.strip()
    if not checkpoint_path or "://" in checkpoint_path:
        return checkpoint_path
    return f"msc://{checkpoint_path}"


def _get_api_url() -> str:
    """Return an explicitly configured service URL, with no internal default."""
    return os.environ.get(API_URL_ENV, "")


def _first_env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _normalize_branch(branch: str) -> str:
    branch = branch.strip()
    for prefix in ("refs/heads/", "origin/"):
        if branch.startswith(prefix):
            branch = branch.removeprefix(prefix)
    return "" if branch == "HEAD" else branch


def _first_branch_env(*names: str) -> str:
    for name in names:
        branch = _normalize_branch(os.environ.get(name, ""))
        if branch:
            return branch
    return ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _get_groot_commit() -> str:
    """Resolve the source commit, preferring CI metadata over local git state."""
    env_commit = _first_env_value(
        "GROOT_COMMIT_HASH",
        "CI_GIT_COMMIT",
        "CI_COMMIT_SHA",
        "CI_COMMIT_SHORT_SHA",
    )
    if env_commit:
        return env_commit
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_branch() -> str:
    """Resolve the source branch, preferring CI metadata over local git state."""
    env_branch = _first_branch_env(
        "GROOT_BRANCH",
        "GROOT_COMMIT_BRANCH",
        "CI_GIT_BRANCH",
        "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME",
        "CI_COMMIT_BRANCH",
        "CI_COMMIT_REF_NAME",
        "GIT_BRANCH",
    )
    if env_branch:
        return env_branch

    for cmd in (["git", "branch", "--show-current"], ["git", "rev-parse", "--abbrev-ref", "HEAD"]):
        try:
            branch = (
                subprocess.check_output(cmd, cwd=_repo_root(), stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            branch = _normalize_branch(branch)
            if branch:
                return branch
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not resolve git branch via %s: %s", " ".join(cmd), exc)
    return ""


def _infer_gpu_type(cluster_type: str) -> str:
    mapping = {
        "colossus": "GB300",
        "gb200": "GB200",
        "gb300": "GB300",
        "h100": "H100",
        "l40": "L40",
        "l40s": "L40S",
    }
    return mapping.get(cluster_type.lower(), cluster_type.upper())


def register_experiment(*, experiment_name: str, **fields: Any) -> dict[str, str]:
    """Leave experiment registration disabled in the OSS package."""
    del experiment_name, fields
    return {}


def register_training_run(*, experiment_id: str, **fields: Any) -> str:
    """Leave training-run registration disabled in the OSS package."""
    del experiment_id, fields
    return ""


def update_experiment(experiment_id: str, **fields: Any) -> None:
    """Leave experiment updates disabled in the OSS package."""
    del experiment_id, fields


def update_training_run(experiment_id: str, training_run_id: str, **fields: Any) -> None:
    """Leave training-run updates disabled in the OSS package."""
    del experiment_id, training_run_id, fields


def append_training_run_status_event(
    experiment_ref_or_id: str,
    run_id: str,
    **fields: Any,
) -> None:
    """Leave training-run status reporting disabled in the OSS package."""
    del experiment_ref_or_id, run_id, fields
