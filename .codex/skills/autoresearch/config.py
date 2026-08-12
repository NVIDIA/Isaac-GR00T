# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configuration loading for AutoResearch studies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

from skills.autoresearch.study_state import Objective

# Schema guards keep study IDs and commits deterministic.
_STUDY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
TRIAL_EVAL_RESULT = "eval/results.json"


# StudyConfig is the normalized contract used by the controller.
@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Infrastructure-neutral configuration for one AutoResearch study."""

    study_id: str
    workspace: Path
    base_commit: str
    train_dataset: str
    eval_dataset: str
    objective: Objective
    max_trials: int
    editable_paths: tuple[str, ...]
    frozen_paths: tuple[str, ...]
    idea_types: tuple[str, ...]
    agent: dict[str, Any]
    eval_result: Path
    output_dir: Path
    train_script: Path
    eval_script: Path
    base_branch: str | None = None
    max_parallel_trials: int = 1
    max_retry: int = 0
    checkpoint_path: str | None = None
    output_dir_name: str = "training"
    script_timeout_s: float = 24 * 60 * 60.0
    git_remote: str = "origin"
    env: dict[str, str] = field(default_factory=dict)
    train_env: dict[str, str] = field(default_factory=dict)
    eval_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataclass invariants before this config reaches the controller."""
        # User scripts own their train/eval data inputs, so datasets are optional.
        for name in ("train_dataset", "eval_dataset"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a non-empty string")
        # Study IDs become state paths and branch fragments, so keep them simple.
        if not isinstance(self.study_id, str) or not _STUDY_ID_RE.fullmatch(self.study_id):
            raise ValueError(
                "study_id must start with a lowercase letter or number and contain only "
                "lowercase letters, numbers, '.', '_' or '-'"
            )
        if len(self.study_id) > 64:
            raise ValueError("study_id must be at most 64 characters")
        if ".." in self.study_id:
            raise ValueError("study_id must not contain consecutive dots")
        # Pin the study to one exact source revision.
        if not isinstance(self.base_commit, str) or not _FULL_COMMIT_RE.fullmatch(self.base_commit):
            raise ValueError("base_commit must be a full 40-character hexadecimal commit SHA")
        object.__setattr__(self, "base_commit", self.base_commit.lower())
        if self.base_branch is not None and (
            not isinstance(self.base_branch, str) or not self.base_branch.strip()
        ):
            raise TypeError("base_branch must be a non-empty string")
        # Trial scheduling limits are validated before any controller uses them.
        if not isinstance(self.workspace, Path):
            raise TypeError("workspace must be a pathlib.Path")
        if isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int):
            raise TypeError("max_trials must be an integer")
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        if isinstance(self.max_parallel_trials, bool) or not isinstance(
            self.max_parallel_trials, int
        ):
            raise TypeError("max_parallel_trials must be an integer")
        if self.max_parallel_trials < 1:
            raise ValueError("max_parallel_trials must be >= 1")
        if isinstance(self.max_retry, bool) or not isinstance(self.max_retry, int):
            raise TypeError("max_retry must be an integer")
        if self.max_retry < 0:
            raise ValueError("max_retry must be >= 0")
        # Execution paths and values are parsed once and shared by every component.
        for name in ("eval_result", "output_dir", "train_script", "eval_script"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        for name in ("output_dir_name", "git_remote"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if self.checkpoint_path is not None and (
            not isinstance(self.checkpoint_path, str) or not self.checkpoint_path
        ):
            raise TypeError("checkpoint_path must be a non-empty string or null")
        if (
            isinstance(self.script_timeout_s, bool)
            or not isinstance(self.script_timeout_s, int | float)
            or self.script_timeout_s < 0
        ):
            raise TypeError("script_timeout_s must be a number >= 0")
        object.__setattr__(self, "script_timeout_s", float(self.script_timeout_s))
        _string_mapping(self.env, "env")
        _string_mapping(self.train_env, "train environment")
        _string_mapping(self.eval_env, "eval environment")
        # Tuple fields and opaque agent settings must be stable and JSON-safe.
        if not self.idea_types:
            raise ValueError("idea_types must not be empty")
        _validate_string_tuple(self.editable_paths, "editable_paths")
        _validate_string_tuple(self.frozen_paths, "frozen_paths")
        _validate_string_tuple(self.idea_types, "idea_types")
        _copy_json_value(self.agent, "agent")

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        """Load one study from a YAML mapping."""
        # Load raw YAML first so all validation errors can cite this file.
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError(f"{config_path}: top-level YAML must be a mapping")

        # Resolve paths relative to the config file exactly once at load time.
        workspace_value = _required_string(raw, "workspace", config_path)
        workspace = Path(workspace_value).expanduser()
        if not workspace.is_absolute():
            workspace = config_path.parent / workspace

        workspace = workspace.resolve()
        # Baseline and Git settings are grouped in YAML, then normalized here.
        baseline = _required_mapping(raw, "baseline", config_path)
        git = _required_mapping(raw, "git", config_path)
        git_remote = _required_string(git, "git_remote", config_path)
        base_branch = _required_string(git, "base_branch", config_path)
        base_commit_value = git.get("base_commit")
        base_commit = (
            _resolve_branch_head(workspace, git_remote, base_branch, config_path)
            if base_commit_value is None
            else _nonempty_string(base_commit_value, "git.base_commit")
        )
        train = _config_section(raw, "train")
        evaluation = _config_section(raw, "eval")
        train_script = _required_absolute_path(baseline, "train_script", config_path)
        eval_script = _required_absolute_path(baseline, "eval_script", config_path)
        eval_result = _required_absolute_path(baseline, "eval_result", config_path)
        if "runs_root" in raw:
            raise ValueError(f"{config_path}: runs_root was renamed to output_dir")
        output_dir = _resolved_path(
            _required_string(raw, "output_dir", config_path),
            config_path.parent,
        )
        for retired in ("branch", "working_dir"):
            if retired in raw:
                raise ValueError(f"{config_path}: {retired} is no longer supported")
        checkpoint_path = train.get("checkpoint") or raw.get("checkpoint_path")
        output_dir_name = train.get("output_dir") or "training"
        agent = _optional_dict(raw, "agent", config_path)
        return cls(
            study_id=_required_string(raw, "study_id", config_path),
            workspace=workspace,
            base_commit=base_commit,
            base_branch=base_branch,
            train_dataset=_resolved_optional_path(raw, "train_dataset", config_path),
            eval_dataset=_resolved_optional_path(raw, "eval_dataset", config_path),
            objective=Objective.from_baseline(baseline),
            max_trials=_required_int(raw, "max_trials", config_path),
            editable_paths=_required_string_tuple(raw, "editable_paths", config_path),
            frozen_paths=_required_string_tuple(raw, "frozen_paths", config_path),
            idea_types=_required_string_tuple(raw, "idea_types", config_path),
            agent=agent,
            eval_result=eval_result,
            output_dir=output_dir,
            train_script=train_script,
            eval_script=eval_script,
            checkpoint_path=_optional_nonempty_string(checkpoint_path, "train.checkpoint"),
            output_dir_name=_nonempty_string(output_dir_name, "train.output_dir"),
            script_timeout_s=_nonnegative_float(baseline, "script_timeout_s"),
            git_remote=git_remote,
            env=_string_mapping(raw.get("env", {}), "env"),
            train_env=_string_mapping(train.get("env", {}), "train environment"),
            eval_env=_string_mapping(evaluation.get("env", {}), "eval environment"),
            max_parallel_trials=_optional_int(raw, "max_parallel_trials", config_path, default=1),
            max_retry=_optional_int(raw, "max_retry", config_path, default=0),
        )


def _resolve_branch_head(
    workspace: Path,
    remote: str,
    branch: str,
    config_path: Path,
) -> str:
    """Resolve an omitted base commit from the locally visible branch head."""
    if not workspace.is_dir():
        raise ValueError(
            f"{config_path}: cannot resolve git.base_branch because workspace does not "
            f"exist: {workspace}"
        )
    refs = (
        f"refs/heads/{branch}",
        f"refs/remotes/{remote}/{branch}",
        branch,
    )
    for ref in refs:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    raise ValueError(
        f"{config_path}: cannot resolve git.base_branch {branch!r} in workspace "
        f"{workspace}; fetch or create the branch, or provide git.base_commit"
    )


def _required_string(data: Mapping[str, Any], key: str, path: Path) -> str:
    """YAML scalar readers keep validation errors tied to the config file path."""
    if key not in data:
        raise KeyError(f"{path}: {key} is required")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path}: {key} must be a non-empty string")
    return value


def _required_absolute_path(
    data: Mapping[str, Any],
    key: str,
    config_path: Path,
) -> Path:
    """Read a required baseline path without silently resolving relative input."""
    value = Path(_required_string(data, key, config_path)).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{config_path}: baseline.{key} must be an absolute path")
    return value.resolve()


def _required_int(data: Mapping[str, Any], key: str, path: Path) -> int:
    """Validate one required config field with a focused error message."""
    if key not in data:
        raise KeyError(f"{path}: {key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: {key} must be an integer")
    return value


def _optional_int(data: Mapping[str, Any], key: str, path: Path, *, default: int) -> int:
    """Read one optional config field while preserving strict type checks."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: {key} must be an integer")
    return value


def _resolved_optional_path(data: Mapping[str, Any], key: str, config_path: Path) -> str:
    """Resolve configured paths once while loading the study."""
    value = data.get(key, "")
    if value == "":
        return ""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{config_path}: {key} must be a string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def _required_mapping(data: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    """Validate one required config field with a focused error message."""
    if key not in data:
        raise KeyError(f"{path}: {key} is required")
    value = data[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: {key} must be a mapping")
    return value


def _optional_dict(data: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    """Read one optional config field while preserving strict type checks."""
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: {key} must be a mapping")
    return _copy_json_mapping(value, f"{path}: {key}")


def _required_string_tuple(data: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """Validate one required config field with a focused error message."""
    if key not in data:
        raise KeyError(f"{path}: {key} is required")
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(f"{path}: {key} must be a list")
    result = tuple(value)
    _validate_string_tuple(result, f"{path}: {key}")
    return result


def _resolved_path(value: str, base_dir: Path) -> Path:
    """Resolve one configured path against the YAML directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _config_section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read one optional top-level config section with strict type checks."""
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    """Validate one execution-config string."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _optional_nonempty_string(value: Any, label: str) -> str | None:
    """Validate one optional execution-config string."""
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _nonnegative_float(data: Mapping[str, Any], key: str) -> float:
    """Read one nonnegative execution-config number."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise TypeError(f"{key} must be a number >= 0")
    return float(value)


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    """Read an execution environment mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise TypeError(f"{label} must map strings to strings")
    return dict(value)


def _validate_string_tuple(value: tuple[str, ...], label: str) -> None:
    """Validate a tuple used by the study configuration."""
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise TypeError(f"{label} entries must be non-empty strings")


def _copy_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Copy a mapping only after JSON-compatible canonicalization."""
    copied = _copy_json_value(value, label)
    if not isinstance(copied, dict):
        raise TypeError(f"{label} must be a mapping")
    return copied


def _copy_json_value(value: Any, label: str) -> Any:
    """Copy an opaque config value while restricting it to JSON-compatible types."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} mapping keys must be strings")
            result[key] = _copy_json_value(item, f"{label}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_copy_json_value(item, label) for item in value]
    raise TypeError(f"{label} contains unsupported value of type {type(value).__name__}")
