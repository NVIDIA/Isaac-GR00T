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
"""Infrastructure-independent models for an AutoResearch study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
import re
from typing import Any, Literal, get_args

# Literal types define the durable vocabulary saved in state.json.
ObjectiveDirection = Literal["minimize", "maximize"]
StageStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "skipped",
]


# Objective owns score comparison so minimize/maximize logic stays centralized.
@dataclass(frozen=True, slots=True)
class Objective:
    """A scalar study objective and its comparison semantics."""

    result_path: str
    direction: ObjectiveDirection
    baseline: float
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        """Validate metric comparison invariants."""
        if not self.result_path.strip():
            raise ValueError("objective result_path must not be empty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("objective direction must be 'minimize' or 'maximize'")
        _require_finite(self.baseline, "objective baseline")
        _require_finite(self.min_delta, "objective min_delta")
        if self.min_delta < 0:
            raise ValueError("objective min_delta must be >= 0")

    def is_better(self, candidate: float, reference: float) -> bool:
        """Return whether candidate improves on reference by at least min_delta."""
        _require_finite(candidate, "candidate score")
        _require_finite(reference, "reference score")
        improvement = (
            reference - candidate if self.direction == "minimize" else candidate - reference
        )
        return improvement > 0 and improvement >= self.min_delta

    @classmethod
    def from_baseline(cls, data: Mapping[str, Any]) -> Objective:
        """Build metric comparison rules from the baseline config section."""
        mapping = _require_mapping(data, "baseline")
        if "baseline_score" in mapping:
            raise ValueError("baseline.baseline_score was renamed to baseline_result")
        return cls(
            result_path=_require_string(mapping, "eval_metric", "baseline"),
            direction=_require_string(mapping, "direction", "baseline"),  # type: ignore[arg-type]
            baseline=_require_number(mapping, "baseline_result", "baseline"),
            min_delta=_optional_number(mapping, "min_delta", "baseline", default=0.0),
        )


# StageState is the durable train/eval sub-state used by all runners.
@dataclass(frozen=True, slots=True)
class StageState:
    """Durable state for one train or evaluation stage."""

    status: StageStatus = "pending"
    attempts: int = 0
    error: str | None = None
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validate dataclass invariants before the value is persisted."""
        valid_statuses = get_args(StageStatus)
        if self.status not in valid_statuses:
            valid = ", ".join(sorted(valid_statuses))
            raise ValueError(f"unknown stage status {self.status!r}; expected one of: {valid}")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise TypeError("stage attempts must be an integer")
        if self.attempts < 0:
            raise ValueError("stage attempts must be >= 0")
        if self.checkpoint is not None and (
            not isinstance(self.checkpoint, str) or not self.checkpoint.strip()
        ):
            raise TypeError("stage checkpoint must be a non-empty string or null")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object into durable JSON-compatible data."""
        data = {
            "status": self.status,
            "attempts": self.attempts,
        }
        if self.error is not None:
            data["error"] = self.error
        if self.checkpoint is not None:
            data["checkpoint"] = self.checkpoint
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageState:
        """Build a typed object from durable or config data."""
        mapping = _require_mapping(data, "stage")
        return cls(
            status=_optional_string(mapping, "status", "stage", default="pending"),  # type: ignore[arg-type]
            attempts=_optional_int(mapping, "attempts", "stage", default=0),
            error=_optional_string(mapping, "error", "stage"),
            checkpoint=_optional_string(mapping, "checkpoint", "stage"),
        )


# Trial records one hypothesis from edit through training, evaluation, and scoring.
@dataclass(frozen=True, slots=True)
class Trial:
    """One isolated hypothesis and its train/evaluation result."""

    trial_id: str
    index: int
    parent_trial_id: str | None
    idea_type: str
    hypothesis: str
    wave_index: int = 0
    parent_revision: str = ""
    composition_source_trial_id: str | None = None
    result_summary: str = ""
    next_guidance: str = ""
    code_branch: str = ""
    code_revision: str = ""
    code_published: bool = False
    parent_score: float | None = None
    train: StageState = field(default_factory=StageState)
    evaluation: StageState = field(default_factory=StageState)
    score: float | None = None
    improved_over_parent: bool | None = None
    is_new_best: bool | None = None
    changed_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate dataclass invariants before the value is persisted."""
        # Trial identity and wave placement must be stable before persistence.
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("trial index must be an integer")
        if self.index < 1:
            raise ValueError("trial index must be >= 1")
        if isinstance(self.wave_index, bool) or not isinstance(self.wave_index, int):
            raise TypeError("trial wave_index must be an integer")
        if self.wave_index < 0:
            raise ValueError("trial wave_index must be >= 0")
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be empty")
        # Scores and comparison flags are validated together to avoid partial results.
        if self.parent_score is not None:
            _require_finite(self.parent_score, "trial parent_score")
        if self.score is not None:
            _require_finite(self.score, "trial score")
        elif self.improved_over_parent is not None or self.is_new_best is not None:
            raise ValueError("trial comparison flags require a score")
        if self.code_published and (not self.code_branch or not self.code_revision):
            raise ValueError("published trial code requires a branch and revision")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object into durable JSON-compatible data."""
        return {
            "trial_id": self.trial_id,
            "index": self.index,
            "parent_trial_id": self.parent_trial_id,
            "idea_type": self.idea_type,
            "hypothesis": self.hypothesis,
            "wave_index": self.wave_index,
            "parent_revision": self.parent_revision,
            "composition_source_trial_id": self.composition_source_trial_id,
            "result_summary": self.result_summary,
            "next_guidance": self.next_guidance,
            "code_branch": self.code_branch,
            "code_revision": self.code_revision,
            "code_published": self.code_published,
            "parent_score": self.parent_score,
            "train": self.train.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "score": self.score,
            "improved_over_parent": self.improved_over_parent,
            "is_new_best": self.is_new_best,
            "changed_files": list(self.changed_files),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trial:
        """Build a typed object from durable or config data."""
        mapping = _require_mapping(data, "trial")
        composition_source_trial_id = _optional_string(
            mapping, "composition_source_trial_id", "trial"
        )
        return cls(
            trial_id=_require_string(mapping, "trial_id", "trial"),
            index=_require_int(mapping, "index", "trial"),
            parent_trial_id=_optional_string(mapping, "parent_trial_id", "trial"),
            idea_type=_require_string(mapping, "idea_type", "trial"),
            hypothesis=_optional_string(mapping, "hypothesis", "trial", default=""),
            wave_index=_optional_int(mapping, "wave_index", "trial", default=0),
            parent_revision=_optional_string(mapping, "parent_revision", "trial", default="") or "",
            composition_source_trial_id=composition_source_trial_id,
            result_summary=_optional_string(mapping, "result_summary", "trial", default=""),
            next_guidance=_optional_string(mapping, "next_guidance", "trial", default=""),
            code_branch=_optional_string(mapping, "code_branch", "trial", default=""),
            code_revision=_optional_string(mapping, "code_revision", "trial", default=""),
            code_published=_optional_bool(mapping, "code_published", "trial") or False,
            parent_score=_optional_number(mapping, "parent_score", "trial"),
            train=StageState.from_dict(_require_mapping(mapping.get("train"), "trial.train")),
            evaluation=StageState.from_dict(
                _require_mapping(mapping.get("evaluation"), "trial.evaluation")
            ),
            score=_optional_number(mapping, "score", "trial"),
            improved_over_parent=_optional_bool(mapping, "improved_over_parent", "trial"),
            is_new_best=_optional_bool(mapping, "is_new_best", "trial"),
            changed_files=_string_tuple(mapping.get("changed_files", ()), "trial.changed_files"),
        )


# StudyState is the complete serializable controller snapshot.
@dataclass(frozen=True, slots=True)
class StudyState:
    """Persisted state for one study, independent of its execution backend."""

    study_id: str
    baseline_score: float
    best_score: float
    base_commit: str
    eval_metric: str
    direction: ObjectiveDirection
    min_delta: float
    best_trial_id: str | None = None
    trials: tuple[Trial, ...] = ()
    active_wave_index: int | None = None
    pending_change_trial_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate dataclass invariants before the value is persisted."""
        # Validate top-level identity and scores before cross-trial checks.
        if not self.study_id.strip():
            raise ValueError("study_id must not be empty")
        _require_finite(self.baseline_score, "study baseline_score")
        _require_finite(self.best_score, "study best_score")
        if not isinstance(self.base_commit, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.base_commit
        ):
            raise ValueError("study base_commit must be a full 40-character hexadecimal commit SHA")
        if not isinstance(self.eval_metric, str) or not self.eval_metric.strip():
            raise ValueError("study eval_metric must not be empty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("study direction must be 'minimize' or 'maximize'")
        _require_finite(self.min_delta, "study min_delta")
        if self.min_delta < 0:
            raise ValueError("study min_delta must be >= 0")
        if self.active_wave_index is not None:
            if isinstance(self.active_wave_index, bool) or not isinstance(
                self.active_wave_index, int
            ):
                raise TypeError("active_wave_index must be an integer or null")
            if self.active_wave_index < 1:
                raise ValueError("active_wave_index must be >= 1")
        # Build trial indexes once so relationship checks are deterministic.
        trial_ids = {trial.trial_id for trial in self.trials}
        trials_by_id = {trial.trial_id: trial for trial in self.trials}
        if len(trial_ids) != len(self.trials):
            raise ValueError("study state contains duplicate trial IDs")
        indexes = {trial.index for trial in self.trials}
        if len(indexes) != len(self.trials):
            raise ValueError("study state contains duplicate trial indexes")
        if tuple(trial.index for trial in self.trials) != tuple(sorted(indexes)):
            raise ValueError("study trials must be ordered by index")
        # Every trial must have a deterministic ID and point only to earlier trials.
        for trial in self.trials:
            expected_id = make_trial_id(self.study_id, trial.wave_index, trial.index)
            if trial.trial_id != expected_id:
                raise ValueError(
                    f"trial {trial.index} has ID {trial.trial_id!r}; expected {expected_id!r}"
                )
            if trial.parent_trial_id is not None and trial.parent_trial_id not in trial_ids:
                raise ValueError(
                    f"trial {trial.trial_id!r} references unknown parent {trial.parent_trial_id!r}"
                )
            if trial.parent_trial_id is not None:
                parent = trials_by_id[trial.parent_trial_id]
                if parent.index >= trial.index:
                    raise ValueError(f"trial {trial.trial_id!r} parent must be an earlier trial")
                if (
                    parent.wave_index > 0
                    and trial.wave_index > 0
                    and parent.wave_index >= trial.wave_index
                ):
                    raise ValueError(
                        f"trial {trial.trial_id!r} parent must be from an earlier wave"
                    )
                if trial.parent_revision and trial.parent_revision != parent.code_revision:
                    raise ValueError(
                        f"trial {trial.trial_id!r} parent revision does not match "
                        f"{parent.trial_id!r}"
                    )
            # Wave trials freeze parent score and revision at wave creation time.
            if trial.wave_index > 0:
                if (
                    trial.parent_trial_id is not None
                    and trials_by_id[trial.parent_trial_id].score is None
                ):
                    raise ValueError(f"trial {trial.trial_id!r} parent must be a scored trial")
                expected_parent_score = (
                    self.baseline_score
                    if trial.parent_trial_id is None
                    else trials_by_id[trial.parent_trial_id].score
                )
                if trial.parent_score != expected_parent_score:
                    raise ValueError(
                        f"trial {trial.trial_id!r} parent score does not match frozen parent"
                    )
                if not trial.parent_revision:
                    raise ValueError(
                        f"trial {trial.trial_id!r} must record its frozen parent revision"
                    )
                if (
                    trial.parent_trial_id is not None
                    and not trials_by_id[trial.parent_trial_id].code_revision
                ):
                    raise ValueError(f"trial {trial.trial_id!r} parent must have a code revision")
            # Composition sources must be earlier beneficial isolated changes.
            if (
                trial.composition_source_trial_id is not None
                and trial.composition_source_trial_id not in trial_ids
            ):
                raise ValueError(
                    f"trial {trial.trial_id!r} references unknown composition source "
                    f"{trial.composition_source_trial_id!r}"
                )
            if trial.composition_source_trial_id == trial.trial_id:
                raise ValueError(f"trial {trial.trial_id!r} cannot compose itself")
            if trial.composition_source_trial_id is not None:
                source = trials_by_id[trial.composition_source_trial_id]
                if source.index >= trial.index:
                    raise ValueError(f"trial {trial.trial_id!r} composition source must be earlier")
                if source.composition_source_trial_id is not None:
                    raise ValueError("composition source must be an original isolated change")
                if source.score is None or source.improved_over_parent is not True:
                    raise ValueError("composition source must be a beneficial scored trial")
        # Pending composition queues contain only useful original changes.
        if len(set(self.pending_change_trial_ids)) != len(self.pending_change_trial_ids):
            raise ValueError("pending change queue contains duplicate trial IDs")
        unknown_pending = set(self.pending_change_trial_ids) - trial_ids
        if unknown_pending:
            raise ValueError(
                "pending change queue references unknown trials: "
                + ", ".join(sorted(unknown_pending))
            )
        for source_id in self.pending_change_trial_ids:
            source = trials_by_id[source_id]
            if source.composition_source_trial_id is not None:
                raise ValueError("pending change queue must contain original isolated changes")
            if source.score is None or source.improved_over_parent is not True:
                raise ValueError("pending change queue must contain beneficial scored trials")
            if not source.code_revision:
                raise ValueError("pending change queue source must have a code revision")
        if self.active_wave_index is not None and not any(
            trial.wave_index == self.active_wave_index for trial in self.trials
        ):
            raise ValueError("active_wave_index does not reference any trials")
        # Best-trial bookkeeping must match the scored trial and cleanup queue.
        if self.best_trial_id is not None and self.best_trial_id not in trial_ids:
            raise ValueError(f"best_trial_id {self.best_trial_id!r} is not in trials")
        if self.best_trial_id is None and self.best_score != self.baseline_score:
            raise ValueError("best_score must equal baseline_score when no best trial is set")
        if self.best_trial_id is not None:
            best = trials_by_id[self.best_trial_id]
            if best.score is None or best.score != self.best_score:
                raise ValueError("best_score must equal the scored best trial")
            if not best.code_revision:
                raise ValueError("best trial must have a code revision")
            included = _included_change_ids(trials_by_id, best)
            if included.intersection(self.pending_change_trial_ids):
                raise ValueError("pending change queue contains a change already in the best")
        # Active waves cannot consume a queued source twice.
        if self.active_wave_index is not None:
            active_sources = {
                trial.composition_source_trial_id
                for trial in self.trials
                if trial.wave_index == self.active_wave_index
                and trial.composition_source_trial_id is not None
            }
            if active_sources.intersection(self.pending_change_trial_ids):
                raise ValueError("pending change is already assigned to the active wave")

        # All trials in one wave must share the same frozen parent context.
        waves: dict[int, set[tuple[str | None, str, float | None]]] = {}
        for trial in self.trials:
            if trial.wave_index > 0:
                waves.setdefault(trial.wave_index, set()).add(
                    (trial.parent_trial_id, trial.parent_revision, trial.parent_score)
                )
        if any(len(parents) != 1 for parents in waves.values()):
            raise ValueError("all trials in a wave must share one frozen parent")

    @classmethod
    def create(
        cls,
        *,
        study_id: str,
        baseline_score: float,
        base_commit: str,
        eval_metric: str,
        direction: ObjectiveDirection,
        min_delta: float,
    ) -> StudyState:
        """Create a fresh study state before any trials exist."""
        return cls(
            study_id=study_id,
            baseline_score=baseline_score,
            best_score=baseline_score,
            base_commit=base_commit,
            eval_metric=eval_metric,
            direction=direction,
            min_delta=min_delta,
        )

    @property
    def next_trial_index(self) -> int:
        """Return the next monotonic trial index."""
        return max((trial.index for trial in self.trials), default=0) + 1

    @property
    def next_trial_id(self) -> str:
        """Return the deterministic ID for the next trial."""
        return make_trial_id(self.study_id, self.next_wave_index, self.next_trial_index)

    @property
    def next_wave_index(self) -> int:
        """Return the next monotonic wave index."""
        return max((trial.wave_index for trial in self.trials), default=0) + 1

    def with_trial(self, trial: Trial) -> StudyState:
        """Append a new trial while preserving immutable state semantics."""
        if any(existing.trial_id == trial.trial_id for existing in self.trials):
            raise ValueError(f"trial {trial.trial_id!r} already exists")
        return replace(self, trials=(*self.trials, trial))

    def get_trial(self, trial_id: str | None) -> Trial:
        """Return one trial by ID."""
        if trial_id is None:
            raise KeyError("trial ID is required")
        try:
            return next(trial for trial in self.trials if trial.trial_id == trial_id)
        except StopIteration as exc:
            raise KeyError(f"unknown trial {trial_id!r}") from exc

    def with_updated_trial(self, trial: Trial) -> StudyState:
        """Return state with one trial replaced by ID."""
        found = False
        updated: list[Trial] = []
        for existing in self.trials:
            if existing.trial_id == trial.trial_id:
                updated.append(trial)
                found = True
            else:
                updated.append(existing)
        if not found:
            raise KeyError(f"unknown trial {trial.trial_id!r}")
        return replace(self, trials=tuple(updated))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object into durable JSON-compatible data."""
        return {
            "study_id": self.study_id,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "base_commit": self.base_commit,
            "eval_metric": self.eval_metric,
            "direction": self.direction,
            "min_delta": self.min_delta,
            "best_trial_id": self.best_trial_id,
            "active_wave_index": self.active_wave_index,
            "pending_change_trial_ids": list(self.pending_change_trial_ids),
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudyState:
        """Build a typed object from durable or config data."""
        mapping = _require_mapping(data, "study state")
        raw_trials = mapping.get("trials", ())
        if not isinstance(raw_trials, list | tuple):
            raise TypeError("study state.trials must be a list")
        return cls(
            study_id=_require_string(mapping, "study_id", "study state"),
            baseline_score=_require_number(mapping, "baseline_score", "study state"),
            best_score=_require_number(mapping, "best_score", "study state"),
            base_commit=_require_string(mapping, "base_commit", "study state"),
            eval_metric=_require_string(mapping, "eval_metric", "study state"),
            direction=_require_string(mapping, "direction", "study state"),  # type: ignore[arg-type]
            min_delta=_require_number(mapping, "min_delta", "study state"),
            best_trial_id=_optional_string(mapping, "best_trial_id", "study state"),
            active_wave_index=_optional_nullable_int(mapping, "active_wave_index", "study state"),
            pending_change_trial_ids=_string_tuple(
                mapping.get("pending_change_trial_ids", ()),
                "study state.pending_change_trial_ids",
            ),
            trials=tuple(Trial.from_dict(item) for item in raw_trials),
        )


def make_trial_id(study_id: str, wave_index: int, index: int) -> str:
    """Return ``{study_id}_wave{wave_index}_trial{index}``."""
    if isinstance(wave_index, bool) or not isinstance(wave_index, int):
        raise TypeError("wave index must be an integer")
    if wave_index < 0:
        raise ValueError("wave index must be >= 0")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("trial index must be an integer")
    if index < 1:
        raise ValueError("trial index must be >= 1")
    slug = re.sub(r"[^a-z0-9.-]+", "-", study_id.lower()).strip(".-")
    if not slug:
        raise ValueError("study_id must contain a letter or number")
    return f"{slug}_wave{wave_index}_trial{index}"


def _included_change_ids(trials_by_id: Mapping[str, Trial], trial: Trial) -> set[str]:
    """Trace isolated changes already included in a composed best recipe."""
    included: set[str] = set()
    current: Trial | None = trial
    while current is not None:
        included.add(current.composition_source_trial_id or current.trial_id)
        current = (
            trials_by_id[current.parent_trial_id] if current.parent_trial_id is not None else None
        )
    return included


def _require_finite(value: float, label: str) -> None:
    """Enforce one state invariant with a focused error."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Enforce one state invariant with a focused error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_string(data: Mapping[str, Any], key: str, label: str) -> str:
    """Enforce one state invariant with a focused error."""
    if key not in data:
        raise KeyError(f"{label}.{key} is required")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_string(
    data: Mapping[str, Any],
    key: str,
    label: str,
    *,
    default: str | None = None,
) -> str | None:
    """Read one optional persisted field with strict type checks."""
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label}.{key} must be a string or null")
    return value


def _require_number(data: Mapping[str, Any], key: str, label: str) -> float:
    """Enforce one state invariant with a focused error."""
    if key not in data:
        raise KeyError(f"{label}.{key} is required")
    value = data[key]
    _require_finite(value, f"{label}.{key}")
    return float(value)


def _optional_number(
    data: Mapping[str, Any],
    key: str,
    label: str,
    *,
    default: float | None = None,
) -> float | None:
    """Read one optional persisted field with strict type checks."""
    value = data.get(key, default)
    if value is None:
        return None
    _require_finite(value, f"{label}.{key}")
    return float(value)


def _require_int(data: Mapping[str, Any], key: str, label: str) -> int:
    """Enforce one state invariant with a focused error."""
    if key not in data:
        raise KeyError(f"{label}.{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer")
    return value


def _optional_int(data: Mapping[str, Any], key: str, label: str, *, default: int) -> int:
    """Read one optional persisted field with strict type checks."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer")
    return value


def _optional_nullable_int(data: Mapping[str, Any], key: str, label: str) -> int | None:
    """Read one optional persisted field with strict type checks."""
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer or null")
    return value


def _optional_bool(data: Mapping[str, Any], key: str, label: str) -> bool | None:
    """Read one optional persisted field with strict type checks."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{label}.{key} must be a boolean or null")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    """Parse string-like persisted input with strict type checks."""
    if not isinstance(value, list | tuple):
        raise TypeError(f"{label} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} entries must be strings")
    return tuple(value)
