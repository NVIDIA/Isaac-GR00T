# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact handoff helpers for the AutoResearch script runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


# Script training artifacts follow the baseline-folder contract supplied by the user.
@dataclass(frozen=True)
class ScriptTrainingArtifacts:
    output_dir: Path
    checkpoint: Path


# Script evaluation artifacts keep the original JSON payload with the extracted metric.
@dataclass(frozen=True)
class ScriptEvaluationArtifacts:
    result_path: Path
    score: float


def collect_script_training_artifacts(
    *,
    run_root: Path,
    checkpoint_path: Path | None = None,
    output_dir_name: str = "training",
) -> ScriptTrainingArtifacts:
    """Locate a checkpoint produced by a user-owned train script."""
    # Script artifacts are scoped to the per-trial run root supplied by config.
    train_root = run_root.resolve()
    output_dir = (train_root / output_dir_name).resolve()
    if checkpoint_path is not None:
        checkpoint = checkpoint_path.resolve()
    else:
        checkpoint = _latest_checkpoint(output_dir)
        if checkpoint is None:
            checkpoint = output_dir
    _require_within(output_dir, train_root, "training output")
    _require_within(checkpoint, train_root, "training checkpoint")
    _require_checkpoint_shape(checkpoint)
    return ScriptTrainingArtifacts(
        output_dir=output_dir,
        checkpoint=checkpoint,
    )


def collect_script_evaluation_artifacts(
    *,
    result_file: Path,
    result_path: str,
) -> ScriptEvaluationArtifacts:
    """Read a user-owned eval result file and extract the configured scalar metric."""
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid evaluation result JSON: {result_file}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"Evaluation result must be a JSON object: {result_file}"
        )
    # The objective metric is a finite scalar at the configured dot path.
    score_value = lookup_result_path(payload, result_path)
    if isinstance(score_value, bool) or not isinstance(score_value, int | float):
        raise RuntimeError(  # noqa: TRY004
            f"Metric {result_path!r} is not numeric in {result_file}"
        )
    score = float(score_value)
    if not math.isfinite(score):
        raise RuntimeError(f"Metric {result_path!r} is not finite in {result_file}")
    return ScriptEvaluationArtifacts(result_path=result_file, score=score)


def lookup_result_path(payload: Mapping[str, Any], result_path: str) -> Any:
    """Resolve a dot-separated mapping path without a JSONPath dependency."""
    current: Any = payload
    for component in result_path.split("."):
        if not component or not isinstance(current, Mapping) or component not in current:
            raise RuntimeError(f"Evaluation result is missing metric path {result_path!r}")
        current = current[component]
    return current


def _latest_checkpoint(output_dir: Path) -> Path | None:
    """Choose the highest numbered checkpoint in a training output."""
    if not output_dir.is_dir():
        return None
    candidates = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key).resolve()


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort checkpoint directories by numeric suffix, then name."""
    suffix = path.name.rsplit("-", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, path.name)


def _require_checkpoint_shape(checkpoint: Path) -> None:
    """Keep artifact paths and shapes explicit so stale outputs are rejected."""
    if not checkpoint.is_dir():
        raise RuntimeError(f"Training checkpoint does not exist: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"Training checkpoint is missing config.json: {checkpoint}")
    if not any(checkpoint.glob("*.safetensors")):
        raise RuntimeError(f"Training checkpoint has no safetensors weights: {checkpoint}")


def _require_within(path: Path, parent: Path, label: str) -> None:
    """Keep artifact paths and shapes explicit so stale outputs are rejected."""
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"{label} {path} is outside expected root {parent}") from exc
