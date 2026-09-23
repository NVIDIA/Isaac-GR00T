# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Strict recipe loading for public N2 benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from groot.infra.closed_loop.budgets import resolve_success_step_budgets

BENCHMARK_SPEC_SCHEMA_VERSION = "groot-n2-benchmark-spec-v2"

_EXPERIMENT = re.compile(r"[a-z][a-z0-9_]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VIDEO_BACKENDS = {"decord", "torchvision_av", "torchcodec"}
_CODECS = {"h264", "none", "mp4v"}
_RESERVED_OVERRIDES = {
    "compute.num_gpus",
    "compute.per_device_batch_size",
    "data.datasets",
    "data.seed",
    "training.max_steps",
    "training.output_dir",
    "training.start_from_checkpoint",
}
_SENSITIVE = re.compile(
    r"(?:^|[._-])(?:api.?key|auth|credential|password|secret|token)(?:$|[._-])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated benchmark recipe plus its retained source identity."""

    path: Path
    digest: str
    benchmark: Mapping[str, Any]
    dataset: Mapping[str, Any]
    base_checkpoint: Mapping[str, Any]
    codec: str
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    expectations: Mapping[str, Mapping[str, Any]]


def load_benchmark_spec(recipe: str, *, builtin_dir: Path | None = None) -> BenchmarkSpec:
    """Load an explicit spec path or an approved built-in recipe."""
    path = _resolve_path(recipe, builtin_dir)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Benchmark recipe is not valid JSON: {path}: {exc}") from exc

    root = _object(payload, "recipe")
    _keys(
        root,
        {
            "schema_version",
            "benchmark",
            "dataset",
            "base_checkpoint",
            "codec",
            "training",
            "evaluation",
            "expectations",
        },
        "recipe",
    )
    if root["schema_version"] != BENCHMARK_SPEC_SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark recipe schema: {root['schema_version']!r}")
    codec = _string(root["codec"], "codec").lower()
    if codec not in _CODECS:
        raise ValueError(f"codec must be one of {sorted(_CODECS)}, got {codec!r}")

    benchmark = _object(root["benchmark"], "benchmark")
    _keys(benchmark, {"recipe_id", "recipe_version", "provenance"}, "benchmark")
    _strings(benchmark, "benchmark")

    dataset = _object(root["dataset"], "dataset")
    _keys(dataset, {"path", "sha256"} if "path" in dataset else {"repo_id", "revision"}, "dataset")
    _strings(dataset, "dataset")
    if "path" in dataset:
        if not _SHA256.fullmatch(dataset["sha256"]):
            raise ValueError("dataset.sha256 must be a lowercase SHA-256 fingerprint")
    else:
        _revision(dataset["revision"], "dataset.revision")

    checkpoint = _object(root["base_checkpoint"], "base_checkpoint")
    _keys(checkpoint, {"uri", "revision"}, "base_checkpoint")
    _strings(checkpoint, "base_checkpoint")
    _revision(checkpoint["revision"], "base_checkpoint.revision")
    if not checkpoint["uri"].startswith("hf://model/"):
        raise ValueError("base_checkpoint.uri must be a pinned hf://model/... URI")
    uri_revision = checkpoint["uri"].partition("@")[2].partition("/")[0]
    if uri_revision != checkpoint["revision"]:
        raise ValueError("base_checkpoint.uri must contain base_checkpoint.revision")

    training = _object(root["training"], "training")
    _keys(
        training,
        {
            "experiment",
            "compute_type",
            "embodiment_tag",
            "seed",
            "max_steps",
            "num_gpus",
            "batch_size",
            "config_overrides",
        },
        "training",
    )
    for key in ("experiment", "compute_type", "embodiment_tag"):
        _string(training[key], f"training.{key}")
    if _EXPERIMENT.fullmatch(training["experiment"]) is None:
        raise ValueError("training.experiment must name a public one-launcher experiment")
    _integer(training["seed"], "training.seed", minimum=0)
    for key in ("max_steps", "num_gpus", "batch_size"):
        _integer(training[key], f"training.{key}", minimum=1)
    _overrides(training["config_overrides"])

    evaluation = _object(root["evaluation"], "evaluation")
    if not evaluation or set(evaluation) - {"open_loop", "closed_loop"}:
        raise ValueError("evaluation must select open_loop, closed_loop, or both")
    if "open_loop" in evaluation:
        open_loop = _object(evaluation["open_loop"], "evaluation.open_loop")
        _validate_open_loop(open_loop)
    if "closed_loop" in evaluation:
        _validate_closed_loop(_object(evaluation["closed_loop"], "evaluation.closed_loop"))

    expectations = _object(root["expectations"], "expectations")
    if set(expectations) - required_metrics(evaluation):
        raise ValueError("Benchmark expectations must match the selected evaluation metrics")
    for metric, raw_bounds in expectations.items():
        bounds = _object(raw_bounds, f"expectations.{metric}")
        _keys(bounds, {"minimum", "maximum", "provenance"}, f"expectations.{metric}")
        minimum = _number(bounds["minimum"], f"expectations.{metric}.minimum")
        maximum = _number(bounds["maximum"], f"expectations.{metric}.maximum")
        if minimum > maximum:
            raise ValueError(f"expectations.{metric}.minimum exceeds maximum")
        _string(bounds["provenance"], f"expectations.{metric}.provenance")

    digest = hashlib.sha256(
        json.dumps(root, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkSpec(
        path=path.resolve(),
        digest=digest,
        benchmark=benchmark,
        dataset=dataset,
        base_checkpoint=checkpoint,
        codec=codec,
        training=training,
        evaluation=evaluation,
        expectations=expectations,
    )


def _validate_open_loop(evaluation: Mapping[str, Any]) -> None:
    _keys(
        evaluation,
        {
            "trajectory_ids",
            "steps",
            "action_horizon",
            "denoising_steps",
            "video_backend",
            "local_only",
            "server_parity_tolerance",
        },
        "evaluation",
    )
    trajectories = evaluation["trajectory_ids"]
    if not isinstance(trajectories, list) or not trajectories:
        raise TypeError("evaluation.trajectory_ids must be a non-empty array")
    for index, trajectory in enumerate(trajectories):
        _integer(trajectory, f"evaluation.trajectory_ids[{index}]", minimum=0)
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("evaluation.trajectory_ids must not contain duplicates")
    for key in ("steps", "action_horizon", "denoising_steps"):
        _integer(evaluation[key], f"evaluation.{key}", minimum=1)
    if evaluation["video_backend"] not in _VIDEO_BACKENDS:
        raise ValueError("evaluation.video_backend is unsupported")
    if not isinstance(evaluation["local_only"], bool):
        raise TypeError("evaluation.local_only must be a boolean")
    _number(
        evaluation["server_parity_tolerance"],
        "evaluation.server_parity_tolerance",
        minimum=0,
    )


def required_metrics(evaluation: Mapping[str, Any]) -> set[str]:
    """Metrics that must be present and finite, even without acceptance ranges."""
    metrics = {"training.final_loss"}
    if "open_loop" in evaluation:
        labels = ("local",) if evaluation["open_loop"]["local_only"] else ("local", "server")
        metrics |= {
            f"evaluation.open_loop.{label}.{metric}"
            for label in labels
            for metric in ("mse", "mae")
        }
    if "closed_loop" in evaluation:
        metrics.add("evaluation.closed_loop.success_rate")
    return metrics


def _validate_closed_loop(evaluation: Mapping[str, Any]) -> None:
    _keys(
        {key: value for key, value in evaluation.items() if key != "success_step_budgets"},
        {
            "backend",
            "suite",
            "episodes_per_task",
            "seed",
            "max_episode_steps",
            "action_steps",
            "denoising_steps",
        },
        "evaluation.closed_loop",
    )
    for key in ("backend", "suite"):
        _string(evaluation[key], f"evaluation.closed_loop.{key}")
    if len(evaluation["backend"].split(":")) != 2:
        raise ValueError("evaluation.closed_loop.backend must use module:Class")
    _integer(evaluation["seed"], "evaluation.closed_loop.seed", minimum=0)
    for key in ("episodes_per_task", "max_episode_steps", "action_steps", "denoising_steps"):
        _integer(evaluation[key], f"evaluation.closed_loop.{key}", minimum=1)
    if "success_step_budgets" in evaluation:
        budgets = evaluation["success_step_budgets"]
        if not isinstance(budgets, list):
            raise TypeError("evaluation.closed_loop.success_step_budgets must be an array")
        resolve_success_step_budgets(budgets, evaluation["max_episode_steps"])


def _resolve_path(recipe: str, builtin_dir: Path | None) -> Path:
    candidate = Path(recipe).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.suffix == ".json" or len(candidate.parts) > 1:
        raise FileNotFoundError(f"Benchmark recipe file does not exist: {candidate}")

    directory = builtin_dir or Path(__file__).with_name("specs")
    built_in = directory / f"{recipe}.json"
    if built_in.is_file():
        return built_in
    raise ValueError(
        f"Unknown benchmark recipe {recipe!r}. No approved built-in recipe matches it. "
        "Pass a path to an approved benchmark spec JSON."
    )


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{label} fields are incomplete or unsupported; missing={missing}, extra={extra}"
        )


def _strings(value: Mapping[str, Any], label: str) -> None:
    for key, item in value.items():
        _string(item, f"{label}.{key}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _revision(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(f"{label} must be a full 40-character lowercase commit revision")


def _integer(value: Any, label: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}")


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return number


def _overrides(value: Any) -> None:
    if not isinstance(value, list):
        raise TypeError("training.config_overrides must be an array")
    seen: set[str] = set()
    for override in value:
        if not isinstance(override, str) or "=" not in override:
            raise TypeError("Each training.config_overrides entry must be PATH=VALUE")
        key = override.partition("=")[0].strip().replace("-", "_")
        if any(
            key == field or key.startswith(field + ".") or field.startswith(key + ".")
            for field in _RESERVED_OVERRIDES
        ):
            raise ValueError(f"Config override replaces runner-owned field {key!r}")
        if _SENSITIVE.search(key):
            raise ValueError(f"Config override contains sensitive field {key!r}")
        if key in seen:
            raise ValueError(f"Duplicate config override for {key!r}")
        seen.add(key)
