# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Ray orchestration and artifacts for portable closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Protocol
import uuid

from groot.infra.closed_loop.budgets import resolve_success_step_budgets
from groot.infra.closed_loop.policies import load_n2_policy
from groot.infra.closed_loop.rollout import (
    TaskAssignment,
    build_task_assignments,
    run_vectorized_rollout,
)


class ClosedLoopBackend(Protocol):
    """Simulator plugin contract; vector environments execute one action per step call."""

    def __init__(self, *, suite: str, options: Mapping[str, Any]) -> None: ...

    @classmethod
    def suite_task_names(cls, suite: str) -> tuple[str, ...]: ...

    def create_vector_env(
        self,
        *,
        task: str,
        num_envs: int,
        seed: int,
        max_episode_steps: int | None,
        record_video: bool,
        video_dir: str,
    ) -> Any: ...


class ClosedLoopEvalError(RuntimeError):
    """Evaluation failed after its diagnostic report was preserved."""

    def __init__(self, message: str, *, report_path: Path) -> None:
        super().__init__(f"{message}. See {report_path}")
        self.report_path = report_path


@dataclass(frozen=True)
class ClosedLoopPolicyConfig:
    """N2 policy configuration passed through the evaluator."""

    checkpoint: str
    cache_root: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("policy checkpoint cannot be empty")
        if any(not isinstance(name, str) or not name for name in self.options):
            raise ValueError("policy option names must be non-empty strings")


@dataclass(frozen=True)
class ClosedLoopEvalConfig:
    """Portable configuration for one local or existing-Ray evaluation."""

    policy: ClosedLoopPolicyConfig
    backend: str
    suite: str
    output_dir: Path = Path("benchmark-results")
    tasks: tuple[str, ...] = ()
    episodes_per_task: int = 1
    num_workers: int | None = None
    ray_address: str | None = None
    cpus_per_worker: float = 1.0
    gpus_per_worker: float = 1.0
    seed: int = 0
    max_episode_steps: int | None = None
    record_video: bool = True
    backend_options: dict[str, Any] = field(default_factory=dict)
    success_step_budgets: Sequence[int | float] | None = None

    def __post_init__(self) -> None:
        _split_class_path(self.backend)
        if not self.suite:
            raise ValueError("suite cannot be empty")
        if self.episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive")
        if self.num_workers is not None and self.num_workers <= 0:
            raise ValueError("num_workers must be positive when provided")
        if self.cpus_per_worker <= 0 or self.gpus_per_worker < 0:
            raise ValueError("worker resources must be non-negative and include a positive CPU")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when provided")
        resolve_success_step_budgets(self.success_step_budgets, self.max_episode_steps)

    @property
    def resolved_success_step_budgets(self) -> tuple[int, ...]:
        """Resolve percentage budgets against the configured episode limit."""
        return resolve_success_step_budgets(self.success_step_budgets, self.max_episode_steps)

    @property
    def resolved_ray_address(self) -> str | None:
        """Prefer the explicit Ray address, then the standard environment variable."""
        return self.ray_address or os.environ.get("RAY_ADDRESS") or None


@dataclass(frozen=True)
class ClosedLoopRunPaths:
    """Filesystem artifacts produced by one evaluation request."""

    output_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.output_dir / "runs" / self.run_id

    @property
    def request_path(self) -> Path:
        return self.run_dir / "request.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "benchmark.json"

    @property
    def history_path(self) -> Path:
        return self.output_dir / "history.jsonl"


class ClosedLoopRayWorker:
    """Load one policy replica and evaluate shared task assignments."""

    def __init__(
        self,
        *,
        backend_path: str,
        checkpoint: str,
        cache_root: str | None,
        policy_options: dict[str, Any],
        suite: str,
        backend_options: dict[str, Any],
    ) -> None:
        self._backend = _load_class(backend_path)(suite=suite, options=backend_options)
        self._policy = load_n2_policy(
            checkpoint=checkpoint,
            cache_root=cache_root,
            options=policy_options,
        )

    def evaluate(
        self,
        assignment: TaskAssignment,
        *,
        seed: int,
        max_episode_steps: int | None,
        record_video: bool,
        run_dir: str,
    ) -> dict[str, Any]:
        env = self._backend.create_vector_env(
            task=assignment.task,
            num_envs=1,
            seed=seed,
            max_episode_steps=max_episode_steps,
            record_video=record_video,
            video_dir=str(Path(run_dir) / "videos"),
        )
        result = run_vectorized_rollout(
            env=env,
            policy=self._policy,
            num_envs=1,
            num_episodes=assignment.num_episodes,
            record_episode_steps=True,
        )
        return {
            "task": assignment.task,
            "episodes": assignment.num_episodes,
            "successes": list(result.successes[: assignment.num_episodes]),
            "episode_infos": {
                name: list(values[: assignment.num_episodes])
                for name, values in result.episode_infos.items()
            },
        }


class ClosedLoopEvaluator:
    """Run a backend plugin through the shared evaluator and persist its report."""

    def __init__(self, config: ClosedLoopEvalConfig) -> None:
        self.config = config

    def run(self) -> Path:
        paths = ClosedLoopRunPaths(self.config.output_dir.expanduser().resolve(), _run_id())
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        started_at = _timestamp()
        ray_started_here = False
        ray_context: dict[str, Any] = {}
        try:
            import ray

            if not ray.is_initialized():
                context = ray.init(address=self.config.resolved_ray_address)
                ray_started_here = True
                ray_context = {"address": context.address_info.get("address")}
            else:
                ray_context = {"address": ray.get_runtime_context().gcs_address}
            backend_class = _load_class(self.config.backend)
            available_tasks = tuple(backend_class.suite_task_names(self.config.suite))
            tasks = _select_tasks(available_tasks, self.config.tasks)
            scheduled_worker_count = _worker_count(self.config, ray.cluster_resources(), len(tasks))
            assignments = tuple(
                assignment
                for assignment in build_task_assignments(
                    list(tasks),
                    num_workers=scheduled_worker_count,
                    num_episodes=self.config.episodes_per_task,
                )
                if assignment.num_episodes
            )
            worker_count = min(scheduled_worker_count, len(assignments))
            _write_json(paths.request_path, _request(self.config, paths, tasks, worker_count))
            results = _run_assignments(
                ray=ray,
                config=self.config,
                paths=paths,
                assignments=assignments,
                worker_count=min(worker_count, len(assignments)),
            )
            report = _completed_report(
                config=self.config,
                paths=paths,
                tasks=tasks,
                results=results,
                worker_count=worker_count,
                ray_context=ray_context,
                started_at=started_at,
            )
            _write_json(paths.report_path, report)
            _append_history(paths, report)
            return paths.report_path
        except Exception as error:
            report = _failed_report(
                config=self.config,
                paths=paths,
                ray_context=ray_context,
                started_at=started_at,
                error=error,
            )
            _write_json(paths.report_path, report)
            _append_history(paths, report)
            raise ClosedLoopEvalError(str(error), report_path=paths.report_path) from error
        finally:
            if ray_started_here:
                ray.shutdown()


def _run_assignments(
    *,
    ray: Any,
    config: ClosedLoopEvalConfig,
    paths: ClosedLoopRunPaths,
    assignments: Sequence[TaskAssignment],
    worker_count: int,
) -> list[dict[str, Any]]:
    worker_type = ray.remote(
        num_cpus=config.cpus_per_worker,
        num_gpus=config.gpus_per_worker,
    )(ClosedLoopRayWorker)
    workers = []
    try:
        for _ in range(worker_count):
            policy_options = _worker_policy_options(
                config.policy.options,
                rebuild_token=paths.run_id,
            )
            worker = worker_type.remote(
                backend_path=config.backend,
                checkpoint=config.policy.checkpoint,
                cache_root=config.policy.cache_root,
                policy_options=policy_options,
                suite=config.suite,
                backend_options=config.backend_options,
            )
            workers.append(worker)
        references = [
            workers[index % worker_count].evaluate.remote(
                assignment,
                seed=config.seed + index * config.episodes_per_task,
                max_episode_steps=config.max_episode_steps,
                record_video=config.record_video,
                run_dir=str(paths.run_dir),
            )
            for index, assignment in enumerate(assignments)
        ]
        return list(ray.get(references))
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)


def _worker_policy_options(options: dict[str, Any], *, rebuild_token: str) -> dict[str, Any]:
    """Coordinate one forced rebuild for each runtime/cache identity."""
    worker_options = dict(options)
    if worker_options.get("rebuild_trt"):
        worker_options["trt_rebuild_token"] = rebuild_token
    return worker_options


def _select_tasks(available: tuple[str, ...], requested: tuple[str, ...]) -> tuple[str, ...]:
    if not available or len(set(available)) != len(available):
        raise ValueError("backend suite tasks must be non-empty and unique")
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"tasks are not part of the suite: {', '.join(unknown)}")
    return requested


def _worker_count(
    config: ClosedLoopEvalConfig, resources: Mapping[str, float], task_count: int
) -> int:
    if config.num_workers is not None:
        return config.num_workers
    by_cpu = max(1, int(resources.get("CPU", 1) // config.cpus_per_worker))
    if config.gpus_per_worker:
        by_gpu = max(1, int(resources.get("GPU", 0) // config.gpus_per_worker))
        return min(task_count, by_cpu, by_gpu)
    return min(task_count, by_cpu)


def _request(
    config: ClosedLoopEvalConfig,
    paths: ClosedLoopRunPaths,
    tasks: Sequence[str],
    worker_count: int,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["tasks"] = list(tasks)
    return {"run_id": paths.run_id, "workers": worker_count, **payload}


def _completed_report(
    *,
    config: ClosedLoopEvalConfig,
    paths: ClosedLoopRunPaths,
    tasks: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    worker_count: int,
    ray_context: Mapping[str, Any],
    started_at: str,
) -> dict[str, Any]:
    budgets = config.resolved_success_step_budgets
    per_task: dict[str, dict[str, Any]] = {}
    episode_results = [
        {
            "task": result["task"],
            "success": bool(success),
            "steps_executed": steps,
            "first_success_step": first_success_step,
        }
        for result in results
        for success, steps, first_success_step in zip(
            result["successes"],
            result["episode_infos"]["steps_executed"],
            result["episode_infos"]["first_success_step"],
            strict=True,
        )
    ]
    for task in tasks:
        task_episodes = [episode for episode in episode_results if episode["task"] == task]
        successes = [episode["success"] for episode in task_episodes]
        if len(successes) != config.episodes_per_task:
            raise ValueError(
                f"backend returned {len(successes)} episodes for {task}; expected {config.episodes_per_task}"
            )
        per_task[task] = {
            "episodes": len(successes),
            "successes": sum(successes),
            "success_rate": sum(successes) / len(successes),
            **_success_budget_summary(task_episodes, budgets),
        }
    episodes = sum(item["episodes"] for item in per_task.values())
    successes = sum(item["successes"] for item in per_task.values())
    task_success_rates = [item["success_rate"] for item in per_task.values()]
    return {
        "run_id": paths.run_id,
        "status": "completed",
        "started_at": started_at,
        "completed_at": _timestamp(),
        "model": asdict(config.policy),
        "benchmark": {
            "backend": config.backend,
            "suite": config.suite,
            "tasks": list(tasks),
            "episodes_per_task": config.episodes_per_task,
        },
        "execution": {"ray": dict(ray_context), "workers": worker_count},
        "episode_results": episode_results,
        "summary": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "task_success_rate_mean": fmean(task_success_rates),
            "task_success_rate_std": pstdev(task_success_rates),
            "per_task": per_task,
            **_success_budget_summary(episode_results, budgets),
        },
        "artifacts": {
            "request": str(paths.request_path),
            "videos_dir": str(paths.run_dir / "videos"),
        },
        "error": None,
    }


def _success_budget_summary(
    episodes: Sequence[Mapping[str, Any]], budgets: Sequence[int]
) -> dict[str, Any]:
    """Report success by budget and normalized area under the empirical step curve.

    Integrate from zero to the largest budget: a success at step s contributes
    max(horizon - s, 0) / horizon. Failures contribute zero to every metric.
    """
    if not budgets:
        return {}
    horizon = max(budgets)
    return {
        "success_rate_by_step_budget": {
            str(budget): sum(
                episode["first_success_step"] is not None
                and episode["first_success_step"] <= budget
                for episode in episodes
            )
            / len(episodes)
            for budget in budgets
        },
        "success_rate_auc": sum(
            max(horizon - episode["first_success_step"], 0)
            for episode in episodes
            if episode["first_success_step"] is not None
        )
        / (len(episodes) * horizon),
    }


def _failed_report(
    *,
    config: ClosedLoopEvalConfig,
    paths: ClosedLoopRunPaths,
    ray_context: Mapping[str, Any],
    started_at: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "run_id": paths.run_id,
        "status": "failed",
        "started_at": started_at,
        "completed_at": _timestamp(),
        "model": asdict(config.policy),
        "benchmark": {"backend": config.backend, "suite": config.suite},
        "execution": {"ray": dict(ray_context)},
        "summary": None,
        "artifacts": {
            "request": str(paths.request_path),
            "videos_dir": str(paths.run_dir / "videos"),
        },
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _append_history(paths: ClosedLoopRunPaths, report: Mapping[str, Any]) -> None:
    paths.history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": report["run_id"],
        "status": report["status"],
        "completed_at": report["completed_at"],
        "checkpoint": report["model"]["checkpoint"],
        "backend": report["benchmark"]["backend"],
        "suite": report["benchmark"]["suite"],
        "summary": report.get("summary"),
        "report_path": str(paths.report_path),
    }
    with paths.history_path.open("a", encoding="utf-8") as history:
        fcntl.flock(history.fileno(), fcntl.LOCK_EX)
        history.write(json.dumps(record, sort_keys=True) + "\n")
        history.flush()
        fcntl.flock(history.fileno(), fcntl.LOCK_UN)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_class(path: str) -> type:
    module_name, class_name = _split_class_path(path)
    value = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(value, type):
        raise TypeError(f"{path} does not resolve to a class")
    return value


def _split_class_path(path: str) -> tuple[str, str]:
    module_name, separator, class_name = path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("class path must use the form 'module:Class'")
    return module_name, class_name


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"
