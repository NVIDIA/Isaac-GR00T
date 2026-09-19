# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Apply a benchmark contract to the existing OSS train/eval path."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Protocol

from groot.core.policy.benchmark_contract import checkpoint_identity, git_identity
from groot.infra.benchmark.dataset import resolve_dataset
from groot.infra.benchmark.report import (
    REPORT_FILENAME,
    REPORT_SCHEMA_VERSION,
    bounded_artifact_identity,
    safe_path,
    sanitize_command,
    sanitize_text,
    to_json_text,
    write_benchmark_report,
)
from groot.infra.benchmark.spec import BenchmarkSpec, load_benchmark_spec, required_metrics
from groot.infra.closed_loop.runner import (
    ClosedLoopEvalConfig,
    ClosedLoopEvalError,
    ClosedLoopEvaluator,
    ClosedLoopPolicyConfig,
)
from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator
from groot.infra.launcher.one_launcher.training_experiment_resolver import DEFAULT_EXPERIMENT_DIR

PASSED_EXIT_CODE = 0
ERROR_EXIT_CODE = 1
FAILED_EXIT_CODE = 2


class ResolvedTrain(Protocol):
    command: list[str]
    output_dir: Path
    final_checkpoint: Path


class BenchmarkRun:
    """Collect provenance and checks from the shared train-eval sequence."""

    def __init__(self, args: argparse.Namespace, report: dict[str, Any]):
        self.args = args
        self.report = report
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.spec: BenchmarkSpec | None = None
        self.stage = "source"

    def record_resolution(self, resolved: ResolvedTrain) -> None:
        config_path = _resolved_config_path(resolved.command)
        if self.args.dry_run:
            sys.stdout.write(
                to_json_text(
                    _dry_run_plan(
                        self.report,
                        resolved=resolved,
                        config_path=config_path,
                        output_dir=self.output_dir,
                    )
                )
            )
            return
        self.stage = "environment"
        self.report["execution"]["environment"] = _execution_environment()
        retained_config = self.output_dir / "resolved-training-config.yaml"
        shutil.copyfile(config_path, retained_config)
        identity = bounded_artifact_identity(retained_config, report_root=self.output_dir)
        self.report["inputs"]["resolved_training_configuration"] = identity
        self.report["artifacts"]["resolved_training_configuration"] = identity
        self.report["inputs"]["dataset"]["prepared_metadata"] = {
            path.relative_to(self.args.dataset_path).as_posix(): bounded_artifact_identity(
                path, report_root=self.output_dir
            )
            for path in sorted((Path(self.args.dataset_path) / "meta").rglob("*"))
            if path.is_file()
        }

    def record_training(self, resolved: ResolvedTrain) -> None:
        checkpoint = Path(resolved.final_checkpoint)
        trainer_state_path = checkpoint / "trainer_state.json"
        completed_steps, final_loss = _training_metrics(trainer_state_path)
        identity = _checkpoint_metadata_identity(checkpoint, report_root=self.output_dir)
        self.report["training"].update(
            {
                "completed_steps": completed_steps,
                "final_loss": _json_number(final_loss),
                "checkpoint": identity,
            }
        )
        self.report["artifacts"]["trainer_state"] = bounded_artifact_identity(
            trainer_state_path, report_root=self.output_dir
        )
        self.report["artifacts"]["checkpoint_metadata"] = identity

    def evaluate(self, checkpoint: Path) -> int:
        """Run each selected existing evaluator against the one trained checkpoint."""
        assert self.spec is not None
        observed = {}
        for mode in ("open_loop", "closed_loop"):
            if mode not in self.spec.evaluation:
                continue
            settings = self.spec.evaluation[mode]
            self.stage = f"evaluation.{mode}"
            evaluation = self.report["evaluation"][mode]
            if mode == "open_loop":
                config = OssOpenLoopEvalConfig(
                    checkpoint=str(checkpoint),
                    dataset_path=self.args.dataset_path,
                    embodiment_tag=self.args.embodiment_tag,
                    output_json=self.args.output_json,
                    traj_ids=settings["trajectory_ids"],
                    trt_engine_dir=self.args.trt_engine_dir or None,
                    enable_trt=self.args.enable_trt,
                    trt_cache_dir=self.args.trt_cache_dir or None,
                    rebuild_trt=self.args.rebuild_trt,
                    use_cuda_graph=self.args.use_cuda_graph,
                    **{key: value for key, value in settings.items() if key != "trajectory_ids"},
                )
                _prepare_open_loop_trt(config)
                result = OssOpenLoopEvaluator(config).run()
                if Path(result.artifacts.model_path).resolve() != Path(checkpoint).resolve():
                    raise ValueError("Open-loop evaluator used a different checkpoint")
                if (
                    Path(result.artifacts.dataset_path).resolve()
                    != Path(self.args.dataset_path).resolve()
                ):
                    raise ValueError("Open-loop evaluator used a different dataset")
                results, metrics = _evaluation_results(result.results)
                labels = {"local"} if settings["local_only"] else {"local", "server"}
                if set(results) != labels:
                    raise ValueError(
                        "Open-loop evaluation did not return the requested runtime results"
                    )
                evaluation["results"] = results
                observed.update(metrics)
                report_path = Path(self.args.output_json)
            else:
                config = _closed_loop_config(settings, checkpoint=checkpoint, args=self.args)
                try:
                    report_path = ClosedLoopEvaluator(config).run()
                except ClosedLoopEvalError as exc:
                    evaluation["report"] = bounded_artifact_identity(
                        exc.report_path, report_root=self.output_dir
                    )
                    raise
                result = json.loads(report_path.read_text(encoding="utf-8"))
                if result["status"] != "completed" or result["model"]["checkpoint"] != str(
                    checkpoint
                ):
                    raise ValueError(
                        "Closed-loop report does not describe the completed trained-checkpoint evaluation"
                    )
                summary = result["summary"]
                if not summary["episodes"] or _raw_number(summary["success_rate"]) is None:
                    raise ValueError(
                        "Closed-loop evaluation returned missing or non-finite results"
                    )
                evaluation["results"] = summary
                observed["evaluation.closed_loop.success_rate"] = summary["success_rate"]
            evaluation["checkpoint"] = self.report["training"]["checkpoint"]
            evaluation["report"] = bounded_artifact_identity(
                report_path, report_root=self.output_dir
            )
        observed["training.completed_steps"] = self.report["training"]["completed_steps"]
        observed["training.final_loss"] = _raw_number(self.report["training"]["final_loss"])
        checks = _checks(self.spec, observed=observed)
        self.report["checks"] = checks
        self.report["status"] = (
            ("passed" if self.spec.expectations else "completed")
            if _checks_pass(checks)
            else "failed"
        )
        return FAILED_EXIT_CODE if self.report["status"] == "failed" else PASSED_EXIT_CODE


def _closed_loop_config(
    settings: Mapping[str, Any], *, checkpoint: Path, args: argparse.Namespace
) -> ClosedLoopEvalConfig:
    return ClosedLoopEvalConfig(
        policy=ClosedLoopPolicyConfig(
            checkpoint=str(checkpoint),
            options={
                "embodiment_tag": args.embodiment_tag,
                "action_steps": settings["action_steps"],
                "denoising_steps": settings["denoising_steps"],
                "trt_engine_dir": args.trt_engine_dir or None,
                "enable_trt": args.enable_trt,
                "trt_cache_dir": args.trt_cache_dir or None,
                "rebuild_trt": args.rebuild_trt,
                "use_cuda_graph": args.use_cuda_graph,
            },
        ),
        output_dir=Path(args.output_dir) / "closed-loop",
        **{
            key: value
            for key, value in settings.items()
            if key not in {"action_steps", "denoising_steps"}
        },
    )


def _prepare_open_loop_trt(config: OssOpenLoopEvalConfig) -> None:
    if not config.enable_trt:
        return
    from groot.core.policy.n2_codec import resolve_n2_codec_roundtrip
    from groot.deployment.n2_trt.resolver import resolve_n2_trt_bundle, resolve_n2_trt_device

    config.device = resolve_n2_trt_device(config.device)
    model_path = OssOpenLoopEvaluator(config).resolve_model_path()
    config.checkpoint = str(model_path)
    config.trt_engine_dir = str(
        resolve_n2_trt_bundle(
            checkpoint=model_path,
            embodiment=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            cache_root=config.trt_cache_dir,
            rebuild=config.rebuild_trt,
            device=config.device,
            codec=resolve_n2_codec_roundtrip(),
        )
    )


def run_benchmark(
    args: argparse.Namespace,
    *,
    run_train_eval: Callable[..., int],
) -> int:
    """Attach a recipe and durable report to the shared train-eval sequence."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    report = _empty_report(args, started_at=started_at)
    should_write_report = True
    benchmark = BenchmarkRun(args, report)

    try:
        source = git_identity(_repository_root())
        report["source"] = {
            "commit": source["sha"],
            "dirty": source["dirty"],
        }
        if source["dirty"]:
            raise ValueError(
                "Benchmark requires a clean source tree; commit changes or use a clean checkout"
            )

        benchmark.stage = "recipe"
        spec = load_benchmark_spec(args.benchmark_recipe)
        benchmark.spec = spec
        _apply_recipe(spec, args=args, output_dir=output_dir)
        _validate_trt_recipe(spec, args=args)
        _populate_spec_report(report, spec=spec, args=args)
        if not args.dry_run:
            benchmark.stage = "dataset"
            report["inputs"]["dataset"] = resolve_dataset(
                spec.dataset, recipe_path=spec.path, destination=Path(args.dataset_path)
            )
        exit_code = run_train_eval(args, benchmark=benchmark)
        if report["status"] == "error" and exit_code != 0:
            raise RuntimeError(f"OSS train-eval exited with status {exit_code}")
        should_write_report = not args.dry_run
    except Exception as exc:  # noqa: BLE001 - report every execution failure.
        report["status"] = "error"
        report["error"] = {
            "stage": benchmark.stage,
            "type": type(exc).__name__,
            "message": sanitize_text(str(exc)),
        }
        exit_code = ERROR_EXIT_CODE
    finally:
        if should_write_report:
            ended_at = datetime.now(timezone.utc)
            report["execution"]["ended_at"] = ended_at.isoformat()
            report["execution"]["duration_seconds"] = round(
                time.monotonic() - started_clock,
                6,
            )
            write_benchmark_report(report_path, report)
            _print_summary(report, report_path=report_path)

    return exit_code


def _empty_report(args: argparse.Namespace, *, started_at: datetime) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": {
            "recipe_id": sanitize_text(args.benchmark_recipe),
            "recipe_version": None,
            "spec_digest": None,
            "provenance": None,
        },
        "status": "error",
        "source": {"commit": None, "dirty": None},
        "inputs": {
            "dataset": None,
            "base_checkpoint": None,
            "resolved_training_configuration": None,
        },
        "execution": {
            "command": sanitize_command(_invocation(args)),
            "mode": "train-eval",
            "seed": None,
            "started_at": started_at.isoformat(),
            "ended_at": None,
            "duration_seconds": None,
            "environment": None,
        },
        "training": {
            "mode": "train",
            "configured_steps": None,
            "completed_steps": None,
            "final_loss": None,
            "checkpoint": None,
        },
        "evaluation": {},
        "expectations": [],
        "checks": [],
        "artifacts": {},
        "error": None,
    }


def _populate_spec_report(
    report: dict[str, Any],
    *,
    spec: BenchmarkSpec,
    args: argparse.Namespace,
) -> None:
    report["benchmark"] = {
        "recipe_id": spec.benchmark["recipe_id"],
        "recipe_version": spec.benchmark["recipe_version"],
        "spec_digest": spec.digest,
        "provenance": sanitize_text(spec.benchmark["provenance"]),
    }
    report["inputs"]["dataset"] = {**spec.dataset, "verified": False}
    report["inputs"]["base_checkpoint"] = {
        "uri": sanitize_text(spec.base_checkpoint["uri"]),
        "revision": spec.base_checkpoint["revision"],
    }
    report["execution"]["seed"] = spec.training["seed"]
    report["training"]["configured_steps"] = spec.training["max_steps"]
    report["evaluation"] = {
        mode: {"settings": settings, "results": None} for mode, settings in spec.evaluation.items()
    }
    report["expectations"] = [
        _expectation_payload(metric, bounds) for metric, bounds in sorted(spec.expectations.items())
    ]
    report["artifacts"]["recipe_spec"] = bounded_artifact_identity(
        spec.path,
        report_root=Path(args.output_dir),
    )


def _apply_recipe(
    spec: BenchmarkSpec,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    provided = {
        token.split("=", maxsplit=1)[0] for token in args.invocation if token.startswith("--")
    }
    unsupported = provided - {
        "--benchmark-recipe",
        "--output-dir",
        "--output-json",
        "--dry-run",
        "--trt-engine-dir",
        "--enable-trt",
        "--trt-cache-dir",
        "--rebuild-trt",
        "--use-cuda-graph",
    }
    if unsupported:
        raise ValueError(
            f"Recipe owns experiment settings; remove: {', '.join(sorted(unsupported))}"
        )
    settings = {
        "experiment": spec.training["experiment"],
        "compute_type": spec.training["compute_type"],
        "experiment_dir": DEFAULT_EXPERIMENT_DIR,
        "embodiment_tag": spec.training["embodiment_tag"],
        "pretrain_checkpoint": spec.base_checkpoint["uri"],
        "max_steps": spec.training["max_steps"],
        "num_gpus": spec.training["num_gpus"],
        "batch_size": spec.training["batch_size"],
        "config_overrides": list(spec.training["config_overrides"]),
        "dataset_path": str(output_dir / "dataset"),
    }
    for name, value in settings.items():
        setattr(args, name, value)
    args.config_overrides.append(f"data.seed={spec.training['seed']}")
    if "--output-json" not in provided:
        args.output_json = str(output_dir / "open-loop-eval.json")
    evaluation_path = Path(args.output_json).expanduser().resolve()
    if evaluation_path in {
        output_dir / REPORT_FILENAME,
        output_dir / "resolved-training-config.yaml",
    } or evaluation_path.is_relative_to(output_dir / "dataset"):
        raise ValueError("--output-json must not overwrite benchmark metadata")
    if "--output-json" in provided and "open_loop" not in spec.evaluation:
        raise ValueError("--output-json requires open_loop evaluation")


def _validate_trt_recipe(spec: BenchmarkSpec, *, args: argparse.Namespace) -> None:
    """Reject incompatible recipe-owned settings before dataset or training work."""
    if not args.enable_trt:
        return
    from groot.deployment.n2_trt.resolver import SUPPORTED_DENOISING_STEPS

    incompatible = {
        mode: settings["denoising_steps"]
        for mode, settings in spec.evaluation.items()
        if settings["denoising_steps"] != SUPPORTED_DENOISING_STEPS
    }
    if incompatible:
        configured = ", ".join(f"{mode}={steps}" for mode, steps in sorted(incompatible.items()))
        raise ValueError(
            "--enable-trt currently requires benchmark recipe denoising_steps="
            f"{SUPPORTED_DENOISING_STEPS}; got {configured}"
        )


def _dry_run_plan(
    report: Mapping[str, Any],
    *,
    resolved: ResolvedTrain,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    training = {
        "command": sanitize_command(resolved.command),
        "resolved_configuration": bounded_artifact_identity(
            config_path,
            report_root=output_dir,
        ),
        "final_checkpoint": safe_path(resolved.final_checkpoint, report_root=output_dir),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": report["benchmark"],
        "source": report["source"],
        "mode": report["execution"]["mode"],
        "dataset": report["inputs"]["dataset"],
        "base_checkpoint": report["inputs"]["base_checkpoint"],
        "training": training,
        "evaluation": report["evaluation"],
        "output_report": REPORT_FILENAME,
        "note": (
            "Dry run only: no training, evaluation, GPU inspection, or benchmark report was run."
        ),
    }


def _resolved_config_path(command: list[str]) -> Path:
    for option in ("--load_config_path", "--load-config-path"):
        if option in command:
            index = command.index(option)
            if index + 1 >= len(command):
                break
            data_config = Path(command[index + 1]).expanduser()
            path = data_config.with_name("resolved_config.yaml")
            if not path.is_file():
                raise FileNotFoundError(f"Resolved training config is missing: {path}")
            return path
    raise ValueError("Resolved OSS training command has no --load_config_path")


def _training_metrics(path: Path) -> tuple[int | None, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Trainer state must be a JSON object: {path}")
    global_step = payload.get("global_step")
    completed_steps = (
        global_step if isinstance(global_step, int) and not isinstance(global_step, bool) else None
    )
    history = payload.get("log_history")
    final_loss: float | None = None
    if isinstance(history, list):
        for row in reversed(history):
            if not isinstance(row, Mapping):
                continue
            for key in ("train_loss", "loss"):
                value = row.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    final_loss = float(value)
                    break
            if final_loss is not None:
                break
    return completed_steps, final_loss


def _checkpoint_metadata_identity(path: Path, *, report_root: Path) -> dict[str, Any]:
    identity = checkpoint_identity(path)
    identity["path"] = safe_path(path, report_root=report_root)
    return identity


def _evaluation_results(
    results: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered: dict[str, Any] = {}
    observed: dict[str, Any] = {}
    for label in sorted(results):
        summary = results[label]
        averages = getattr(summary, "average", None)
        if not isinstance(averages, Mapping):
            raise TypeError(f"Evaluation result {label!r} has no average metrics")
        rendered_average = {
            metric: _json_number(value) for metric, value in sorted(averages.items())
        }
        rendered[label] = {"average": rendered_average}
        for metric, value in averages.items():
            observed[f"evaluation.open_loop.{label}.{metric}"] = value
    return rendered, observed


def _checks(
    spec: BenchmarkSpec,
    *,
    observed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        _range_check(
            metric="training.completed_steps",
            observed=observed.get("training.completed_steps"),
            minimum=float(spec.training["max_steps"]),
            maximum=float(spec.training["max_steps"]),
            provenance="benchmark recipe training.max_steps",
        )
    ]
    for metric in sorted(required_metrics(spec.evaluation)):
        bounds = spec.expectations.get(
            metric,
            {
                "minimum": None,
                "maximum": None,
                "provenance": "Required finite result; acceptance range not specified",
            },
        )
        checks.append(
            _range_check(
                metric=metric,
                observed=observed.get(metric),
                minimum=bounds["minimum"],
                maximum=bounds["maximum"],
                provenance=bounds["provenance"],
            )
        )
    return sorted(checks, key=lambda item: item["metric"])


def _range_check(
    *,
    metric: str,
    observed: Any,
    minimum: float | None,
    maximum: float | None,
    provenance: str,
) -> dict[str, Any]:
    number = _raw_number(observed)
    reason = None
    if number is None:
        passed = False
        reason = "missing or non-finite metric"
    else:
        passed = (minimum is None or number >= minimum) and (maximum is None or number <= maximum)
        if not passed:
            reason = "outside approved range"
    return {
        "metric": metric,
        "observed": _json_number(number),
        "minimum": minimum,
        "maximum": maximum,
        "passed": passed,
        "applicable": True,
        "provenance": sanitize_text(provenance),
        "reason": reason,
    }


def _checks_pass(checks: list[Mapping[str, Any]]) -> bool:
    applicable = [check for check in checks if check.get("applicable")]
    return bool(applicable) and all(check.get("passed") is True for check in applicable)


def _expectation_payload(
    metric: str,
    bounds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "metric": metric,
        "minimum": bounds["minimum"],
        "maximum": bounds["maximum"],
        "provenance": sanitize_text(bounds["provenance"]),
    }


def _execution_environment() -> dict[str, Any]:
    pytorch_version = None
    cuda_version = None
    try:
        torch = importlib.import_module("torch")
        pytorch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
    except ModuleNotFoundError:
        pass

    gpus: list[dict[str, str]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",", maxsplit=2)]
                if len(fields) == 3:
                    gpus.append(dict(zip(("name", "uuid", "driver_version"), fields, strict=True)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "gpus": gpus,
        "cuda": cuda_version,
        "python": platform.python_version(),
        "pytorch": pytorch_version,
        "platform": platform.platform(),
    }


def _invocation(args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "groot.infra.oss", *args.invocation]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _raw_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_number(value: Any) -> int | float | None:
    number = _raw_number(value)
    if number is None:
        return None
    if isinstance(value, int):
        return value
    return number


def _print_summary(report: Mapping[str, Any], *, report_path: Path) -> None:
    status = report["status"]
    recipe_id = report["benchmark"]["recipe_id"]
    sys.stdout.write(f"Benchmark {recipe_id}: {status}\n")
    if status == "failed":
        for check in report["checks"]:
            if check.get("applicable") and check.get("passed") is not True:
                sys.stdout.write(
                    f"FAILED {check['metric']}: observed={check['observed']} "
                    f"range=[{check['minimum']}, {check['maximum']}]\n"
                )
    if report["error"] is not None:
        sys.stdout.write(f"ERROR {report['error']['stage']}: {report['error']['message']}\n")
    sys.stdout.write(f"Report: {report_path.name}\n")
