# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Command-line interface for portable closed-loop evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from groot.infra.closed_loop.runner import (
    ClosedLoopEvalConfig,
    ClosedLoopEvalError,
    ClosedLoopEvaluator,
    ClosedLoopPolicyConfig,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add portable closed-loop evaluation arguments."""
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", required=True, help="Backend plugin as module:Class.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--denoising-steps", type=int)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Backend-specific JSON value; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace) -> ClosedLoopEvalConfig:
    """Convert parsed CLI arguments to a portable evaluator configuration."""
    policy_options: dict[str, object] = {
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "action_steps": args.action_steps,
    }
    if args.denoising_steps is not None:
        policy_options["denoising_steps"] = args.denoising_steps
    return ClosedLoopEvalConfig(
        policy=ClosedLoopPolicyConfig(
            checkpoint=args.checkpoint,
            cache_root=args.cache_root or None,
            options=policy_options,
        ),
        backend=args.backend,
        suite=args.suite,
        tasks=tuple(args.tasks),
        episodes_per_task=args.episodes_per_task,
        num_workers=args.num_workers,
        ray_address=args.ray_address or None,
        cpus_per_worker=args.cpus_per_worker,
        gpus_per_worker=args.gpus_per_worker,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        record_video=not args.no_record_video,
        output_dir=args.output_dir,
        backend_options=parse_backend_options(args.backend_option),
    )


def parse_backend_options(values: Sequence[str]) -> dict[str, object]:
    """Parse repeatable ``NAME=JSON`` backend options."""
    options: dict[str, object] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or not name:
            raise ValueError(f"--backend-option must use NAME=JSON, got {value!r}")
        options[name] = json.loads(encoded)
    return options


def run(config: ClosedLoopEvalConfig, *, dry_run: bool) -> int:
    """Run or describe a portable closed-loop evaluation."""
    if dry_run:
        _emit(f"checkpoint={config.policy.checkpoint}")
        _emit(f"backend={config.backend}")
        _emit(f"suite={config.suite}")
        _emit(f"ray_address={config.resolved_ray_address or '<local>'}")
        _emit(f"output_dir={config.output_dir.expanduser().resolve()}")
        return 0
    try:
        report_path = ClosedLoopEvaluator(config).run()
    except ClosedLoopEvalError as error:
        sys.stderr.write(f"closed-loop evaluation failed: {error}\n")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _emit(f"BENCHMARK_SUMMARY={json.dumps(report['summary'], sort_keys=True)}")
    _emit(f"BENCHMARK_REPORT={report_path}")
    return 0


def _emit(value: str) -> None:
    sys.stdout.write(f"{value}\n")


def parser() -> argparse.ArgumentParser:
    """Build the standalone closed-loop parser."""
    result = argparse.ArgumentParser(description="Run portable closed-loop evaluation.")
    add_arguments(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone entrypoint."""
    args = parser().parse_args(argv)
    return run(config_from_args(args), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
