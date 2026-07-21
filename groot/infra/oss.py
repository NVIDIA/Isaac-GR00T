# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSS training and open-loop evaluation entrypoint.

The OSS CLI is intentionally the public launch surface for package users and CI
smoke tests. Training resolves a one-launcher experiment config, applies
OSS-safe overrides, and executes the underlying
``groot.core.training.experiment.experiment_ray`` command. Evaluation runs the
OSS open-loop evaluator against either a local checkpoint path or an HF model
URI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import getpass
import os
from pathlib import Path
import subprocess
import sys

from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator
from groot.infra.launcher.one_launcher.experiment_ray_train import ExperimentRayTrainCommand
from groot.infra.launcher.one_launcher.training_experiment_resolver import (
    DEFAULT_EXPERIMENT_DIR,
    TrainingExperimentResolver,
)


# TODO(cuonguyen): Consolidate this OSS train path with `uv ol train` once the  # noqa: TD003
# one-launcher CLI can run compute-platform-independent local training from the
# OSS package. See https://gitlab-master.nvidia.com/GR00T/gr00t/gr00t/-/merge_requests/4377#note_60682123
@dataclass(frozen=True)
class ResolvedOssTrain:
    experiment: str
    compute_type: str
    command: list[str]
    env: dict[str, str]
    output_dir: Path
    final_checkpoint: Path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "train":
        resolved = resolve_train(args)
        return _run_train(resolved, dry_run=args.dry_run)
    if args.command == "eval":
        config = _eval_config(args)
        return _run_eval(config, dry_run=args.dry_run)
    if args.command == "train-eval":
        resolved = resolve_train(args)
        train_status = _run_train(resolved, dry_run=args.dry_run)
        if train_status != 0 or args.dry_run:
            return train_status
        checkpoint = args.checkpoint or str(resolved.final_checkpoint)
        return _run_eval(_eval_config(args, checkpoint=checkpoint), dry_run=False)
    raise AssertionError(f"unhandled command {args.command!r}")


def resolve_train(args: argparse.Namespace) -> ResolvedOssTrain:
    experiment = args.experiment
    compute_type = args.compute_type
    overrides = _oss_train_overrides(args)
    resolver = TrainingExperimentResolver.init_default(experiment_dir=args.experiment_dir)
    resolved = resolver.resolve(
        name=experiment,
        compute_type=compute_type,
        config_overrides=overrides,
    )
    env = dict(os.environ)
    env.update(resolved.env_vars)
    env.setdefault("JOB_NAME", _job_name(experiment, compute_type))
    env.setdefault("USER_SLUG", getpass.getuser())
    env["BATCH_SIZE"] = str(resolved.batch_size)
    env["NUM_GPUS"] = str(resolved.num_gpus)
    env.setdefault("TRAINING_OUTPUT_ROOT", "/tmp/groot_outputs")  # noqa: S108
    env.setdefault("GROOT_CHECKPOINT_ROOT", os.environ.get("GROOT_CHECKPOINT_ROOT", ""))

    command = ExperimentRayTrainCommand.init_from_experiment(
        experiment=experiment,
        compute_type=compute_type,
        config_overrides=overrides,
        experiment_dir=args.experiment_dir,
        env=env,
    )
    output_dir = Path(env["TRAINING_OUTPUT_ROOT"]) / env["USER_SLUG"] / env["JOB_NAME"]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    max_steps = int(resolved.training_section.get("max_steps", 0))
    final_checkpoint = output_dir / f"checkpoint-{max_steps}"
    return ResolvedOssTrain(
        experiment=experiment,
        compute_type=compute_type,
        command=command.build_command(),
        env=env,
        output_dir=output_dir,
        final_checkpoint=final_checkpoint,
    )


def _run_train(resolved: ResolvedOssTrain, *, dry_run: bool) -> int:
    if dry_run:
        _emit(" ".join(resolved.command))
        _emit(f"JOB_NAME={resolved.env['JOB_NAME']}")
        _emit(f"USER_SLUG={resolved.env['USER_SLUG']}")
        _emit(f"BATCH_SIZE={resolved.env['BATCH_SIZE']}")
        _emit(f"NUM_GPUS={resolved.env['NUM_GPUS']}")
        _emit(f"OUTPUT_DIR={resolved.output_dir}")
        _emit(f"FINAL_CHECKPOINT_PATH={resolved.final_checkpoint}")
        return 0
    subprocess.run(resolved.command, env=resolved.env, check=True)
    _emit(f"FINAL_CHECKPOINT_PATH={resolved.final_checkpoint}")
    return 0


def _run_eval(config: OssOpenLoopEvalConfig, *, dry_run: bool) -> int:
    if dry_run:
        _emit(f"model_path={config.model_path or ''}")
        _emit(f"model_uri={config.model_uri or ''}")
        _emit(f"dataset_path={config.dataset_path or ''}")
        _emit(f"output_json={config.output_json}")
        _emit(f"traj_ids={','.join(str(item) for item in config.traj_ids)}")
        return 0
    OssOpenLoopEvaluator(config).run()
    return 0


def _emit(line: str) -> None:
    sys.stdout.write(f"{line}\n")


def _oss_train_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    overrides = list(args.config_overrides)
    overrides.extend(
        [
            "training.upload_checkpoints=false",
            "training.upload_every=null",
            "training.model.tokenizer_path='{COSMOS_MODEL_PATH}'",
            ("training.model.diffusion_model_cfg.diffusion_model_pretrained_path='{DIFFUSION_TRANSFORMER_PATH}'"),
            "training.model.text_encoder_cfg.text_encoder_pretrained_path='{COSMOS_MODEL_PATH}'",
            "training.model.vae_cfg.vae_pretrained_path='{VAE_PATH}'",
        ]
    )
    if not os.environ.get("WANDB_API_KEY"):
        overrides.append("training.use_wandb=false")
    if args.max_steps is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    if args.num_gpus is not None:
        overrides.append(f"compute.num_gpus={args.num_gpus}")
    if args.batch_size is not None:
        overrides.append(f"compute.per_device_batch_size={args.batch_size}")
    if args.output_dir:
        overrides.append(f"training.output_dir={args.output_dir}")
    if args.pretrain_checkpoint:
        overrides.append(f"training.start_from_checkpoint={args.pretrain_checkpoint}")
    elif os.environ.get("PRETRAIN_CHECKPOINT"):
        overrides.append("training.start_from_checkpoint='{PRETRAIN_CHECKPOINT}'")
    return tuple(overrides)


def _eval_config(args: argparse.Namespace, *, checkpoint: str | None = None) -> OssOpenLoopEvalConfig:
    checkpoint = checkpoint or args.checkpoint
    if not checkpoint:
        raise SystemExit("--checkpoint is required for eval")
    model_path = None
    model_uri = None
    if checkpoint.startswith("hf://model/"):
        model_uri = checkpoint
    else:
        model_path = checkpoint
    return OssOpenLoopEvalConfig(
        model_path=model_path,
        model_uri=model_uri,
        dataset_path=args.dataset_path,
        output_json=args.output_json,
        traj_ids=args.traj_ids,
        steps=args.steps,
        action_horizon=args.action_horizon,
        denoising_steps=args.denoising_steps,
        server_parity_tolerance=args.server_parity_tolerance,
        local_only=args.local_only,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m groot.infra.oss")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers.add_parser("train", help="Run OSS post-training locally."))
    _add_eval_parser(subparsers.add_parser("eval", help="Run OSS open-loop evaluation."))
    train_eval = subparsers.add_parser(
        "train-eval",
        help="Run OSS post-training and then open-loop evaluation.",
    )
    _add_train_parser(train_eval)
    _add_eval_options(train_eval, checkpoint_required=False)
    return parser.parse_args(argv)


def _add_train_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", help="One-launcher experiment name.")
    parser.add_argument("--compute-type", default="h100")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--pretrain-checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a one-launcher experiment config value.",
    )
    parser.add_argument("--dry-run", action="store_true")


def _add_eval_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", help="One-launcher experiment name.")
    _add_eval_options(parser, checkpoint_required=True)
    parser.add_argument("--dry-run", action="store_true")


def _add_eval_options(parser: argparse.ArgumentParser, *, checkpoint_required: bool) -> None:
    parser.add_argument("--checkpoint", required=checkpoint_required, default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-json", default="/tmp/groot_oss_open_loop_eval.json")  # noqa: S108
    parser.add_argument("--traj-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-parity-tolerance", type=float, default=0.01)
    parser.add_argument("--local-only", action="store_true")


def _job_name(experiment: str, compute_type: str) -> str:
    slug = experiment.lower().replace("_", "-")
    return f"oss-{slug}-{compute_type}"


if __name__ == "__main__":
    raise SystemExit(main())
