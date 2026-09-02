# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from groot.core.data import schema
from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator
from groot.infra.launcher.one_launcher.experiment_ray_train import ExperimentRayTrainCommand
from groot.infra.launcher.one_launcher.training_experiment_resolver import (
    DEFAULT_EXPERIMENT_DIR,
    TrainingExperimentResolver,
)

OSS_TRAINING_EXPERIMENT = "n2_oss_pretrain"
OSS_COSMOS_MODEL_URI = (
    "hf://model/nvidia/Cosmos-Reason1-7B@375e24000b24baed78f4618d3dd779e47cd96323"
)
OSS_DIFFUSION_TRANSFORMER_URI = (
    "hf://model/nvidia/Cosmos-Predict2.5-2B@0d37c7498f54cee3c599d438d895a0a4a8608064/transformer"
)
OSS_VAE_URI = (
    "hf://model/Wan-AI/Wan2.1-I2V-14B-480P@6b73f84e66371cdfe870c72acd6826e1d61cf279/Wan2.1_VAE.pth"
)
OSS_PRETRAIN_CHECKPOINT_URI = (
    "hf://model/nvidia/GR00T2Internal@4fe662abaf302ee46916881c447377da8c4bd15a/base_200k"
)


# TODO(cuonguyen): Consolidate this OSS train path with `uv ol train` once the  # noqa: TD003
# one-launcher CLI can run compute-platform-independent local training from the
# OSS package.
@dataclass(frozen=True)
class ResolvedOssTrain:
    experiment: str
    compute_type: str
    command: list[str]
    env: dict[str, str]
    output_dir: Path
    final_checkpoint: Path


@dataclass(frozen=True)
class OssPolicyServerConfig:
    checkpoint: str
    cache_root: str | None
    host: str
    port: int
    embodiment_tag: str
    device: str
    denoising_steps: int


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
    if args.command == "policy-server":
        return _run_policy_server(_policy_server_config(args), dry_run=args.dry_run)
    if args.command == "closed-loop-eval":
        return _run_closed_loop_eval(args)
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
    # Local OSS workers must reuse the complete environment that launched Ray.
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
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
        _emit(shlex.join(resolved.command))
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
        _emit(f"checkpoint={config.checkpoint or ''}")
        _emit(f"dataset_path={config.dataset_path or ''}")
        _emit(f"output_json={config.output_json}")
        _emit(f"traj_ids={','.join(str(item) for item in config.traj_ids)}")
        return 0
    OssOpenLoopEvaluator(config).run()
    return 0


def _run_closed_loop_eval(args: argparse.Namespace) -> int:
    from groot.infra.closed_loop.cli import config_from_args, run

    return run(config_from_args(args), dry_run=args.dry_run)


def _run_policy_server(config: OssPolicyServerConfig, *, dry_run: bool) -> int:
    if dry_run:
        _emit(f"checkpoint={config.checkpoint}")
        _emit(f"cache_root={config.cache_root or ''}")
        _emit(f"address=tcp://{config.host}:{config.port}")
        _emit(f"embodiment_tag={config.embodiment_tag}")
        _emit(f"device={config.device}")
        _emit(f"denoising_steps={config.denoising_steps}")
        return 0

    artifact_config = OssOpenLoopEvalConfig(
        checkpoint=config.checkpoint,
        cache_root=config.cache_root,
    )
    model_path = OssOpenLoopEvaluator(artifact_config).resolve_model_path()
    _run_gr00t_policy_server(model_path, config)
    return 0


def _run_gr00t_policy_server(model_path: str | Path, config: OssPolicyServerConfig) -> None:
    from groot.core.policy import run_gr00t_server

    run_gr00t_server.main(
        run_gr00t_server.ServerConfig(
            model_path=str(model_path),
            embodiment_tag=_server_embodiment_tag(config.embodiment_tag),
            host=config.host,
            port=config.port,
            device=config.device,
            num_inference_timesteps=config.denoising_steps,
        )
    )


def _server_embodiment_tag(embodiment_tag: str) -> schema.EmbodimentTag:
    try:
        return schema.EmbodimentTag[embodiment_tag]
    except KeyError:
        return schema.EmbodimentTag(embodiment_tag)


def _emit(line: str) -> None:
    sys.stdout.write(f"{line}\n")


def _oss_train_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    user_overrides: list[str] = []
    if args.dataset_path:
        public_dataset = [
            {
                "dataset_paths": [args.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": args.embodiment_tag,
            }
        ]
        user_overrides.append(f"data.datasets={json.dumps(public_dataset)}")
    elif not _has_dataset_path_override(args.config_overrides):
        raise SystemExit(
            "--dataset-path is required for OSS training unless the complete "
            "data.datasets list is provided with --set."
        )
    user_overrides.extend(args.config_overrides)
    defaults = {
        "training.upload_checkpoints": "false",
        "training.upload_every": "null",
        "training.msc_profile_url": "''",
        "training.experiment_tracking": "disabled",
        "training.model.tokenizer_path": f"'{OSS_COSMOS_MODEL_URI}'",
        "training.model.diffusion_model_cfg.diffusion_model_pretrained_path": (
            f"'{OSS_DIFFUSION_TRANSFORMER_URI}'"
        ),
        "training.model.text_encoder_cfg.text_encoder_pretrained_path": (
            f"'{OSS_COSMOS_MODEL_URI}'"
        ),
        "training.model.vae_cfg.vae_pretrained_path": f"'{OSS_VAE_URI}'",
        "training.start_from_checkpoint": f"'{OSS_PRETRAIN_CHECKPOINT_URI}'",
    }
    overrides = [f"{key}={value}" for key, value in defaults.items()]
    if not os.environ.get("WANDB_API_KEY"):
        overrides.append("training.use_wandb=false")
    overrides.extend(user_overrides)
    if args.max_steps is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    if args.num_gpus is not None:
        overrides.append(f"compute.num_gpus={args.num_gpus}")
    if args.batch_size is not None:
        overrides.append(f"compute.per_device_batch_size={args.batch_size}")
    if args.output_dir:
        overrides.append(f"training.output_dir={args.output_dir}")
    if args.pretrain_checkpoint:
        overrides.append(f"training.start_from_checkpoint='{args.pretrain_checkpoint}'")
    return tuple(overrides)


def _override_key(override: str) -> str:
    return override.split("=", maxsplit=1)[0].strip()


def _has_dataset_path_override(overrides: Sequence[str]) -> bool:
    return any(_override_key(override) == "data.datasets" for override in overrides)


def _eval_config(
    args: argparse.Namespace, *, checkpoint: str | None = None
) -> OssOpenLoopEvalConfig:
    checkpoint = checkpoint or args.checkpoint
    if not checkpoint:
        raise SystemExit("--checkpoint is required for eval")
    return OssOpenLoopEvalConfig(
        checkpoint=checkpoint,
        dataset_path=args.dataset_path,
        output_json=args.output_json,
        traj_ids=args.traj_ids,
        steps=args.steps,
        action_horizon=args.action_horizon,
        video_backend=args.video_backend,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        server_parity_tolerance=args.server_parity_tolerance,
        local_only=args.local_only,
    )


def _policy_server_config(args: argparse.Namespace) -> OssPolicyServerConfig:
    return OssPolicyServerConfig(
        checkpoint=args.checkpoint,
        cache_root=args.cache_root or None,
        host=args.host,
        port=args.port,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        denoising_steps=args.denoising_steps,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m groot.infra.oss")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers.add_parser("train", help="Run OSS post-training locally."))
    _add_eval_parser(subparsers.add_parser("eval", help="Run OSS open-loop evaluation."))
    _add_policy_server_parser(
        subparsers.add_parser("policy-server", help="Run an OSS GR00T policy server.")
    )
    from groot.infra.closed_loop.cli import add_arguments

    add_arguments(
        subparsers.add_parser(
            "closed-loop-eval",
            help="Run a simulator backend locally or on an existing Ray cluster.",
        )
    )
    train_eval = subparsers.add_parser(
        "train-eval",
        help="Run OSS post-training and then open-loop evaluation.",
    )
    _add_train_parser(train_eval)
    _add_eval_options(
        train_eval,
        checkpoint_required=False,
        include_dataset_path=False,
        include_embodiment_tag=False,
    )
    return parser.parse_args(argv)


def _add_train_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(experiment=OSS_TRAINING_EXPERIMENT)
    parser.add_argument("--compute-type", default="h100")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Local LeRobot dataset path to use for OSS training.",
    )
    parser.add_argument(
        "--embodiment-tag",
        required=True,
        help="Dataset embodiment tag used for data loading and N2 action/state head selection.",
    )
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
    parser.set_defaults(experiment=OSS_TRAINING_EXPERIMENT)
    _add_eval_options(parser, checkpoint_required=True)
    parser.add_argument("--dry-run", action="store_true")


def _add_policy_server_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--embodiment-tag", default="xdof_relative_eef_relative_joint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--dry-run", action="store_true")


def _add_eval_options(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_required: bool,
    include_dataset_path: bool = True,
    include_embodiment_tag: bool = True,
) -> None:
    parser.add_argument("--checkpoint", required=checkpoint_required, default="")
    if include_dataset_path:
        parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-json", default="/tmp/groot_oss_open_loop_eval.json")  # noqa: S108
    parser.add_argument("--traj-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchvision_av", "torchcodec"),
        default="torchcodec",
    )
    if include_embodiment_tag:
        parser.add_argument("--embodiment-tag", default="xdof_relative_eef_relative_joint")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--server-parity-tolerance", type=float, default=0.01)
    parser.add_argument("--local-only", action="store_true")


def _job_name(experiment: str, compute_type: str) -> str:
    slug = experiment.lower().replace("_", "-")
    return f"oss-{slug}-{compute_type}"


if __name__ == "__main__":
    raise SystemExit(main())
