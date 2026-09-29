# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OSS training and evaluation entrypoint.

The OSS CLI is intentionally the public launch surface for package users and CI
smoke tests. Training resolves a one-launcher experiment config, applies
OSS-safe overrides, and executes the underlying
``groot.core.training.experiment.experiment_ray`` command. Evaluation runs the
OSS open-loop evaluator against either a local checkpoint path or an HF model
URI.

Train-eval optionally records a benchmark report under an approved recipe.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groot.infra.benchmark.runner import BenchmarkRun

from groot.core.data import schema
from groot.infra.benchmark.runner import run_benchmark
from groot.infra.dataset_preparation import prepare_lerobot_dataset
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
OSS_N2_CODEC_ENV = "GROOT_N2_CODEC_ROUNDTRIP"
OSS_N2_CODECS = ("none", "h264", "mp4v")


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
    trt_engine_dir: str | None = None
    enable_trt: bool = False
    trt_cache_dir: str | None = None
    rebuild_trt: bool = False
    use_cuda_graph: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if hasattr(args, "codec"):
        with _oss_codec_environment(args.codec) as codec:
            args.codec = codec
            return _dispatch(args)
    return _dispatch(args)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "train":
        _prepare_train_dataset(args, dry_run=args.dry_run)
        resolved = resolve_train(args)
        return _run_train(resolved, dry_run=args.dry_run)
    if args.command == "eval":
        config = _eval_config(args)
        return _run_eval(config, dry_run=args.dry_run)
    if args.command == "train-eval":
        if args.benchmark_recipe:
            return run_benchmark(args, run_train_eval=_run_train_eval)
        return _run_train_eval(args)
    if args.command == "policy-server":
        return _run_policy_server(_policy_server_config(args), dry_run=args.dry_run)
    if args.command == "closed-loop-eval":
        return _run_closed_loop_eval(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def _run_train_eval(args: argparse.Namespace, *, benchmark: BenchmarkRun | None = None) -> int:
    # Benchmark recipes own the codec. Re-scope the OSS process after the
    # recipe has been applied so open-loop policy construction uses that value.
    if benchmark and args.codec != _active_oss_codec():
        with _oss_codec_environment(args.codec):
            return _run_train_eval_sequence(args, benchmark=benchmark)
    return _run_train_eval_sequence(args, benchmark=benchmark)


def _run_train_eval_sequence(
    args: argparse.Namespace, *, benchmark: BenchmarkRun | None = None
) -> int:
    if benchmark:
        benchmark.stage = "dataset"
    _prepare_train_dataset(args, dry_run=args.dry_run)
    if benchmark:
        benchmark.stage = "resolution"
    resolved = resolve_train(args)
    if benchmark:
        benchmark.record_resolution(resolved)
        if args.dry_run:
            return 0
        benchmark.stage = "training"
    train_status = _run_train(resolved, dry_run=args.dry_run)
    if train_status != 0 or args.dry_run:
        if args.dry_run:
            _emit(f"codec={_active_oss_codec()}")
        return train_status
    if benchmark:
        benchmark.record_training(resolved)
        benchmark.stage = "evaluation"
        return benchmark.evaluate(resolved.final_checkpoint)
    return _run_eval(_eval_config(args, checkpoint=str(resolved.final_checkpoint)), dry_run=False)


def _prepare_train_dataset(args: argparse.Namespace, *, dry_run: bool) -> None:
    if dry_run or not args.dataset_path:
        return
    prepare_lerobot_dataset(Path(args.dataset_path), args.embodiment_tag)


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
        # The one-launcher default is cwd-relative, so resolving from a checkout
        # writes generated YAML into it. Anchor it to the run's output root.
        output_root=Path(env["TRAINING_OUTPUT_ROOT"]) / ".ol" / "experiments",
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
        _emit(f"trt_engine_dir={config.trt_engine_dir or ''}")
        _emit(f"enable_trt={config.enable_trt}")
        _emit(f"trt_cache_dir={config.trt_cache_dir or ''}")
        _emit(f"rebuild_trt={config.rebuild_trt}")
        _emit(f"use_cuda_graph={config.use_cuda_graph}")
        _emit(f"codec={_active_oss_codec()}")
        return 0
    _prepare_eval_trt(config)
    OssOpenLoopEvaluator(config).run()
    return 0


def _prepare_eval_trt(config: OssOpenLoopEvalConfig) -> None:
    if not config.enable_trt:
        return
    from groot.core.policy.n2_codec import resolve_n2_codec_roundtrip
    from groot.deployment.n2_trt.resolver import resolve_n2_trt_device

    config.device = resolve_n2_trt_device(config.device)
    model_path = OssOpenLoopEvaluator(config).resolve_model_path()
    config.checkpoint = str(model_path)
    config.trt_engine_dir = str(
        _resolve_or_build_trt(
            checkpoint=model_path,
            embodiment=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            cache_dir=config.trt_cache_dir,
            rebuild=config.rebuild_trt,
            device=config.device,
            codec=resolve_n2_codec_roundtrip(),
        )
    )


def _resolve_or_build_trt(
    *,
    checkpoint: str | Path,
    embodiment: str,
    denoising_steps: int | None,
    cache_dir: str | Path | None,
    rebuild: bool,
    device: str | None,
    codec: str,
) -> Path:
    from groot.deployment.n2_trt.resolver import resolve_n2_trt_bundle

    return resolve_n2_trt_bundle(
        checkpoint=checkpoint,
        embodiment=embodiment,
        denoising_steps=denoising_steps,
        cache_root=cache_dir,
        rebuild=rebuild,
        device=device,
        codec=codec,
    )


def _run_closed_loop_eval(args: argparse.Namespace) -> int:
    from groot.infra.closed_loop.cli import config_from_args, prepare_trt_args, run

    prepare_trt_args(args)
    config = config_from_args(args)
    config = dataclass_replace(
        config,
        policy=dataclass_replace(
            config.policy,
            options={
                **config.policy.options,
                "codec_roundtrip": _active_oss_codec(),
            },
        ),
    )
    if args.dry_run:
        _emit(f"codec={_active_oss_codec()}")
        return run(config, dry_run=True)
    return run(config, dry_run=False)


def _run_policy_server(config: OssPolicyServerConfig, *, dry_run: bool) -> int:
    if dry_run:
        _emit(f"checkpoint={config.checkpoint}")
        _emit(f"cache_root={config.cache_root or ''}")
        _emit(f"address=tcp://{config.host}:{config.port}")
        _emit(f"embodiment_tag={config.embodiment_tag}")
        _emit(f"device={config.device}")
        _emit(f"denoising_steps={config.denoising_steps}")
        _emit(f"trt_engine_dir={config.trt_engine_dir or ''}")
        _emit(f"enable_trt={config.enable_trt}")
        _emit(f"trt_cache_dir={config.trt_cache_dir or ''}")
        _emit(f"rebuild_trt={config.rebuild_trt}")
        _emit(f"use_cuda_graph={config.use_cuda_graph}")
        _emit(f"codec={_active_oss_codec()}")
        return 0

    artifact_config = OssOpenLoopEvalConfig(
        checkpoint=config.checkpoint,
        cache_root=config.cache_root,
    )
    model_path = OssOpenLoopEvaluator(artifact_config).resolve_model_path()
    if config.enable_trt:
        from groot.core.policy.n2_codec import resolve_n2_codec_roundtrip
        from groot.deployment.n2_trt.resolver import resolve_n2_trt_device

        device = resolve_n2_trt_device(config.device)
        config = dataclass_replace(
            config,
            checkpoint=str(model_path),
            device=device,
            trt_engine_dir=str(
                _resolve_or_build_trt(
                    checkpoint=model_path,
                    embodiment=config.embodiment_tag,
                    denoising_steps=config.denoising_steps,
                    cache_dir=config.trt_cache_dir,
                    rebuild=config.rebuild_trt,
                    device=device,
                    codec=resolve_n2_codec_roundtrip(),
                )
            ),
        )
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
            trt_engine_dir=config.trt_engine_dir,
            use_cuda_graph=config.use_cuda_graph,
        )
    )


def _server_embodiment_tag(embodiment_tag: str) -> schema.EmbodimentTag:
    try:
        return schema.EmbodimentTag[embodiment_tag]
    except KeyError:
        return schema.EmbodimentTag(embodiment_tag)


def _emit(line: str) -> None:
    sys.stdout.write(f"{line}\n")


def _active_oss_codec() -> str:
    return os.environ[OSS_N2_CODEC_ENV]


@contextmanager
def _oss_codec_environment(requested: str) -> Iterator[str]:
    """Scope the OSS CLI codec without changing shared policy defaults."""
    previous = os.environ.get(OSS_N2_CODEC_ENV)
    os.environ[OSS_N2_CODEC_ENV] = requested
    try:
        yield requested
    finally:
        if previous is None:
            os.environ.pop(OSS_N2_CODEC_ENV, None)
        else:
            os.environ[OSS_N2_CODEC_ENV] = previous


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
        "training.use_async_checkpoint_save": "false",
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
        trt_engine_dir=args.trt_engine_dir or None,
        enable_trt=args.enable_trt,
        trt_cache_dir=args.trt_cache_dir or None,
        rebuild_trt=args.rebuild_trt,
        use_cuda_graph=args.use_cuda_graph,
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
        trt_engine_dir=args.trt_engine_dir or None,
        enable_trt=args.enable_trt,
        trt_cache_dir=args.trt_cache_dir or None,
        rebuild_trt=args.rebuild_trt,
        use_cuda_graph=args.use_cuda_graph,
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

    closed_loop = subparsers.add_parser(
        "closed-loop-eval",
        help="Run a simulator backend locally or on an existing Ray cluster.",
    )
    add_arguments(closed_loop)
    _add_oss_codec_option(closed_loop)
    train_eval = subparsers.add_parser(
        "train-eval",
        help="Run OSS post-training and then open-loop evaluation.",
        allow_abbrev=False,
    )
    _add_train_parser(train_eval, embodiment_required=False)
    _add_eval_options(
        train_eval,
        checkpoint_required=False,
        include_dataset_path=False,
        include_embodiment_tag=False,
    )
    train_eval.add_argument(
        "--benchmark-recipe",
        default="",
        help="Approved recipe ID or spec JSON fixing run settings and enabling a benchmark report.",
    )
    args = parser.parse_args(argv)
    _validate_acceleration_args(args, parser)
    if args.command == "train-eval":
        if args.benchmark_recipe:
            if not args.output_dir:
                parser.error("--benchmark-recipe requires --output-dir")
            args.invocation = list(sys.argv[1:] if argv is None else argv)
        elif not args.embodiment_tag:
            parser.error("--embodiment-tag is required for train-eval without --benchmark-recipe")
    return args


def _add_train_parser(parser: argparse.ArgumentParser, *, embodiment_required: bool = True) -> None:
    parser.add_argument("--experiment", default=OSS_TRAINING_EXPERIMENT)
    parser.add_argument("--compute-type", default="h100")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Local LeRobot dataset path to use for OSS training.",
    )
    parser.add_argument(
        "--embodiment-tag",
        required=embodiment_required,
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
    _add_oss_codec_option(parser)
    _add_acceleration_options(parser)
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
    _add_oss_codec_option(parser)
    _add_acceleration_options(parser)
    parser.add_argument("--server-parity-tolerance", type=float, default=0.01)
    parser.add_argument("--local-only", action="store_true")


def _add_acceleration_options(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trt-engine-dir", default="")
    selection.add_argument(
        "--enable-trt",
        action="store_true",
        help="Reuse or build and validate a target-local TensorRT engine bundle.",
    )
    parser.add_argument(
        "--trt-cache-dir",
        default="",
        help="Cache root for bundles created by --enable-trt.",
    )
    parser.add_argument(
        "--rebuild-trt",
        action="store_true",
        help="Rebuild the matching cached bundle used by --enable-trt.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
    )


def _add_oss_codec_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--codec",
        choices=OSS_N2_CODECS,
        default="none",
        help="Observation codec roundtrip used by this OSS inference command. Defaults to none.",
    )


def _validate_acceleration_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if getattr(args, "rebuild_trt", False) and not getattr(args, "enable_trt", False):
        parser.error("--rebuild-trt requires --enable-trt")
    if getattr(args, "trt_cache_dir", "") and not getattr(args, "enable_trt", False):
        parser.error("--trt-cache-dir requires --enable-trt")
    if not getattr(args, "enable_trt", False):
        return
    from groot.deployment.n2_trt.resolver import SUPPORTED_DENOISING_STEPS

    if getattr(args, "denoising_steps", SUPPORTED_DENOISING_STEPS) not in (
        None,
        SUPPORTED_DENOISING_STEPS,
    ):
        parser.error(
            f"--enable-trt currently requires --denoising-steps {SUPPORTED_DENOISING_STEPS}"
        )


def _job_name(experiment: str, compute_type: str) -> str:
    slug = experiment.lower().replace("_", "-")
    return f"oss-{slug}-{compute_type}"


if __name__ == "__main__":
    raise SystemExit(main())
