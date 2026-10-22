# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the public ``groot.infra.oss`` entrypoint."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest
import tomllib
import yaml

from groot.infra import oss


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, current_device: int) -> None:
    """Provide the CUDA subset used by the device resolver without requiring PyTorch."""

    class FakeDevice:
        def __init__(self, value: str) -> None:
            device_type, separator, index = value.partition(":")
            self.type = device_type
            self.index = int(index) if separator else None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=FakeDevice,
            cuda=SimpleNamespace(current_device=lambda: current_device),
        ),
    )


def _install_fake_ray(
    monkeypatch: pytest.MonkeyPatch, *, initialized: bool = False
) -> list[dict[str, object]]:
    """Capture the OSS job runtime without starting a local Ray cluster."""
    init_calls: list[dict[str, object]] = []

    def init(**kwargs: object) -> None:
        nonlocal initialized
        initialized = True
        init_calls.append(kwargs)

    def shutdown() -> None:
        nonlocal initialized
        initialized = False

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=init,
            is_initialized=lambda: initialized,
            shutdown=shutdown,
        ),
    )
    return init_calls


def test_oss_cli_does_not_eagerly_import_optional_policy_server_runtime() -> None:
    assert not hasattr(oss, "run_gr00t_server")


def test_oss_build_trt_infers_single_embodiment_and_prints_bundle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "modality_configs": {"test_embodiment": {"action": {}, "state": {}}}
                }
            }
        )
    )
    bundle = tmp_path / "trt-cache" / "bundle"
    builds: list[dict[str, object]] = []
    monkeypatch.setattr(
        oss,
        "_resolve_or_build_trt",
        lambda **kwargs: builds.append(kwargs) or bundle,
    )

    assert (
        oss.main(
            [
                "build-trt",
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cuda:1",
                "--codec",
                "h264",
                "--trt-cache-dir",
                str(tmp_path / "trt-cache"),
                "--rebuild-trt",
            ]
        )
        == 0
    )

    assert builds == [
        {
            "checkpoint": checkpoint,
            "embodiment": "test_embodiment",
            "denoising_steps": 4,
            "cache_dir": str(tmp_path / "trt-cache"),
            "rebuild": True,
            "device": "cuda:1",
            "codec": "h264",
        }
    ]
    assert capsys.readouterr().out == f"TRT_ENGINE_DIR={bundle}\n"


def test_oss_build_trt_requires_selection_for_multiple_embodiments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "modality_configs": {
                        "embodiment_a": {},
                        "embodiment_b": {},
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError, match="pass --embodiment-tag"):
        oss.main(["build-trt", "--checkpoint", str(checkpoint)])


def test_oss_build_trt_accepts_explicit_embodiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    bundle = tmp_path / "bundle"
    builds: list[dict[str, object]] = []
    monkeypatch.setattr(
        oss,
        "_resolve_or_build_trt",
        lambda **kwargs: builds.append(kwargs) or bundle,
    )

    assert (
        oss.main(
            [
                "build-trt",
                "--checkpoint",
                str(checkpoint),
                "--embodiment-tag",
                "selected_embodiment",
            ]
        )
        == 0
    )
    assert builds[0]["embodiment"] == "selected_embodiment"


@pytest.mark.parametrize(
    "argv",
    [
        ["train-eval"],
        ["train-eval", "--benchmark-recipe", "recipe.json", "--dataset-path", "data"],
        ["benchmark"],
        [
            "train-eval",
            "--benchmark-recipe",
            "recipe.json",
            "--output-dir",
            "output",
            "--success-step-budgets",
            "100",
        ],
    ],
)
def test_oss_train_eval_rejects_incomplete_inputs(argv):
    with pytest.raises(SystemExit) as exc:
        oss.main(argv)
    assert exc.value.code == 2


def test_oss_train_eval_benchmark_recipe_delegates_to_report_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[object] = []

    def fake_run(args: object, *, run_train_eval) -> int:
        assert run_train_eval is oss._run_train_eval  # noqa: SLF001
        captured.append(args)
        return 7

    monkeypatch.setattr(oss, "run_benchmark", fake_run)

    status = oss.main(
        [
            "train-eval",
            "--benchmark-recipe",
            str(tmp_path / "approved-spec.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert status == 7
    assert len(captured) == 1
    args = captured[0]
    assert args.command == "train-eval"
    assert args.benchmark_recipe == str(tmp_path / "approved-spec.json")
    assert args.output_dir == str(tmp_path / "output")
    assert args.dry_run is True


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], []),
        (["--success-step-budgets", "100", "200"], [100, 200]),
        (["--max-episode-steps", "400"], [100, 200, 300, 400]),
        (["--max-episode-steps", "400", "--success-step-budgets", ".25", "1.0"], [100, 400]),
    ],
)
def test_oss_closed_loop_dry_run_uses_generic_backend(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, arguments, expected
) -> None:
    engine_dir = tmp_path / "engines"
    status = oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "checkpoint",
            "--backend",
            "custom.backend:Backend",
            "--suite",
            "suite-a",
            "--embodiment-tag",
            "custom_tag",
            "--trt-engine-dir",
            str(engine_dir),
            "--use-cuda-graph",
            "--output-dir",
            str(tmp_path),
            *arguments,
            "--dry-run",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "backend=custom.backend:Backend" in output
    assert f"trt_engine_dir={engine_dir}" in output
    assert "use_cuda_graph=True" in output
    if expected:
        assert f"success_step_budgets={expected}" in output
    else:
        assert "success_step_budgets=" not in output


def test_oss_closed_loop_dry_run_reports_automatic_trt_settings(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    trt_cache = tmp_path / "trt-cache"

    status = oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "checkpoint",
            "--backend",
            "custom.backend:Backend",
            "--suite",
            "suite-a",
            "--embodiment-tag",
            "custom_tag",
            "--enable-trt",
            "--trt-cache-dir",
            str(trt_cache),
            "--rebuild-trt",
            "--dry-run",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "trt_engine_dir=\n" in output
    assert "enable_trt=True" in output
    assert f"trt_cache_dir={trt_cache}" in output
    assert "rebuild_trt=True" in output
    assert "success_step_budgets=" not in output


def test_oss_closed_loop_codec_reaches_worker_when_ray_is_preinitialized(
    monkeypatch, tmp_path: Path
) -> None:
    from groot.infra.closed_loop import cli as closed_loop_cli

    configs: list[object] = []
    monkeypatch.delenv(oss.OSS_N2_CODEC_ENV, raising=False)
    ray_inits = _install_fake_ray(monkeypatch, initialized=True)
    monkeypatch.setattr(
        closed_loop_cli,
        "run",
        lambda config, *, dry_run: configs.append(config) or int(dry_run),
    )

    status = oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "hf://model/org/repo@revision/checkpoint",
            "--backend",
            "custom.backend:Backend",
            "--suite",
            "suite-a",
            "--embodiment-tag",
            "test_embodiment",
            "--enable-trt",
            "--codec",
            "h264",
            "--trt-cache-dir",
            str(tmp_path / "trt-cache"),
        ]
    )

    assert status == 0
    assert len(configs) == 1
    assert configs[0].policy.checkpoint == "hf://model/org/repo@revision/checkpoint"
    assert configs[0].policy.options["enable_trt"] is True
    assert configs[0].policy.options["trt_cache_dir"] == str(tmp_path / "trt-cache")
    assert configs[0].policy.options["rebuild_trt"] is False
    assert configs[0].policy.options["codec_roundtrip"] == "h264"
    assert ray_inits == []
    assert oss.OSS_N2_CODEC_ENV not in os.environ
    assert configs[0].max_episode_steps is None
    assert configs[0].resolved_success_step_budgets == ()


def test_oss_closed_loop_reuses_synced_environment_for_local_ray(monkeypatch) -> None:
    from groot.infra.closed_loop import cli as closed_loop_cli

    observed: list[str | None] = []
    monkeypatch.setenv(oss.OSS_RAY_UV_RUN_RUNTIME_ENV, "1")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(
        closed_loop_cli,
        "run",
        lambda _config, *, dry_run: observed.append(os.environ.get(oss.OSS_RAY_UV_RUN_RUNTIME_ENV))
        or int(dry_run),
    )

    status = oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "checkpoint",
            "--backend",
            "custom.backend:Backend",
            "--suite",
            "suite-a",
            "--embodiment-tag",
            "test_embodiment",
        ]
    )

    assert status == 0
    assert observed == ["0"]
    assert os.environ[oss.OSS_RAY_UV_RUN_RUNTIME_ENV] == "1"


def test_oss_closed_loop_preserves_uv_propagation_for_existing_ray(monkeypatch) -> None:
    from groot.infra.closed_loop import cli as closed_loop_cli

    observed: list[str | None] = []
    monkeypatch.setenv(oss.OSS_RAY_UV_RUN_RUNTIME_ENV, "1")
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster:10001")
    monkeypatch.setattr(
        closed_loop_cli,
        "run",
        lambda _config, *, dry_run: observed.append(os.environ.get(oss.OSS_RAY_UV_RUN_RUNTIME_ENV))
        or int(dry_run),
    )

    status = oss.main(
        [
            "closed-loop-eval",
            "--checkpoint",
            "checkpoint",
            "--backend",
            "custom.backend:Backend",
            "--suite",
            "suite-a",
            "--embodiment-tag",
            "test_embodiment",
        ]
    )

    assert status == 0
    assert observed == ["1"]
    assert os.environ[oss.OSS_RAY_UV_RUN_RUNTIME_ENV] == "1"


@pytest.mark.parametrize(
    ("ray_address", "expected_during_evaluation"),
    [(None, "0"), ("ray://cluster:10001", "1")],
)
def test_oss_train_eval_closed_loop_benchmark_scopes_uv_replay(
    monkeypatch,
    tmp_path: Path,
    ray_address: str | None,
    expected_during_evaluation: str,
) -> None:
    observed: list[str | None] = []
    checkpoint = tmp_path / "checkpoint"

    class FakeBenchmark:
        spec = SimpleNamespace(evaluation={"closed_loop": {}})
        stage = ""

        def record_resolution(self, _resolved: object) -> None:
            pass

        def record_training(self, _resolved: object) -> None:
            pass

        def evaluate(self, evaluated_checkpoint: Path) -> int:
            assert evaluated_checkpoint == checkpoint
            observed.append(os.environ.get(oss.OSS_RAY_UV_RUN_RUNTIME_ENV))
            return 0

    def fake_run_benchmark(args: object, *, run_train_eval) -> int:
        return run_train_eval(args, benchmark=FakeBenchmark())

    monkeypatch.setenv(oss.OSS_RAY_UV_RUN_RUNTIME_ENV, "1")
    if ray_address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", ray_address)
    monkeypatch.setattr(oss, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(oss, "_prepare_train_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        oss,
        "resolve_train",
        lambda _args: SimpleNamespace(final_checkpoint=checkpoint),
    )
    monkeypatch.setattr(oss, "_run_train", lambda *_args, **_kwargs: 0)

    status = oss.main(
        [
            "train-eval",
            "--benchmark-recipe",
            "recipe.json",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert status == 0
    assert observed == [expected_during_evaluation]
    assert os.environ[oss.OSS_RAY_UV_RUN_RUNTIME_ENV] == "1"


def test_oss_public_hf_dependencies_are_commit_pinned() -> None:
    for uri in (
        oss.OSS_COSMOS_MODEL_URI,
        oss.OSS_DIFFUSION_TRANSFORMER_URI,
        oss.OSS_VAE_URI,
        oss.OSS_PRETRAIN_CHECKPOINT_URI,
    ):
        revision = uri.split("@", maxsplit=1)[1].split("/", maxsplit=1)[0]
        assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_oss_train_dry_run_uses_one_launcher(monkeypatch, capsys, tmp_path: Path) -> None:
    _set_training_env(monkeypatch)
    dataset_path = tmp_path / "lerobot_dataset"
    output_dir = tmp_path / "train output"

    status = oss.main(
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--dataset-path",
            str(dataset_path),
            "--compute-type",
            "h100",
            "--max-steps",
            "2",
            "--num-gpus",
            "8",
            "--batch-size",
            "4",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    command = shlex.split(output.splitlines()[0])
    assert status == 0
    assert "python -m groot.core.training.experiment.experiment_ray" in output
    assert f"{oss.OSS_TRAINING_EXPERIMENT}/h100/data.yaml" in output
    assert "--training.no-upload_checkpoints" in output
    assert "--training.no-use_async_checkpoint_save" in output
    assert "--training.num_gpus 8" in output
    assert f"--model.tokenizer_path {oss.OSS_COSMOS_MODEL_URI}" in output
    expected_diffusion_arg = (
        "--model.diffusion-model-cfg.diffusion_model_pretrained_path "
        f"{oss.OSS_DIFFUSION_TRANSFORMER_URI}"
    )
    assert expected_diffusion_arg in output
    assert f"--model.vae-cfg.vae_pretrained_path {oss.OSS_VAE_URI}" in output
    assert f"--training.start_from_checkpoint {oss.OSS_PRETRAIN_CHECKPOINT_URI}" in output
    assert command[command.index("--training.output_dir") + 1] == str(output_dir)
    assert command[command.index("--training.msc_profile_url") + 1] == ""
    assert "FINAL_CHECKPOINT_PATH=" in output


def test_oss_train_user_overrides_follow_defaults_for_last_wins_resolution(
    tmp_path: Path,
) -> None:
    args = oss._parse_args(  # noqa: SLF001
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--set",
            "training.upload-checkpoints=true",
            "--set",
            "training.use-async-checkpoint-save=true",
        ]
    )

    overrides = oss._oss_train_overrides(args)  # noqa: SLF001

    assert overrides.index("training.upload_checkpoints=false") < overrides.index(
        "training.upload-checkpoints=true"
    )
    assert overrides.index("training.use_async_checkpoint_save=false") < overrides.index(
        "training.use-async-checkpoint-save=true"
    )


def test_oss_train_invokes_experiment_ray_without_running_gpu_backend(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    output_dir = tmp_path / "train-output"
    dataset_path = tmp_path / "lerobot_dataset"
    calls: list[dict[str, object]] = []

    def fake_run(
        command: list[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, "env": env, "check": check})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(oss.subprocess, "run", fake_run)

    status = oss.main(
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--compute-type",
            "h100",
            "--dataset-path",
            str(dataset_path),
            "--max-steps",
            "2",
            "--num-gpus",
            "8",
            "--batch-size",
            "4",
            "--output-dir",
            str(output_dir),
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert len(calls) == 1
    assert calls[0]["check"] is True
    command = calls[0]["command"]
    assert isinstance(command, list)
    assert command[:3] == ["python", "-m", "groot.core.training.experiment.experiment_ray"]
    assert "--training.num_gpus" in command
    assert command[command.index("--training.num_gpus") + 1] == "8"
    assert "--training.max_steps" in command
    assert command[command.index("--training.max_steps") + 1] == "2"
    assert "--training.output_dir" in command
    assert command[command.index("--training.output_dir") + 1] == str(output_dir)
    assert "--model.tokenizer_path" in command
    assert command[command.index("--model.tokenizer_path") + 1] == oss.OSS_COSMOS_MODEL_URI
    assert "--model.vae-cfg.vae_pretrained_path" in command
    assert command[command.index("--model.vae-cfg.vae_pretrained_path") + 1] == oss.OSS_VAE_URI
    assert "--training.start_from_checkpoint" in command
    assert (
        command[command.index("--training.start_from_checkpoint") + 1]
        == oss.OSS_PRETRAIN_CHECKPOINT_URI
    )
    assert command[command.index("--model.action-cfg.num-embodiments") + 1] == "32"
    assert "--training.checkpoint_embodiment_slot" not in command
    assert "--training.msc_profile_url" in command
    assert command[command.index("--training.msc_profile_url") + 1] == ""
    _assert_config_contains_dataset(command, dataset_path, "so100_2rgb")
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["NUM_GPUS"] == "8"
    assert env["BATCH_SIZE"] == "4"
    assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert f"FINAL_CHECKPOINT_PATH={output_dir / 'checkpoint-2'}" in output


def test_oss_train_prepares_public_libero_metadata_before_launch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    dataset_path = tmp_path / "libero"
    prepare_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        oss,
        "prepare_lerobot_dataset",
        lambda path, embodiment_tag: prepare_calls.append((path, embodiment_tag)),
    )
    monkeypatch.setattr(oss.subprocess, "run", lambda *_args, **_kwargs: None)

    status = oss.main(
        [
            "train",
            "--experiment",
            "n2_oss_libero_10",
            "--embodiment-tag",
            "libero_sim",
            "--dataset-path",
            str(dataset_path),
            "--max-steps",
            "1",
        ]
    )

    assert status == 0
    assert prepare_calls == [(dataset_path, "libero_sim")]


def test_oss_train_allows_explicit_pretrain_checkpoint_override(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    checkpoint = tmp_path / "checkpoint-override"
    dataset_path = tmp_path / "lerobot_dataset"

    status = oss.main(
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--dataset-path",
            str(dataset_path),
            "--pretrain-checkpoint",
            str(checkpoint),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert f"--training.start_from_checkpoint {checkpoint}" in output


def test_oss_train_preserves_user_model_and_checkpoint_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    tokenizer = tmp_path / "local-tokenizer"
    checkpoint = tmp_path / "local-checkpoint"
    args = oss._parse_args(  # noqa: SLF001 - exercise the public CLI parser.
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--set",
            f"training.model.tokenizer_path='{tokenizer}'",
            "--set",
            f"training.start_from_checkpoint='{checkpoint}'",
            "--set",
            "training.use_wandb=true",
        ]
    )

    command = oss.resolve_train(args).command

    assert command[command.index("--model.tokenizer_path") + 1] == str(tokenizer)
    assert command[command.index("--training.start_from_checkpoint") + 1] == str(checkpoint)
    assert "--training.use_wandb" in command
    assert "--training.no-use_wandb" not in command


def test_oss_train_allows_experiment_dir_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom_experiment_dir = tmp_path / "experiments"
    captured: dict[str, object] = {}

    class FakeResolvedExperiment:
        env_vars: ClassVar[dict[str, str]] = {}
        batch_size = 4
        num_gpus = 8
        training_section: ClassVar[dict[str, int]] = {"max_steps": 2}

    class FakeResolver:
        def resolve(self, **kwargs):
            captured["resolve"] = kwargs
            return FakeResolvedExperiment()

    class FakeTrainCommand:
        def build_command(self) -> list[str]:
            return ["python", "-m", "fake_train"]

    def fake_init_default(*, experiment_dir: Path):
        captured["resolver_experiment_dir"] = experiment_dir
        return FakeResolver()

    def fake_init_from_experiment(**kwargs):
        captured["command"] = kwargs
        return FakeTrainCommand()

    monkeypatch.setattr(
        oss.TrainingExperimentResolver,
        "init_default",
        staticmethod(fake_init_default),
    )
    monkeypatch.setattr(
        oss.ExperimentRayTrainCommand,
        "init_from_experiment",
        staticmethod(fake_init_from_experiment),
    )

    args = oss._parse_args(  # noqa: SLF001 - exercise the public CLI parser.
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--experiment-dir",
            str(custom_experiment_dir),
            "--dataset-path",
            str(tmp_path / "lerobot_dataset"),
        ]
    )

    resolved = oss.resolve_train(args)

    assert captured["resolver_experiment_dir"] == custom_experiment_dir
    assert captured["command"]["experiment_dir"] == custom_experiment_dir
    assert resolved.command == ["python", "-m", "fake_train"]


def test_oss_train_requires_dataset_path_or_dataset_override(monkeypatch) -> None:
    _set_training_env(monkeypatch)

    with pytest.raises(SystemExit, match="--dataset-path is required"):
        oss.main(["train", "--embodiment-tag", "so100_2rgb", "--dry-run"])


def test_oss_train_requires_embodiment_tag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oss.main(
            [
                "train",
                "--dataset-path",
                str(tmp_path / "dataset"),
                "--dry-run",
            ]
        )


def test_oss_train_accepts_documented_dataset_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    dataset_path = tmp_path / "lerobot_dataset"
    datasets = [
        {
            "dataset_paths": [str(dataset_path)],
            "embodiment_tag": "so100_2rgb",
        }
    ]

    args = oss._parse_args(  # noqa: SLF001 - exercise the public CLI parser.
        [
            "train",
            "--embodiment-tag",
            "so100_2rgb",
            "--set",
            f"data.datasets={json.dumps(datasets)}",
            "--dry-run",
        ]
    )

    resolved = oss.resolve_train(args)

    _assert_config_contains_dataset(resolved.command, dataset_path, "so100_2rgb")


def test_oss_train_rejects_experiment_positional() -> None:
    with pytest.raises(SystemExit):
        oss.main(["train", "n2_robocasa365", "--dry-run"])


def test_onboarding_skill_uses_supported_cli_shape() -> None:
    skill_root = _package_root() / "skills/onboarding"
    skill = "\n".join(path.read_text(encoding="utf-8") for path in sorted(skill_root.rglob("*.md")))

    for line in skill.splitlines():
        if not line.startswith("uv run ") or "python -m groot.infra.oss" not in line:
            continue
        command = shlex.split(line.removesuffix("\\").strip())
        subcommand_index = command.index("groot.infra.oss") + 1
        if len(command) > subcommand_index + 1:
            assert command[subcommand_index + 1].startswith("--"), line


def test_oss_agent_guidance_references_existing_surfaces() -> None:
    package_root = _package_root()
    guidance = (package_root / "AGENTS.md").read_text(encoding="utf-8")

    assert "skills/evaluation" not in guidance
    assert "skills/post-train" not in guidance
    for target in re.findall(r"]\(([^)#]+)\)", guidance):
        assert (package_root / target).exists(), target


def test_oss_eval_dry_run_uses_open_loop_eval_config(capsys, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    engine_dir = tmp_path / "engines"
    status = oss.main(
        [
            "eval",
            "--embodiment-tag",
            "so100_2rgb",
            "--checkpoint",
            "hf://model/org/repo@main/checkpoint-100",
            "--dataset-path",
            str(dataset_path),
            "--trt-engine-dir",
            str(engine_dir),
            "--use-cuda-graph",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "checkpoint=hf://model/org/repo@main/checkpoint-100" in output
    assert f"dataset_path={dataset_path}" in output
    assert "output_json=/tmp/groot_oss_open_loop_eval.json" in output
    assert f"trt_engine_dir={engine_dir}" in output
    assert "use_cuda_graph=True" in output


def test_oss_eval_codec_default_ignores_preexisting_runtime_state(
    monkeypatch, tmp_path: Path
) -> None:
    observed: list[str | None] = []

    class FakeEvaluator:
        def __init__(self, _config: oss.OssOpenLoopEvalConfig) -> None:
            pass

        def run(self) -> None:
            observed.append(os.environ.get(oss.OSS_N2_CODEC_ENV))

    monkeypatch.setenv(oss.OSS_N2_CODEC_ENV, "h264")
    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)

    status = oss.main(
        [
            "eval",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
        ]
    )

    assert status == 0
    assert observed == ["none"]
    assert os.environ[oss.OSS_N2_CODEC_ENV] == "h264"


def test_oss_eval_invokes_open_loop_evaluator_without_running_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROOT_N2_NUM_VIDEO_STEPS", "1")
    checkpoint = tmp_path / "checkpoint-2"
    dataset_path = tmp_path / "dataset"
    output_json = tmp_path / "eval.json"
    engine_dir = tmp_path / "engines"
    configs: list[oss.OssOpenLoopEvalConfig] = []

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            configs.append(config)

        def run(self) -> None:
            return None

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)

    status = oss.main(
        [
            "eval",
            "--embodiment-tag",
            "so100_2rgb",
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(dataset_path),
            "--traj-ids",
            "0",
            "3",
            "--steps",
            "2",
            "--action-horizon",
            "2",
            "--denoising-steps",
            "1",
            "--server-parity-tolerance",
            "0.2",
            "--output-json",
            str(output_json),
            "--local-only",
            "--trt-engine-dir",
            str(engine_dir),
            "--use-cuda-graph",
        ]
    )

    assert status == 0
    assert len(configs) == 1
    config = configs[0]
    assert config.checkpoint == str(checkpoint)
    assert config.dataset_path == str(dataset_path)
    assert config.output_json == str(output_json)
    assert config.traj_ids == [0, 3]
    assert config.steps == 2
    assert config.action_horizon == 2
    assert config.denoising_steps == 1
    assert config.server_parity_tolerance == 0.2
    assert config.local_only is True
    assert config.trt_engine_dir == str(engine_dir)
    assert config.use_cuda_graph is True


def test_oss_eval_enable_trt_resolves_checkpoint_and_builds_cached_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    engine_dir = tmp_path / "trt-cache" / "bundle"
    engine_dir.mkdir(parents=True)
    configs: list[oss.OssOpenLoopEvalConfig] = []
    builds: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            self.config = config
            configs.append(config)

        def resolve_model_path(self) -> Path:
            return checkpoint

        def run(self) -> None:
            return None

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    _install_fake_torch(monkeypatch, current_device=0)
    monkeypatch.setattr(
        oss,
        "_resolve_or_build_trt",
        lambda **kwargs: builds.append(kwargs) or engine_dir,
    )

    status = oss.main(
        [
            "eval",
            "--checkpoint",
            "hf://model/org/repo@revision/checkpoint",
            "--embodiment-tag",
            "test_embodiment",
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--enable-trt",
            "--codec",
            "h264",
            "--trt-cache-dir",
            str(tmp_path / "trt-cache"),
            "--rebuild-trt",
        ]
    )

    assert status == 0
    assert builds == [
        {
            "checkpoint": checkpoint,
            "embodiment": "test_embodiment",
            "denoising_steps": 4,
            "cache_dir": str(tmp_path / "trt-cache"),
            "rebuild": True,
            "device": "cuda:0",
            "codec": "h264",
        }
    ]
    assert configs[-1].checkpoint == str(checkpoint)
    assert configs[-1].device == "cuda:0"
    assert configs[-1].trt_engine_dir == str(engine_dir)


def test_oss_eval_enable_trt_surfaces_eager_trt_drift(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from groot.deployment.n2_trt import auto_builder, policy_measurement

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    engine_dir = tmp_path / "bundle"
    engine_dir.mkdir()
    receipt = engine_dir / policy_measurement.POLICY_MEASUREMENT_FILENAME
    receipt.write_text(
        json.dumps(
            {
                "status": "complete_with_drift",
                "equivalence": {
                    "action.gripper": {"max_abs": 0.01234567, "rel_rmse": 0.00432109},
                    "action.joint": {"max_abs": 0.25, "rel_rmse": 0.125},
                },
                "exact_equivalence_breaches": ["drift"],
            }
        )
    )
    builds: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(self, _config: oss.OssOpenLoopEvalConfig) -> None:
            pass

        def resolve_model_path(self) -> Path:
            return checkpoint

        def run(self) -> None:
            return None

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    _install_fake_torch(monkeypatch, current_device=0)
    monkeypatch.setattr(
        auto_builder,
        "resolve_or_build_engine_bundle",
        lambda **kwargs: builds.append(kwargs) or engine_dir,
    )

    with caplog.at_level(logging.WARNING, logger=policy_measurement.__name__):
        assert (
            oss.main(
                [
                    "eval",
                    "--checkpoint",
                    str(checkpoint),
                    "--embodiment-tag",
                    "test_embodiment",
                    "--codec",
                    "h264",
                    "--enable-trt",
                ]
            )
            == 0
        )

    warning = caplog.text
    assert "TensorRT decoded-action drift from eager" in warning
    assert "action.gripper: max_abs=0.0123457, rel_rmse=0.00432109" in warning
    assert "action.joint: max_abs=0.25, rel_rmse=0.125" in warning
    assert str(receipt) in warning
    assert builds[0]["codec"] == "h264"


def test_oss_eval_uses_one_explicit_nonzero_device_for_build_and_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from groot.deployment.n2_trt import auto_builder, policy_measurement

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    engine_dir = tmp_path / "bundle"
    engine_dir.mkdir()
    (engine_dir / policy_measurement.POLICY_MEASUREMENT_FILENAME).write_text(
        '{"equivalence": {}, "exact_equivalence_breaches": []}'
    )
    builds: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            self.config = config

        def resolve_model_path(self) -> Path:
            return checkpoint

    _install_fake_torch(monkeypatch, current_device=1)
    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        auto_builder,
        "resolve_or_build_engine_bundle",
        lambda **kwargs: builds.append(kwargs) or engine_dir,
    )
    config = oss.OssOpenLoopEvalConfig(
        checkpoint=str(checkpoint),
        embodiment_tag="test_embodiment",
        enable_trt=True,
        device=None,
    )

    oss._prepare_eval_trt(config)  # noqa: SLF001

    assert config.device == "cuda:1"
    assert config.trt_engine_dir == str(engine_dir)
    assert builds[0]["device"] == "cuda:1"


def test_oss_cli_rejects_incoherent_automatic_trt_options(monkeypatch) -> None:
    monkeypatch.setenv("GROOT_N2_NUM_VIDEO_STEPS", "2")
    with pytest.raises(SystemExit):
        oss.main(
            [
                "eval",
                "--checkpoint",
                "checkpoint",
                "--trt-engine-dir",
                "engines",
                "--enable-trt",
            ]
        )

    with pytest.raises(SystemExit):
        oss.main(["eval", "--checkpoint", "checkpoint", "--rebuild-trt"])

    with pytest.raises(ValueError, match="positive.*exact multiple"):
        oss.main(
            [
                "eval",
                "--checkpoint",
                "checkpoint",
                "--enable-trt",
                "--denoising-steps",
                "3",
                "--dry-run",
            ]
        )


_TRT_CLI_ROUTES = [
    ("build-trt", []),
    *[
        (command, selection)
        for command in ("train-eval", "eval", "policy-server", "closed-loop-eval")
        for selection in (["--enable-trt"], ["--trt-engine-dir", "unused-engines"])
    ],
]


def _schedule_cli_args(command, selection, action_steps):
    arguments = [command, "--embodiment-tag", "libero_sim", *selection]
    if command == "train-eval":
        arguments.extend(["--dataset-path", "unused-dataset"])
    else:
        arguments.extend(["--checkpoint", "unused-checkpoint"])
    if command == "closed-loop-eval":
        arguments.extend(["--backend", "unused:Backend", "--suite", "unused"])
    arguments.extend(["--denoising-steps", str(action_steps)])
    return arguments


@pytest.mark.parametrize(("command", "selection"), _TRT_CLI_ROUTES)
@pytest.mark.parametrize(("video_steps", "action_steps"), [("2", 3), ("0", 4), ("2", 0)])
def test_oss_invalid_trt_schedule_fails_before_work(
    monkeypatch, command, selection, video_steps, action_steps
):
    monkeypatch.setenv("GROOT_N2_NUM_VIDEO_STEPS", video_steps)

    def unexpected(*_args, **_kwargs):
        pytest.fail(
            "invalid TRT schedules must fail before any dataset, training or evaluation work"
        )

    for name in (
        "_prepare_train_dataset",
        "resolve_train",
        "_run_train",
        "_run_eval",
        "_run_build_trt",
        "_run_policy_server",
        "_run_closed_loop_eval",
    ):
        monkeypatch.setattr(oss, name, unexpected)

    with pytest.raises(ValueError, match="positive.*exact multiple"):
        oss.main(_schedule_cli_args(command, selection, action_steps))


@pytest.mark.parametrize(("command", "selection"), _TRT_CLI_ROUTES)
def test_oss_valid_custom_trt_schedule_reaches_dispatch(monkeypatch, command, selection):
    monkeypatch.setenv("GROOT_N2_NUM_VIDEO_STEPS", "3")
    # Explicit CLI steps take precedence over an unused environment default.
    monkeypatch.setenv("GROOT_N2_NUM_ACTION_STEPS", "invalid")
    calls = []
    monkeypatch.setattr(oss, "_dispatch", lambda args: calls.append(args) or 0)

    assert oss.main(_schedule_cli_args(command, selection, 6)) == 0
    assert len(calls) == 1
    assert calls[0].denoising_steps == 6


@pytest.mark.parametrize(
    "selection", [["--enable-trt"], ["--trt-engine-dir", "unused-engines"], []]
)
@pytest.mark.parametrize(("video_steps", "action_steps"), [("1", 1), ("2", 2), ("2", 6), ("3", 6)])
def test_plain_train_eval_valid_custom_schedule_runs_all_stages(
    monkeypatch, tmp_path, selection, video_steps, action_steps
):
    monkeypatch.setenv("GROOT_N2_NUM_VIDEO_STEPS", video_steps)
    calls = []
    checkpoint = tmp_path / "checkpoint"

    def prepare(*_args, **_kwargs):
        calls.append("dataset")

    def resolve(_args):
        calls.append("resolution")
        return SimpleNamespace(final_checkpoint=checkpoint)

    def train(*_args, **_kwargs):
        calls.append("training")
        return 0

    def evaluate(config, **_kwargs):
        calls.append("evaluation")
        assert config.checkpoint == str(checkpoint)
        assert config.denoising_steps == action_steps
        return 0

    monkeypatch.setattr(oss, "_prepare_train_dataset", prepare)
    monkeypatch.setattr(oss, "resolve_train", resolve)
    monkeypatch.setattr(oss, "_run_train", train)
    monkeypatch.setattr(oss, "_run_eval", evaluate)

    assert oss.main(_schedule_cli_args("train-eval", selection, action_steps)) == 0
    assert calls == ["dataset", "resolution", "training", "evaluation"]


def test_oss_policy_server_dry_run_reports_effective_config(capsys, tmp_path: Path) -> None:
    engine_dir = tmp_path / "engines"
    status = oss.main(
        [
            "policy-server",
            "--checkpoint",
            "hf://model/org/repo@main/checkpoint-100",
            "--host",
            "127.0.0.1",
            "--port",
            "6000",
            "--device",
            "cpu",
            "--denoising-steps",
            "2",
            "--trt-engine-dir",
            str(engine_dir),
            "--use-cuda-graph",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "checkpoint=hf://model/org/repo@main/checkpoint-100" in output
    assert "embodiment_tag=xdof_relative_eef_relative_joint" in output
    assert "address=tcp://127.0.0.1:6000" in output
    assert "device=cpu" in output
    assert "denoising_steps=2" in output
    assert f"trt_engine_dir={engine_dir}" in output
    assert "use_cuda_graph=True" in output


def test_policy_server_uses_enum_member_for_irregular_embodiment() -> None:
    server_tag = oss._server_embodiment_tag("onex_eve_gripper")  # noqa: SLF001
    assert server_tag.name == "ONE_X_EVE_GRIPPER"


def test_oss_policy_server_invokes_core_server_in_process_without_running_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    engine_dir = tmp_path / "engines"
    configs: list[oss.OssOpenLoopEvalConfig] = []
    calls: list[tuple[Path, oss.OssPolicyServerConfig]] = []
    codecs: list[str | None] = []

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            configs.append(config)

        def resolve_model_path(self) -> Path:
            return checkpoint

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        oss,
        "_run_gr00t_policy_server",
        lambda model_path, config: (
            calls.append((model_path, config)),
            codecs.append(os.environ.get(oss.OSS_N2_CODEC_ENV)),
        ),
    )

    status = oss.main(
        [
            "policy-server",
            "--checkpoint",
            "hf://model/org/repo@main/checkpoint-100",
            "--host",
            "127.0.0.1",
            "--port",
            "6000",
            "--device",
            "cpu",
            "--denoising-steps",
            "2",
            "--cache-root",
            str(tmp_path / "hf-cache"),
            "--trt-engine-dir",
            str(engine_dir),
            "--codec",
            "mp4v",
            "--use-cuda-graph",
        ]
    )

    assert status == 0
    assert len(configs) == 1
    assert codecs == ["mp4v"]
    assert configs[0].checkpoint == "hf://model/org/repo@main/checkpoint-100"
    assert configs[0].cache_root == str(tmp_path / "hf-cache")

    assert calls == [
        (
            checkpoint,
            oss.OssPolicyServerConfig(
                checkpoint="hf://model/org/repo@main/checkpoint-100",
                cache_root=str(tmp_path / "hf-cache"),
                host="127.0.0.1",
                port=6000,
                embodiment_tag="xdof_relative_eef_relative_joint",
                device="cpu",
                denoising_steps=2,
                trt_engine_dir=str(engine_dir),
                use_cuda_graph=True,
            ),
        )
    ]


def test_oss_policy_server_enable_trt_builds_bundle_before_serving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    engine_dir = tmp_path / "trt-cache" / "bundle"
    engine_dir.mkdir(parents=True)
    calls: list[tuple[Path, oss.OssPolicyServerConfig]] = []
    builds: list[dict[str, object]] = []

    class FakeEvaluator:
        def __init__(self, _config: oss.OssOpenLoopEvalConfig) -> None:
            pass

        def resolve_model_path(self) -> Path:
            return checkpoint

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    _install_fake_torch(monkeypatch, current_device=0)
    monkeypatch.setattr(
        oss,
        "_resolve_or_build_trt",
        lambda **kwargs: builds.append(kwargs) or engine_dir,
    )
    monkeypatch.setattr(
        oss,
        "_run_gr00t_policy_server",
        lambda model_path, config: calls.append((model_path, config)),
    )

    status = oss.main(
        [
            "policy-server",
            "--checkpoint",
            "hf://model/org/repo@revision/checkpoint-100",
            "--embodiment-tag",
            "test_embodiment",
            "--enable-trt",
            "--trt-cache-dir",
            str(tmp_path / "trt-cache"),
        ]
    )

    assert status == 0
    assert builds == [
        {
            "checkpoint": checkpoint,
            "embodiment": "test_embodiment",
            "denoising_steps": 4,
            "cache_dir": str(tmp_path / "trt-cache"),
            "rebuild": False,
            "device": "cuda:0",
            "codec": "none",
        }
    ]
    assert calls[0][0] == checkpoint
    assert calls[0][1].checkpoint == str(checkpoint)
    assert calls[0][1].device == "cuda:0"
    assert calls[0][1].trt_engine_dir == str(engine_dir)


def test_oss_policy_server_bare_cuda_uses_real_resolver_with_current_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from groot.deployment.n2_trt import auto_builder, policy_measurement

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    engine_dir = tmp_path / "bundle"
    engine_dir.mkdir()
    (engine_dir / policy_measurement.POLICY_MEASUREMENT_FILENAME).write_text(
        '{"equivalence": {}, "exact_equivalence_breaches": []}'
    )
    builds: list[dict[str, object]] = []
    calls: list[oss.OssPolicyServerConfig] = []

    class FakeEvaluator:
        def __init__(self, _config: oss.OssOpenLoopEvalConfig) -> None:
            pass

        def resolve_model_path(self) -> Path:
            return checkpoint

    _install_fake_torch(monkeypatch, current_device=1)
    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        auto_builder,
        "resolve_or_build_engine_bundle",
        lambda **kwargs: builds.append(kwargs) or engine_dir,
    )
    monkeypatch.setattr(
        oss,
        "_run_gr00t_policy_server",
        lambda _model_path, config: calls.append(config),
    )

    status = oss.main(
        [
            "policy-server",
            "--checkpoint",
            str(checkpoint),
            "--embodiment-tag",
            "test_embodiment",
            "--enable-trt",
        ]
    )

    assert status == 0
    assert builds[0]["device"] == "cuda:1"
    assert calls[0].device == "cuda:1"
    assert calls[0].trt_engine_dir == str(engine_dir)


def test_oss_policy_server_builds_server_config_at_runtime(monkeypatch, tmp_path: Path) -> None:
    from groot.core import policy as policy_package

    class FakeServerConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    calls: list[FakeServerConfig] = []
    fake_server_module = SimpleNamespace(ServerConfig=FakeServerConfig, main=calls.append)
    monkeypatch.setattr(policy_package, "run_gr00t_server", fake_server_module, raising=False)

    engine_dir = tmp_path / "engines"
    config = oss.OssPolicyServerConfig(
        checkpoint=str(tmp_path / "checkpoint-100"),
        cache_root=None,
        host="127.0.0.1",
        port=6000,
        embodiment_tag="onex_eve_gripper",
        device="cpu",
        denoising_steps=2,
        trt_engine_dir=str(engine_dir),
        use_cuda_graph=True,
    )

    oss._run_gr00t_policy_server(tmp_path / "checkpoint-100", config)  # noqa: SLF001

    assert len(calls) == 1
    server_config = calls[0]
    assert server_config.model_path == str(tmp_path / "checkpoint-100")
    assert server_config.embodiment_tag.name == "ONE_X_EVE_GRIPPER"
    assert server_config.num_inference_timesteps == 2
    assert server_config.trt_engine_dir == str(engine_dir)
    assert server_config.use_cuda_graph is True


def test_oss_train_eval_always_evaluates_new_final_checkpoint_with_hf_vae(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import huggingface_hub

    from groot.deployment.n2_trt import resolve_checkpoint_vae_weights

    _set_training_env(monkeypatch)
    output_dir = tmp_path / "train-output"
    engine_dir = tmp_path / "engines"
    remote_vae = "hf://model/organization/vae@revision/vae.pth"
    eval_configs: list[oss.OssOpenLoopEvalConfig] = []
    builds: list[dict[str, object]] = []
    resolved_vae: list[Path] = []

    def fake_run(
        command: list[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert env["NUM_GPUS"] == "8"
        assert env[oss.OSS_RAY_UV_RUN_RUNTIME_ENV] == "0"
        assert check is True
        checkpoint = output_dir / "checkpoint-2"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(
            json.dumps({"vae_cfg": {"vae_pretrained_path": remote_vae}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    def snapshot_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "organization/vae"
        assert kwargs["revision"] == "revision"
        snapshot = tmp_path / "hf-cache" / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "vae.pth").write_bytes(b"fixture-vae")
        return str(snapshot)

    def resolve_bundle(**kwargs: object) -> Path:
        builds.append(kwargs)
        resolved_vae.append(resolve_checkpoint_vae_weights(kwargs["checkpoint"]))
        return engine_dir

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            self.config = config

        def resolve_model_path(self) -> Path:
            assert self.config.checkpoint is not None
            return Path(self.config.checkpoint)

        def run(self) -> None:
            eval_configs.append(self.config)

    monkeypatch.setattr(oss.subprocess, "run", fake_run)
    _install_fake_torch(monkeypatch, current_device=0)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    monkeypatch.setattr(oss, "_resolve_or_build_trt", resolve_bundle)

    status = oss.main(
        [
            "train-eval",
            "--embodiment-tag",
            "so100_2rgb",
            "--compute-type",
            "h100",
            "--max-steps",
            "2",
            "--num-gpus",
            "8",
            "--batch-size",
            "4",
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(tmp_path / "old-checkpoint"),
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--traj-ids",
            "0",
            "--steps",
            "2",
            "--enable-trt",
            "--trt-cache-dir",
            str(tmp_path / "trt-cache"),
            "--use-cuda-graph",
        ]
    )

    assert status == 0
    assert builds == [
        {
            "checkpoint": output_dir / "checkpoint-2",
            "embodiment": "so100_2rgb",
            "denoising_steps": 4,
            "cache_dir": str(tmp_path / "trt-cache"),
            "rebuild": False,
            "device": "cuda:0",
            "codec": "none",
        }
    ]
    assert resolved_vae == [(tmp_path / "hf-cache" / "snapshot" / "vae.pth").resolve()]
    assert len(eval_configs) == 1
    assert eval_configs[0].checkpoint == str(output_dir / "checkpoint-2")
    assert eval_configs[0].device == "cuda:0"
    assert eval_configs[0].trt_engine_dir == str(engine_dir)
    assert eval_configs[0].use_cuda_graph is True
    assert not (output_dir / "benchmark-report.json").exists()


def test_oss_infra_surface_exposes_only_temporary_data_credentials() -> None:
    package_root = _package_root()

    # GEAR workdir-upload policy is internal launch infrastructure, not part of
    # the OSS runtime or packaging surface.
    assert not list(package_root.rglob(".gearignore"))

    exported_files = [
        package_root / "groot/infra/oss.py",
        package_root / "groot/infra/dataset_preparation.py",
        package_root / "groot/infra/job/data_credentials.py",
    ]
    exported_directories = [
        package_root / "groot/infra/benchmark",
        package_root / "groot/infra/launcher",
    ]
    eval_dir = package_root / "groot/infra/eval"
    assert (package_root / "groot/infra/experiments").is_dir()
    public_configs = package_root / "groot/infra/experiments/training/configs"
    exported_files.append(public_configs / f"{oss.OSS_TRAINING_EXPERIMENT}.py")
    exported_files.extend(
        public_configs / name
        for name in (
            "n2_oss_libero_10.py",
            "n2_oss_libero_goal.py",
            "n2_oss_libero_object.py",
            "n2_oss_libero_spatial.py",
        )
    )

    for exported_file in exported_files:
        assert exported_file.is_file()
    for exported_directory in exported_directories:
        assert exported_directory.is_dir()
    assert eval_dir.is_dir()

    task_config_dir = eval_dir / "configs/tasks"
    approved_prefixes = ("libero_pro_", "robocasa365_", "robodojo_")
    exported_task_configs = sorted(task_config_dir.glob("*.yaml"))
    assert exported_task_configs
    assert all(path.name.startswith(approved_prefixes) for path in exported_task_configs)
    assert not (public_configs / "n2_egoscale_mega_soup_pretrain.py").exists()
    assert not (public_configs / "n2_yam_simulation_stack_bowls.py").exists()
    assert not (public_configs / "n2_robocasa365.py").exists()
    job_dir = package_root / "groot/infra/job"
    assert (job_dir / "__init__.py").is_file()
    assert not (job_dir / "__main__.py").exists()
    assert not (job_dir / "ray_cluster.py").exists()
    assert not (job_dir / "ray_job.py").exists()
    assert not (package_root / "groot/infra/ci").exists()


def test_oss_tensorrt_runtime_and_tools_are_packaged() -> None:
    package_root = _package_root()
    runtime_dir = package_root / "groot/deployment/n2_trt"
    engine_tools = runtime_dir / "tools"
    deployment_files = (
        package_root / "scripts/deployment/README.md",
        package_root / "scripts/deployment/thor/activate_thor.sh",
        package_root / "scripts/deployment/thor/install_deps.sh",
        package_root / "scripts/deployment/thor/pyproject.toml",
        package_root / "scripts/deployment/thor/uv.lock",
    )

    for filename in (
        "benchmark_adapter.py",
        "contract.py",
        "installer.py",
        "manifest.py",
        "options.py",
        "runtime.py",
        "validation.py",
    ):
        assert (runtime_dir / filename).is_file()
    for filename in (
        "__init__.py",
        "build_engines.py",
        "export_transformer.py",
        "export_vae.py",
    ):
        assert (engine_tools / filename).is_file()
    assert not (package_root / "scripts/deployment/n2").exists()
    assert all(path.is_file() for path in deployment_files)


def _lock_packages() -> dict[tuple[str, str], str]:
    text = (_package_root() / "uv.lock").read_text(encoding="utf-8")
    blocks = text.split("\n[[package]]\n")
    packages: dict[tuple[str, str], str] = {}
    for block in blocks:
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name is None or version is None:
            continue
        packages[(name.group(1), version.group(1))] = block
    return packages


def _pyproject_pin(name: str, *, aarch64: bool | None = None) -> str:
    """Return the version pyproject.toml pins for ``name`` on Linux aarch64, elsewhere, or both."""
    text = (_package_root() / "pyproject.toml").read_text(encoding="utf-8")
    versions = {
        version
        for version, marker in re.findall(rf'"{re.escape(name)}==([^ ";]+)(?: ; ([^"]*))?"', text)
        if aarch64 is None or ("platform_machine == 'aarch64'" in marker) == aarch64
    }
    assert len(versions) == 1, f"{name} pins {versions}"
    return versions.pop()


def _has_linux_aarch64_wheels(block: str) -> bool:
    return (
        "platform_machine == 'aarch64'" in block
        and "sys_platform == 'linux'" in block
        and any(
            tag in block
            for tag in (
                "manylinux_2_27_aarch64",
                "manylinux_2_28_aarch64",
                "manylinux_2_35_aarch64",
            )
        )
    )


def test_locked_training_stack_supports_linux_aarch64_and_keeps_x86_cu128() -> None:
    """Catch an x86-only pin regressing the GB300 lock, and the reverse."""
    packages = _lock_packages()

    arm_torch_version = _pyproject_pin("torch", aarch64=True)
    arm_torch = packages[("torch", arm_torch_version)]
    assert _has_linux_aarch64_wheels(arm_torch)
    assert "x86_64" not in arm_torch
    for name in ("torchvision", "torchcodec"):
        assert _has_linux_aarch64_wheels(packages[(name, _pyproject_pin(name, aarch64=True))])
    # triton is not pinned; the aarch64 torch wheel pulls it in.
    assert any(
        _has_linux_aarch64_wheels(block)
        for (name, _), block in packages.items()
        if name == "triton"
    )
    arm_trt = packages[("tensorrt-cu13", _pyproject_pin("tensorrt-cu13", aarch64=True))]
    assert "platform_machine == 'aarch64' and sys_platform == 'linux'" in arm_trt

    x86_torch_version = f"{_pyproject_pin('torch', aarch64=False)}+cu128"
    x86_torch = packages[("torch", x86_torch_version)]
    assert "platform_machine != 'aarch64'" in x86_torch
    assert "manylinux_2_28_x86_64" in x86_torch
    assert "platform_machine == 'aarch64' and sys_platform == 'linux'" not in x86_torch
    # flash-attn installs on both architectures, against each one's torch.
    flash = packages[("flash-attn", _pyproject_pin("flash-attn"))]
    assert f'version = "{x86_torch_version}"' in flash
    assert f'version = "{arm_torch_version}"' in flash

    x86_nvrx = packages[
        ("nvidia-resiliency-ext", _pyproject_pin("nvidia-resiliency-ext", aarch64=False))
    ]
    arm_nvrx = packages[
        ("nvidia-resiliency-ext", _pyproject_pin("nvidia-resiliency-ext", aarch64=True))
    ]
    assert "manylinux_2_31_x86_64" in x86_nvrx
    assert "manylinux_2_39_aarch64" in arm_nvrx
    # Training installs NVRx by default; deployment, and so the Thor image, does not.
    extras = tomllib.loads((_package_root() / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]
    assert sum(dep.startswith("nvidia-resiliency-ext==") for dep in extras["training"]) == 2
    assert not any(dep.startswith("nvidia-resiliency-ext") for dep in extras["deployment"])
    thor_lock = tomllib.loads(
        (_package_root() / "scripts/deployment/thor/uv.lock").read_text(encoding="utf-8")
    )
    assert "nvidia-resiliency-ext" not in {package["name"] for package in thor_lock["package"]}


def test_oss_gpu_ci_job_runs_gpu_marked_package_tests() -> None:
    ci_config = (_package_root() / "ci/test.oss.gitlab-ci.yml").read_text(encoding="utf-8")
    workflow_config = (_package_root() / ".gitlab-ci.yml").read_text(encoding="utf-8")
    gpu_job = _job_block(ci_config, ".oss_gpu_job")
    l40s_job = _job_block(ci_config, "test.oss.gpu.l40s")
    gb300_job = _job_block(ci_config, "test.oss.gpu.gb300")

    assert '$CI_PIPELINE_SOURCE == "merge_request_event"' in workflow_config
    assert '$OSS_TESTS == "true"' not in workflow_config
    assert "IPP_RUNNER_GPU_TYPE: NVIDIA-L40S" in l40s_job
    assert "- ipp-group-gear" in l40s_job
    assert "IPP_RUNNER_MEMORY_LIMIT: 64Gi" in l40s_job
    assert "- gpu/gb300" in gb300_job
    for job, arch in ((l40s_job, "amd64"), (gb300_job, "arm64")):
        assert "extends: .oss_gpu_job" in job
        assert f"- image.oss.{arch}\n" in job
        assert f"groot-ci-oss:${{CI_COMMIT_SHORT_SHA}}-{arch}\n" in job
    assert "HF_HUB_OFFLINE" not in gpu_job
    assert "GROOT_HF_ARTIFACT_CACHE_ROOT" not in gpu_job
    assert "OSS_REBUILD_PYAV_FROM_SOURCE" not in ci_config
    assert "OSS_REBUILD_OPENCV_FROM_SOURCE" not in ci_config
    assert "--export-oss-for-tests" in ci_config
    assert "unset PYTHONPATH" in ci_config
    assert "groot imported outside OSS export" in ci_config
    assert '"$OSS_PACKAGE_ROOT/tests/groot"' in gpu_job
    assert "-m gpu" in gpu_job
    assert "--durations=0" in gpu_job
    assert "--tb=short" in gpu_job
    assert "-vv" in gpu_job
    assert "-ra" in gpu_job
    assert "uv sync --frozen --inexact --extra training --extra dev --extra tensorrt" in gpu_job
    assert "uv pip install" not in gpu_job
    assert 'GR00T_DOCKER_IMAGE_TESTS: "1"' in gpu_job
    assert '"$OSS_PACKAGE_ROOT/tests/test_dockerfile.py"' in gpu_job
    assert "python - <<" not in gpu_job
    assert "--no-install-package" not in ci_config
    assert "--system-site-packages" not in ci_config
    # The standalone OSS repository keeps its lock at the root.
    assert "\n        - uv.lock\n" in _job_block(ci_config, "test.oss.cpu")
    assert '-m "not gpu"' in ci_config


def test_oss_cpu_ci_job_runs_the_dockerfile_tests_in_the_x86_image() -> None:
    """Catch the image checks in tests/test_dockerfile.py silently not running in OSS CI."""
    ci_config = (_package_root() / "ci/test.oss.gitlab-ci.yml").read_text(encoding="utf-8")
    cpu_job = _job_block(ci_config, "test.oss.cpu")

    assert "- image.oss.amd64\n" in cpu_job
    assert "groot-ci-oss:${CI_COMMIT_SHORT_SHA}-amd64\n" in cpu_job
    assert 'GR00T_DOCKER_IMAGE_TESTS: "1"' in cpu_job
    pytest_call = cpu_job[cpu_job.index("python -m pytest") :]
    assert '"$OSS_PACKAGE_ROOT/tests/test_dockerfile.py"' in pytest_call
    assert "uv run python -" not in cpu_job


def test_oss_ci_images_build_the_user_dockerfile_with_its_defaults() -> None:
    """Catch the OSS test images drifting from what `docker/build.sh` builds."""
    image_ci = (_package_root() / "ci/image.oss.gitlab-ci.yml").read_text(encoding="utf-8")
    build = _job_block(image_ci, ".image.oss")

    assert "--dockerfile docker/Dockerfile" in build
    assert "--build-arg" not in image_ci
    assert "--export-oss-for-tests" in build
    assert '--platform "linux/${OSS_IMAGE_ARCH}"' in build
    for arch in ("amd64", "arm64"):
        assert f"OSS_IMAGE_ARCH: {arch}" in _job_block(image_ci, f"image.oss.{arch}")


def test_oss_test_job_change_triggers_are_covered_by_their_image_jobs() -> None:
    """Catch a test job selected by a change whose required image job is not in the pipeline."""
    jobs = {}
    for path in sorted((_package_root() / "ci").glob("*.gitlab-ci.yml")):
        jobs.update(yaml.load(path.read_text(encoding="utf-8"), Loader=_GitLabCiLoader) or {})

    for name in (
        "test.oss.cpu",
        "test.oss.gpu.l40s",
        "test.oss.gpu.gb300",
        "build.oss.dependencies",
    ):
        test_paths, _ = _merge_request_change_paths(jobs, name)
        for need in _inherited(jobs, name, "needs"):
            image_paths, always = _merge_request_change_paths(jobs, need)
            if not always:
                assert test_paths <= image_paths, (name, need, sorted(test_paths - image_paths))


def test_oss_ci_variable_trigger_does_not_force_image_jobs() -> None:
    package_root = _package_root()
    oss_image_ci = (package_root / "ci/image.oss.gitlab-ci.yml").read_text(encoding="utf-8")
    _assert_job_does_not_run_for_oss_tests(oss_image_ci, ".image.oss")

    monorepo_root = _monorepo_root()
    internal_base_minimal_ci = (
        monorepo_root / "ci/internal/image.base.minimal.gitlab-ci.yml"
    ).read_text(encoding="utf-8")
    internal_minimal_ci = (monorepo_root / "ci/internal/image.minimal.gitlab-ci.yml").read_text(
        encoding="utf-8"
    )
    internal_train_ci = (monorepo_root / "ci/internal/image.train.gitlab-ci.yml").read_text(
        encoding="utf-8"
    )

    _assert_job_does_not_run_for_oss_tests(internal_base_minimal_ci, "image.base.minimal")
    _assert_job_does_not_run_for_oss_tests(internal_minimal_ci, "image.minimal")
    _assert_job_does_not_run_for_oss_tests(internal_train_ci, "image.base.train.arm64")
    _assert_job_does_not_run_for_oss_tests(internal_train_ci, "image.train.arm64")
    _assert_job_does_not_run_for_oss_tests(internal_train_ci, "image.train.amd64")
    _assert_job_does_not_run_for_oss_tests(internal_train_ci, "image.train")


def _set_training_env(monkeypatch) -> None:
    monkeypatch.setenv("USER", "tester")


def _package_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )


def _monorepo_root() -> Path:
    root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "ci/internal/image.train.gitlab-ci.yml").exists()
        ),
        None,
    )
    if root is None:
        pytest.skip("internal CI config assertions only apply in the monorepo")
    return root


def _assert_job_does_not_run_for_oss_tests(ci_config: str, job_name: str) -> None:
    assert "- if: '$OSS_TESTS == \"true\"'" not in _job_block(ci_config, job_name)


def _assert_config_contains_dataset(
    command: list[str], dataset_path: Path, embodiment_tag: str
) -> None:
    assert "--data.datasets.0.dataset_paths" not in command
    assert "--load_config_path" in command
    data_config = Path(command[command.index("--load_config_path") + 1])
    body = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    assert body["data"]["datasets"][0]["dataset_paths"] == [str(dataset_path)]
    assert body["data"]["datasets"][0]["embodiment_tag"] == embodiment_tag


class _GitLabCiLoader(yaml.SafeLoader):
    pass


_GitLabCiLoader.add_constructor("!reference", lambda loader, node: loader.construct_sequence(node))


def _merge_request_change_paths(jobs: dict, job_name: str) -> tuple[set[str], bool]:
    """Return a job's merge-request `changes` paths, and whether it runs on every MR."""
    rules = _inherited(jobs, job_name, "rules")
    if rules is None:
        return set(), True
    paths: set[str] = set()
    for rule in rules:
        if "merge_request_event" not in rule.get("if", ""):
            continue
        if "changes" not in rule:
            return set(), True
        changes = rule["changes"]
        paths.update(changes["paths"] if isinstance(changes, dict) else changes)
    return paths, False


def _inherited(jobs: dict, job_name: str, key: str):
    """Return a job's `key`, following `extends` the way GitLab merges it (last parent wins)."""
    job = jobs[job_name]
    if key in job:
        return job[key]
    extends = job.get("extends", [])
    for parent in reversed([extends] if isinstance(extends, str) else extends):
        value = _inherited(jobs, parent, key)
        if value is not None:
            return value
    return None


def _job_block(ci_config: str, job_name: str) -> str:
    job_match = re.search(
        rf"(?ms)^{re.escape(job_name)}:\n.*?(?=^[A-Za-z0-9_.-]+:|\Z)",
        ci_config,
    )
    assert job_match is not None
    return job_match.group(0)
