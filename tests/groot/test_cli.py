# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the public OSS CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml

from groot.infra import oss


def test_oss_cli_does_not_eagerly_import_optional_policy_server_runtime() -> None:
    assert not hasattr(oss, "run_gr00t_server")


@pytest.mark.parametrize(
    "argv",
    [
        ["train-eval"],
        ["train-eval", "--benchmark-recipe", "recipe.json", "--dataset-path", "data"],
        ["benchmark"],
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


def test_oss_closed_loop_dry_run_uses_generic_backend(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
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
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert status == 0
    assert "backend=custom.backend:Backend" in capsys.readouterr().out


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
    skill = (_package_root() / "skills/onboarding/SKILL.md").read_text(encoding="utf-8")

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


def test_oss_eval_invokes_open_loop_evaluator_without_running_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
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

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            configs.append(config)

        def resolve_model_path(self) -> Path:
            return checkpoint

    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        oss,
        "_run_gr00t_policy_server",
        lambda model_path, config: calls.append((model_path, config)),
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
            "--use-cuda-graph",
        ]
    )

    assert status == 0
    assert len(configs) == 1
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


def test_oss_train_eval_passes_trained_checkpoint_to_eval_without_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch)
    output_dir = tmp_path / "train-output"
    eval_configs: list[oss.OssOpenLoopEvalConfig] = []

    def fake_run(
        command: list[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert env["NUM_GPUS"] == "8"
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    class FakeEvaluator:
        def __init__(self, config: oss.OssOpenLoopEvalConfig) -> None:
            eval_configs.append(config)

        def run(self) -> None:
            return None

    monkeypatch.setattr(oss.subprocess, "run", fake_run)
    monkeypatch.setattr(oss, "OssOpenLoopEvaluator", FakeEvaluator)

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
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--traj-ids",
            "0",
            "--steps",
            "2",
        ]
    )

    assert status == 0
    assert len(eval_configs) == 1
    assert eval_configs[0].checkpoint == str(output_dir / "checkpoint-2")
    assert not (output_dir / "benchmark-report.json").exists()


def test_oss_infra_surface_exposes_only_temporary_data_credentials() -> None:
    package_root = _package_root()

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
    exported_files.append(public_configs / "n2_oss_libero_10.py")

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


def test_oss_gpu_ci_job_runs_gpu_marked_package_tests() -> None:
    ci_config = (_package_root() / "ci/test.oss.gitlab-ci.yml").read_text(encoding="utf-8")
    workflow_config = (_package_root() / ".gitlab-ci.yml").read_text(encoding="utf-8")
    gpu_job = _job_block(ci_config, "test.oss.gpu")

    assert "test.oss.gpu:" in ci_config
    assert '$CI_PIPELINE_SOURCE == "merge_request_event"' in workflow_config
    assert '$OSS_TESTS == "true"' not in workflow_config
    assert "- image.train" in gpu_job
    assert "groot-ci-train:${CI_COMMIT_SHORT_SHA}" in gpu_job
    assert "-arm64" not in gpu_job
    assert "HF_HUB_OFFLINE" not in gpu_job
    assert "GROOT_HF_ARTIFACT_CACHE_ROOT" not in gpu_job
    assert "OSS_REBUILD_PYAV_FROM_SOURCE" not in ci_config
    assert "OSS_REBUILD_OPENCV_FROM_SOURCE" not in ci_config
    assert "--export-oss-for-tests" in ci_config
    assert "unset PYTHONPATH" in ci_config
    assert "groot imported outside OSS export" in ci_config
    assert '"$OSS_PACKAGE_ROOT/tests/groot"' in gpu_job
    assert "-m gpu" in gpu_job
    assert "tensorrt-cu13==10.16.1.11" in gpu_job
    assert '-m "not gpu"' in ci_config


def test_oss_ci_variable_trigger_does_not_force_image_jobs() -> None:
    package_root = _package_root()
    oss_base_minimal_ci = (package_root / "ci/image.base.minimal.gitlab-ci.yml").read_text(
        encoding="utf-8"
    )
    oss_minimal_ci = (package_root / "ci/image.minimal.gitlab-ci.yml").read_text(encoding="utf-8")
    oss_train_ci = (package_root / "ci/image.train.gitlab-ci.yml").read_text(encoding="utf-8")

    _assert_job_does_not_run_for_oss_tests(oss_base_minimal_ci, "image.base.minimal")
    _assert_job_does_not_run_for_oss_tests(oss_minimal_ci, "image.minimal")
    _assert_job_does_not_run_for_oss_tests(oss_train_ci, "image.train.arm64")
    _assert_job_does_not_run_for_oss_tests(oss_train_ci, "image.train.amd64")
    _assert_job_does_not_run_for_oss_tests(oss_train_ci, "image.train")

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


def _job_block(ci_config: str, job_name: str) -> str:
    job_match = re.search(
        rf"(?ms)^{re.escape(job_name)}:\n.*?(?=^[A-Za-z0-9_.-]+:|\Z)",
        ci_config,
    )
    assert job_match is not None
    return job_match.group(0)
