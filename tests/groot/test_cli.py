# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public OSS CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
import subprocess

from groot.infra import oss


def test_oss_train_dry_run_uses_one_launcher(monkeypatch, capsys, tmp_path: Path) -> None:
    paths = _set_training_env(monkeypatch, tmp_path)

    status = oss.main(
        [
            "train",
            "n2_yam_simulation_stack_bowls",
            "--compute-type",
            "h100",
            "--max-steps",
            "2",
            "--num-gpus",
            "8",
            "--batch-size",
            "4",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "python -m groot.core.training.experiment.experiment_ray" in output
    assert "n2_yam_simulation_stack_bowls/h100/data.yaml" in output
    assert "--training.no-upload_checkpoints" in output
    assert "--training.num_gpus 8" in output
    assert f"--model.tokenizer_path {paths['cosmos']}" in output
    expected_diffusion_arg = (
        f"--model.diffusion-model-cfg.diffusion_model_pretrained_path {paths['diffusion']}"
    )
    assert expected_diffusion_arg in output
    assert f"--model.vae-cfg.vae_pretrained_path {paths['vae']}" in output
    assert "FINAL_CHECKPOINT_PATH=" in output


def test_oss_train_invokes_experiment_ray_without_running_gpu_backend(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    paths = _set_training_env(monkeypatch, tmp_path)
    output_dir = tmp_path / "train-output"
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
            "n2_yam_simulation_stack_bowls",
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
    assert command[command.index("--model.tokenizer_path") + 1] == str(paths["cosmos"])
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["NUM_GPUS"] == "8"
    assert env["BATCH_SIZE"] == "4"
    assert env["PRETRAIN_CHECKPOINT"] == str(paths["pretrain"])
    assert f"FINAL_CHECKPOINT_PATH={output_dir / 'checkpoint-2'}" in output


def test_oss_eval_dry_run_uses_open_loop_eval_config(capsys, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    status = oss.main(
        [
            "eval",
            "n2_yam_simulation_stack_bowls",
            "--checkpoint",
            "hf://model/org/repo@main/checkpoint-100",
            "--dataset-path",
            str(dataset_path),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "model_uri=hf://model/org/repo@main/checkpoint-100" in output
    assert f"dataset_path={dataset_path}" in output
    assert "output_json=/tmp/groot_oss_open_loop_eval.json" in output


def test_oss_eval_invokes_open_loop_evaluator_without_running_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    dataset_path = tmp_path / "dataset"
    output_json = tmp_path / "eval.json"
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
            "n2_yam_simulation_stack_bowls",
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
        ]
    )

    assert status == 0
    assert len(configs) == 1
    config = configs[0]
    assert config.model_path == str(checkpoint)
    assert config.model_uri is None
    assert config.dataset_path == str(dataset_path)
    assert config.output_json == str(output_json)
    assert config.traj_ids == [0, 3]
    assert config.steps == 2
    assert config.action_horizon == 2
    assert config.denoising_steps == 1
    assert config.server_parity_tolerance == 0.2
    assert config.local_only is True


def test_oss_train_eval_passes_trained_checkpoint_to_eval_without_gpu_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _set_training_env(monkeypatch, tmp_path)
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
            "n2_yam_simulation_stack_bowls",
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
    assert eval_configs[0].model_path == str(output_dir / "checkpoint-2")
    assert eval_configs[0].model_uri is None


def test_oss_infra_symlinks_expose_only_temporary_data_credentials() -> None:
    package_root = _package_root()

    assert (package_root / "groot/infra/oss.py").is_symlink()
    assert (package_root / "groot/infra/launcher").is_symlink()
    assert (package_root / "groot/infra/experiments").is_symlink()
    assert (package_root / "groot/infra/eval").is_symlink()
    assert (package_root / "groot/infra/job/__init__.py").is_file()
    assert (package_root / "groot/infra/job/data_credentials.py").is_symlink()
    assert not (package_root / "groot/infra/job/__main__.py").exists()
    assert not (package_root / "groot/infra/job/ray_cluster.py").exists()
    assert not (package_root / "groot/infra/job/ray_job.py").exists()
    assert not (package_root / "groot/infra/ci").exists()


def _set_training_env(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "cosmos": tmp_path / "cosmos",
        "diffusion": tmp_path / "diffusion",
        "vae": tmp_path / "vae",
        "pretrain": tmp_path / "pretrain",
    }
    monkeypatch.setenv("COSMOS_MODEL_PATH", str(paths["cosmos"]))
    monkeypatch.setenv("DIFFUSION_TRANSFORMER_PATH", str(paths["diffusion"]))
    monkeypatch.setenv("VAE_PATH", str(paths["vae"]))
    monkeypatch.setenv("PRETRAIN_CHECKPOINT", str(paths["pretrain"]))
    monkeypatch.setenv("USER", "tester")
    return paths


def _package_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )
