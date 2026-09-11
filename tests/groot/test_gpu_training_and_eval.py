# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Functional OSS train/eval GPU smoke tests."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

from groot.infra import oss

_PRETRAIN_CHECKPOINT = "hf://model/org/repo@commit-sha/base_200k"
_CHECKPOINT_SENTINEL = 7.0

pytestmark = pytest.mark.gpu


def test_oss_training_and_open_loop_eval_gpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run real OSS training/eval with synthetic model dependencies and a fixture dataset."""
    import torch

    assert torch.cuda.is_available(), "OSS GPU smoke requires torch.cuda.is_available()"
    hf_calls = _install_external_artifact_fakes(monkeypatch, tmp_path)
    dataset_path = _dataset_fixture()
    output_dir = tmp_path / "train-output"
    max_steps = os.environ.get("OSS_GPU_TRAIN_MAX_STEPS", "2")
    resolved = _resolve_fixture_training(
        output_dir=output_dir,
        dataset_path=dataset_path,
        max_steps=max_steps,
    )

    _assert_resolved_training_command(resolved, output_dir, max_steps, dataset_path)
    _run_training_in_process(resolved, monkeypatch)
    assert any(
        kind == "snapshot"
        and call["repo_id"] == "org/repo"
        and "base_200k/**" in call.get("allow_patterns", ())
        for kind, call in hf_calls
    ), "GPU smoke did not exercise real HF checkpoint resolution"
    _assert_training_checkpoint(output_dir, max_steps, expected_sentinel=_CHECKPOINT_SENTINEL)
    _run_open_loop_eval(
        output_dir / f"checkpoint-{max_steps}",
        dataset_path,
        tmp_path,
        monkeypatch,
    )


def _resolve_fixture_training(
    *,
    output_dir: Path,
    dataset_path: Path,
    max_steps: str,
) -> oss.ResolvedOssTrain:
    return oss.resolve_train(
        oss._parse_args(  # noqa: SLF001 - exercise the same resolved command as the CLI.
            [
                "train",
                "--embodiment-tag",
                "xdof_relative_eef_relative_joint",
                "--compute-type",
                os.environ.get("OSS_GPU_COMPUTE_TYPE", "gb300"),
                "--max-steps",
                max_steps,
                "--num-gpus",
                os.environ.get("OSS_GPU_TRAIN_NUM_GPUS", "1"),
                "--batch-size",
                os.environ.get("OSS_GPU_TRAIN_BATCH_SIZE", "1"),
                "--output-dir",
                str(output_dir),
                "--pretrain-checkpoint",
                _PRETRAIN_CHECKPOINT,
                "--dataset-path",
                str(dataset_path),
                # N2 training needs video deltas with future frames (delta > 0);
                # otherwise all latents are conditioning and num_denoise_latent == 0.
                "--set",
                "data.modality_configs.xdof_relative_eef_relative_joint.video.delta_indices=[-6,-3,0,3,6,9,12]",
                "--set",
                "training.model.action-cfg.action-horizon=40",
                "--set",
                "training.model.action-cfg.max-action-horizon=40",
                "--set",
                (
                    "data.modality_configs.xdof_relative_eef_relative_joint."
                    f"action.delta_indices={json.dumps(list(range(40)))}"
                ),
                "--set",
                "data.num_shards_per_epoch=1",
                "--set",
                "data.episode_sampling_rate=0.05",
                "--set",
                "data.shard_size=64",
                "--set",
                "data.video_backend=mock",
                "--set",
                f"training.save_steps={max_steps}",
                "--set",
                "training.save_total_limit=1",
                "--set",
                "training.use_wandb=false",
                "--set",
                "training.enable_profiling=false",
                "--set",
                "training.msc_profile_url=''",
                "--set",
                "training.dataloader_num_workers=0",
                "--set",
                "training.dataloader_pin_memory=false",
                "--set",
                "training.use_async_checkpoint_save=false",
                "--set",
                "training.torch_compile=false",
                "--set",
                "training.straggler.enabled=false",
            ]
        )
    )


def _run_training_in_process(
    resolved: oss.ResolvedOssTrain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from groot.core.training.experiment import experiment_ray

    for key, value in resolved.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SKIP_EXPERIMENT_REGISTRATION", "1")
    for key in (
        "ACCELERATE_TORCH_DEVICE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "RAY_ADDRESS",
        "WORLD_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)
    config = _parse_experiment_ray_config(resolved.command)
    try:
        experiment_ray.train_func(config)
    finally:
        with contextlib.suppress(ImportError):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _parse_experiment_ray_config(command: list[str]):
    import tyro

    from groot.core.training.configs import base_config

    args = command[3:]
    config = tyro.cli(
        base_config.Config,
        default=base_config.get_default_config(),
        description=__doc__,
        args=args,
    )
    if config.load_config_path:
        config_path = Path(config.load_config_path)
        assert config_path.exists(), f"Config path does not exist: {config_path}"
        config = config.load(config_path)
        config.load_config_path = None
        config = tyro.cli(base_config.Config, default=config, description=__doc__, args=args)
    return config


def _run_open_loop_eval(
    checkpoint: Path,
    dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_json = tmp_path / "oss_policy_eval.json"
    monkeypatch.setenv("GROOT_N2_CODEC_ROUNDTRIP", "none")
    args = oss._parse_args(  # noqa: SLF001 - reuse the OSS eval CLI parser.
        [
            "eval",
            "--embodiment-tag",
            "xdof_relative_eef_relative_joint",
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(dataset_path),
            "--traj-ids",
            os.environ.get("OSS_GPU_EVAL_TRAJ_IDS", "0"),
            "--steps",
            os.environ.get("OSS_GPU_EVAL_STEPS", "1"),
            "--action-horizon",
            os.environ.get("OSS_GPU_EVAL_ACTION_HORIZON", "2"),
            "--denoising-steps",
            os.environ.get("OSS_GPU_EVAL_DENOISING_STEPS", "2"),
            "--server-parity-tolerance",
            os.environ.get("OSS_GPU_EVAL_SERVER_PARITY_TOLERANCE", "0.02"),
            "--output-json",
            str(output_json),
            "--local-only",
        ]
    )
    config = oss._eval_config(args)  # noqa: SLF001 - exercise the OSS eval config path.
    config.video_backend = "mock"  # type: ignore[assignment]
    status = oss._run_eval(config, dry_run=False)  # noqa: SLF001

    assert status == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["model_path"] == str(checkpoint)
    assert report["dataset_path"] == str(dataset_path)
    assert "local" in report["results"]


def _assert_resolved_training_command(
    resolved: oss.ResolvedOssTrain,
    output_dir: Path,
    max_steps: str,
    dataset_path: Path,
) -> None:
    command = resolved.command
    assert command[:3] == ["python", "-m", "groot.core.training.experiment.experiment_ray"]
    assert "--training.output_dir" in command
    assert command[command.index("--training.output_dir") + 1] == str(output_dir)
    assert "--training.max_steps" in command
    assert command[command.index("--training.max_steps") + 1] == max_steps
    assert "--data.datasets.0.dataset_paths" not in command
    assert "--load_config_path" in command
    data_config = Path(command[command.index("--load_config_path") + 1])
    data_config_text = data_config.read_text(encoding="utf-8")
    assert str(dataset_path) in data_config_text
    assert "video_backend: mock" in data_config_text
    assert "--model.tokenizer_path" in command
    assert command[command.index("--model.tokenizer_path") + 1] == oss.OSS_COSMOS_MODEL_URI
    assert "--model.vae-cfg.vae_pretrained_path" in command
    assert command[command.index("--model.vae-cfg.vae_pretrained_path") + 1] == oss.OSS_VAE_URI
    assert "--training.start_from_checkpoint" in command
    assert command[command.index("--training.start_from_checkpoint") + 1] == _PRETRAIN_CHECKPOINT
    assert "--training.msc_profile_url" in command
    assert command[command.index("--training.msc_profile_url") + 1] == ""
    assert resolved.env["NUM_GPUS"] == os.environ.get("OSS_GPU_TRAIN_NUM_GPUS", "1")
    assert resolved.env["BATCH_SIZE"] == os.environ.get("OSS_GPU_TRAIN_BATCH_SIZE", "1")


def _dataset_fixture() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tests/groot/core/training/experiment/dataloader/fixtures/backpack_2ep"
        if candidate.is_dir():
            return candidate
    message = "local backpack_2ep LeRobot fixture is unavailable"
    if os.environ.get("CI"):
        raise AssertionError(message)
    pytest.skip(message)


def _install_external_artifact_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[tuple[str, dict]]:
    """Replace only heavyweight external model artifacts, not image dependencies."""
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    from groot.core.training.models.n2 import model as model_module
    from groot.core.training.models.n2 import processor as processor_module

    monkeypatch.setattr(model_module, "CosmosTextEncoder", _mock_cosmos_text_encoder_class())
    monkeypatch.setattr(processor_module, "HuggingfaceTokenizer", _MockHuggingfaceTokenizer)

    hf_calls = _install_fake_hf_resolver(monkeypatch, tmp_path)

    return hf_calls


def _mock_cosmos_text_encoder_class():
    import torch
    from torch import nn

    class MockCosmosTextEncoder(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.max_seq_len = int(os.environ.get("OSS_GPU_MOCK_TEXT_SEQ_LEN", "512"))
            self.output_dim = int(os.environ.get("OSS_GPU_MOCK_TEXT_EMBED_DIM", "100352"))
            self._device_anchor = nn.Parameter(
                torch.zeros((), dtype=torch.bfloat16),
                requires_grad=False,
            )

        def prepare_input_ids(self, prompts):
            return torch.zeros(
                len(prompts),
                self.max_seq_len,
                dtype=torch.long,
                device=self._device_anchor.device,
            )

        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            return torch.zeros(
                batch_size,
                seq_len,
                self.output_dim,
                device=input_ids.device,
                dtype=torch.bfloat16,
            )

    return MockCosmosTextEncoder


class _MockHuggingfaceTokenizer:
    def __init__(self, _path: str, seq_len: int, **_kwargs):
        self.seq_len = seq_len

    def __call__(self, sequence, **_kwargs):
        import torch

        return torch.zeros(len(sequence), self.seq_len, dtype=torch.long)

    def save_pretrained(self, *_args, **_kwargs):
        return ()


def _install_fake_hf_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[tuple[str, dict]]:
    """Exercise the real HF URI resolver against a local fake Hub snapshot."""
    import huggingface_hub

    cache_root = tmp_path / "hf-cache"
    calls: list[tuple[str, dict]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        repo_id = kwargs["repo_id"].replace("/", "--")
        snapshot = cache_root / f"models--{repo_id}" / "snapshots" / "commit-sha"
        for pattern in kwargs.get("allow_patterns") or ():
            if pattern.endswith("/**"):
                artifact = pattern.removesuffix("/**")
                artifact_dir = snapshot / artifact
                artifact_dir.mkdir(parents=True, exist_ok=True)
                if artifact == "base_200k":
                    from safetensors.torch import save_file
                    import torch

                    save_file(
                        {
                            "text_encoder._device_anchor": torch.tensor(
                                _CHECKPOINT_SENTINEL,
                                dtype=torch.bfloat16,
                            )
                        },
                        artifact_dir / "model.safetensors",
                    )
        if not kwargs.get("allow_patterns"):
            snapshot.mkdir(parents=True, exist_ok=True)
        return str(snapshot)

    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    return calls


def _assert_training_checkpoint(
    output_dir: Path,
    max_steps: str,
    *,
    expected_sentinel: float,
) -> None:
    from safetensors import safe_open

    checkpoint = output_dir / f"checkpoint-{max_steps}"
    assert output_dir.is_dir(), f"training did not create output directory: {output_dir}"
    assert checkpoint.is_dir(), f"training did not create final checkpoint: {checkpoint}"
    assert any(
        path.is_file() for path in checkpoint.rglob("*")
    ), f"training checkpoint is empty: {checkpoint}"
    for shard in checkpoint.glob("model*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as state_dict:
            available_keys = state_dict.keys()
            if "text_encoder._device_anchor" in available_keys:
                actual = state_dict.get_tensor("text_encoder._device_anchor").item()
                assert actual == expected_sentinel
                return
    raise AssertionError("training checkpoint did not preserve the synthetic sentinel weight")
