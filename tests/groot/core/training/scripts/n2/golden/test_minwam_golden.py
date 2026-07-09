# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OSS Min-WAM golden launcher helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

from groot.infra.job import RayJobSubmissionResult
import pytest


def test_oss_training_symlinks_stay_minwam_n2_scoped() -> None:
    package_root = _package_root()
    expected_paths = (
        "groot/core/training/configs/model/n2.py",
        "groot/core/training/models/n2",
        "groot/core/training/models/modules/flowmatching_modules.py",
        "groot/core/data/dataset/sharded_mixture_dataset_grouped.py",
        "groot/core/data/dataset/sharded_single_step_dataset_grouped.py",
        "groot/core/data/dataset/sharded_single_step_rl_dataset_grouped.py",
        "groot/core/data/dataset/deltacat_sampled_stats.py",
        "groot/core/training/experiment/dist_utils.py",
        "groot/core/training/open_loop",
        "groot/core/policy",
    )
    for expected_path in expected_paths:
        assert (package_root / expected_path).exists()

    symlinks = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_symlink()
    }
    package_symlinks = {path for path in symlinks if path.startswith("groot/")}
    if package_symlinks:
        for expected_path in expected_paths:
            assert expected_path in package_symlinks
    forbidden_fragments = (
        "groot/core/training/experiment/submit.py",
        "groot/core/training/experiment/manifest.py",
        "groot/core/training/experiment/snapshots.py",
        "groot/core/training/models/modules/dit.py",
        "groot/core/training/models/modules/eagle_backbone.py",
        "groot/core/training/models/modules/embodiment_conditioned_mlp.py",
        "groot/core/training/models/modules/gemma_backbone.py",
        "groot/core/training/models/modules/modeling_eagle.py",
        "groot/core/training/models/modules/nn.py",
        "groot/core/training/models/modules/qwen3_backbone.py",
        "groot/core/training/models/modules/xemb_qwen2_5vl.py",
    )
    for forbidden_fragment in forbidden_fragments:
        assert not (package_root / forbidden_fragment).exists()
    assert not [
        link
        for link in package_symlinks
        if any(fragment in link for fragment in forbidden_fragments)
    ]


def test_oss_submit_uses_installed_package_without_ray_working_dir(
    monkeypatch, tmp_path, capsys
) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    calls = {}
    log_file = tmp_path / "ray.log"
    checkpoint_file = tmp_path / "final_checkpoint_path.txt"

    class FakeRayCluster:
        def __init__(self, _spec):
            raise AssertionError("cluster should not be launched")

    class FakeRayJob:
        def __init__(self, spec):
            calls["job_spec"] = spec

        def submit(self):
            return RayJobSubmissionResult(
                cluster_id=calls["job_spec"].cluster_id,
                submission_id="ray-job",
                followup_commands=(),
            )

        def tail_logs(self, submission_id, output=None):
            calls["tailed"] = (submission_id, output)
            output.write("ray log secret-token hf-secret\n")

        def wait_until_terminal(self, submission_id, *, poll_interval_seconds):
            calls["waited"] = (submission_id, poll_interval_seconds)
            return "SUCCEEDED"

    monkeypatch.setattr(minwam_sim_oss, "RayCluster", FakeRayCluster)
    monkeypatch.setattr(minwam_sim_oss, "RayJob", FakeRayJob)
    monkeypatch.setenv("CLUSTER_TYPE", "h100")
    monkeypatch.setenv("USER_SLUG", "test-user")
    monkeypatch.setenv("MAX_STEPS", "100")
    monkeypatch.setenv("WANDB_ENTITY", "secret-token")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REPO_ID", "org/final")
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REVISION", "main")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    minwam_sim_oss._submit_to_cluster(
        minwam_sim_oss.Recipe(
            label="Test OSS Golden",
            data_yaml="data.yaml",
            default_job_name="test-job",
            max_n_cameras=3,
        ),
        "groot.core.training.scripts.n2.golden.launch_n2_minwam_sim_oss_golden",
        minwam_sim_oss.Submit(
            cluster_id="wf-existing",
            compute_type="h100",
            tail_logs=True,
            log_file=str(log_file),
            checkpoint_artifact_file=str(checkpoint_file),
        ),
    )

    spec = calls["job_spec"]
    assert spec.cluster_id == "wf-existing"
    assert spec.job_name == "test-job"
    assert spec.working_dir is None
    assert spec.entrypoint == (
        "cd /tmp && python -m "
        "groot.core.training.scripts.n2.golden.launch_n2_minwam_sim_oss_golden run"
    )
    assert spec.env_vars["CLUSTER_TYPE"] == "h100"
    assert spec.env_vars["JOB_NAME"] == "test-job"
    assert spec.env_vars["MAX_STEPS"] == "100"
    assert spec.env_vars["HF_TOKEN"] == "hf-secret"
    assert "WANDB_API_KEY" not in spec.env_vars
    assert spec.pip_packages == ()
    assert calls["waited"] == ("ray-job", 60.0)
    assert "ray log [REDACTED] [REDACTED]" in capsys.readouterr().out
    assert log_file.read_text() == "ray log [REDACTED] [REDACTED]\n"
    assert checkpoint_file.read_text() == "hf://model/org/final@main/checkpoint-100\n"


def test_oss_training_args_enable_wandb_only_with_api_key(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    _set_required_training_env(monkeypatch)
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert "--training.use_wandb" not in minwam_sim_oss._training_args(recipe)

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    args = minwam_sim_oss._training_args(recipe)

    assert "--training.use_wandb" in args
    assert args[args.index("--training.wandb_project") + 1] == "test-project"
    assert minwam_sim_oss._submit_env_vars(recipe, "h100")["WANDB_API_KEY"] == "test-key"


def test_oss_training_args_support_max_steps_override(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    _set_required_training_env(monkeypatch)
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.setenv("MAX_STEPS", "100")
    args = minwam_sim_oss._training_args(recipe)

    assert args[args.index("--training.max_steps") + 1] == "100"
    assert minwam_sim_oss._submit_env_vars(recipe, "h100")["MAX_STEPS"] == "100"


def test_oss_training_args_do_not_upload_checkpoints_to_msc(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    _set_required_training_env(monkeypatch)
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    args = minwam_sim_oss._training_args(recipe)

    assert "--training.upload_checkpoints" not in args
    assert "--training.upload_every" not in args
    assert "--training.msc_checkpoint_url" not in args


def test_oss_training_args_use_hf_base_model_checkpoint(monkeypatch, tmp_path) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    _set_required_training_env(monkeypatch, pretrain_checkpoint=False)
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )
    calls = []

    def fake_snapshot_download(*, repo_id, repo_type, revision, local_dir):
        calls.append((repo_id, repo_type, revision, local_dir))
        return str(local_dir / "resolved-checkpoint")

    monkeypatch.setenv("CLUSTER_TYPE", "h100")
    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("GOLDEN_RUN_BASE_MODEL_REPO_ID", "org/base")
    monkeypatch.setenv("GOLDEN_RUN_BASE_MODEL_REVISION", "checkpoint-1")
    monkeypatch.setattr(minwam_sim_oss, "_snapshot_download", fake_snapshot_download)

    args = minwam_sim_oss._training_args(recipe)

    assert args[args.index("--training.start_from_checkpoint") + 1].endswith("resolved-checkpoint")
    assert calls == [
        (
            "org/base",
            "model",
            "checkpoint-1",
            tmp_path / "models" / "org--base" / "checkpoint-1",
        )
    ]


def test_oss_dataset_root_uses_hf_dataset(monkeypatch, tmp_path) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )
    calls = []

    def fake_snapshot_download(*, repo_id, repo_type, revision, local_dir):
        calls.append((repo_id, repo_type, revision, local_dir))
        return str(local_dir)

    monkeypatch.setenv("CLUSTER_TYPE", "h100")
    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REPO_ID", "org/data")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REVISION", "main")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_LOCAL_NAME", "dataset_local")
    monkeypatch.delenv("LFS_CACHE_PATH", raising=False)
    monkeypatch.setattr(minwam_sim_oss, "_snapshot_download", fake_snapshot_download)

    dataset_root = minwam_sim_oss._dataset_root(recipe)

    assert dataset_root == str(tmp_path / "datasets" / "org--data" / "main")
    assert calls[0][3] == Path(dataset_root) / "datasets" / "dataset_local"
    assert calls == [
        (
            "org/data",
            "dataset",
            "main",
            tmp_path / "datasets" / "org--data" / "main" / "datasets" / "dataset_local",
        )
    ]


def test_oss_training_args_require_explicit_model_asset_paths(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.setenv("PRETRAIN_CHECKPOINT", "/tmp/checkpoint")
    monkeypatch.delenv("COSMOS_MODEL_PATH", raising=False)
    monkeypatch.delenv("DIFFUSION_TRANSFORMER_PATH", raising=False)
    monkeypatch.delenv("VAE_PATH", raising=False)

    with pytest.raises(ValueError, match="COSMOS_MODEL_PATH"):
        minwam_sim_oss._training_args(recipe)


def test_oss_submit_env_forwards_hf_artifact_vars_with_worker_dataset_root(
    monkeypatch, tmp_path
) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.setenv("GOLDEN_RUN_DATASET_REPO_ID", "org/data")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REVISION", "main")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_LOCAL_NAME", "dataset_local")
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REPO_ID", "org/final")
    monkeypatch.setenv("GOLDEN_RUN_FINAL_MODEL_REVISION", "main")
    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("COSMOS_MODEL_PATH", "/tmp/cosmos")
    monkeypatch.setenv("DIFFUSION_TRANSFORMER_PATH", "/tmp/diffusion-transformer")
    monkeypatch.setenv("VAE_PATH", "/tmp/vae.pth")
    monkeypatch.setenv("NUM_NODES", "8")
    monkeypatch.delenv("LFS_CACHE_PATH", raising=False)

    env_vars = minwam_sim_oss._submit_env_vars(recipe, "h100")

    assert env_vars["GOLDEN_RUN_DATASET_REPO_ID"] == "org/data"
    assert env_vars["GOLDEN_RUN_FINAL_MODEL_REPO_ID"] == "org/final"
    assert env_vars["GOLDEN_RUN_FINAL_MODEL_REVISION"] == "main"
    assert env_vars["HF_TOKEN"] == "hf-secret"
    assert env_vars["HF_HUB_OFFLINE"] == "0"
    assert env_vars["COSMOS_MODEL_PATH"] == "/tmp/cosmos"
    assert env_vars["DIFFUSION_TRANSFORMER_PATH"] == "/tmp/diffusion-transformer"
    assert env_vars["VAE_PATH"] == "/tmp/vae.pth"
    assert env_vars["NUM_NODES"] == "8"
    assert env_vars["LFS_CACHE_PATH"] == str(tmp_path / "datasets" / "org--data" / "main")


def test_oss_submit_env_accepts_huggingface_token_alias(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-secret")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    env_vars = minwam_sim_oss._submit_env_vars(recipe, "h100")

    assert minwam_sim_oss._hf_token() == "hf-secret"
    assert env_vars["HF_TOKEN"] == "hf-secret"


def test_oss_hf_artifacts_to_download_uses_node_local_paths(monkeypatch, tmp_path) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    recipe = minwam_sim_oss.Recipe(
        label="Test OSS Golden",
        data_yaml="data.yaml",
        default_job_name="test-job",
        max_n_cameras=3,
    )

    monkeypatch.setenv("GROOT_HF_ARTIFACT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("GOLDEN_RUN_BASE_MODEL_REPO_ID", "org/base")
    monkeypatch.setenv("GOLDEN_RUN_BASE_MODEL_REVISION", "feature/checkpoint")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REPO_ID", "org/data")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_REVISION", "main")
    monkeypatch.setenv("GOLDEN_RUN_DATASET_LOCAL_NAME", "dataset_local")

    assert minwam_sim_oss._hf_artifacts_to_download(recipe) == [
        (
            "org/base",
            "model",
            "feature/checkpoint",
            tmp_path / "models" / "org--base" / "feature--checkpoint",
        ),
        (
            "org/data",
            "dataset",
            "main",
            tmp_path / "datasets" / "org--data" / "main" / "datasets" / "dataset_local",
        ),
    ]


def test_oss_hf_prefetch_defaults_to_single_download(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()

    monkeypatch.delenv("GROOT_HF_PREFETCH_MAX_CONCURRENT", raising=False)
    assert minwam_sim_oss._hf_prefetch_max_concurrent() == 1

    monkeypatch.setenv("GROOT_HF_PREFETCH_MAX_CONCURRENT", "3")
    assert minwam_sim_oss._hf_prefetch_max_concurrent() == 3


def test_oss_expected_ray_nodes_uses_required_gpu_nodes(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()

    monkeypatch.setenv("NUM_NODES", "8")
    monkeypatch.setenv("NUM_GPUS", "64")
    monkeypatch.delenv("GROOT_GPUS_PER_NODE", raising=False)

    assert minwam_sim_oss._expected_ray_nodes() == 8


def test_oss_waits_for_expected_ray_nodes(monkeypatch) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()

    class FakeRay:
        def __init__(self) -> None:
            self.calls = 0

        def nodes(self):
            self.calls += 1
            if self.calls == 1:
                return [{"Alive": True, "NodeID": "node-1"}]
            return [{"Alive": True, "NodeID": "node-1"}, {"Alive": True, "NodeID": "node-2"}]

    fake_ray = FakeRay()
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setattr(minwam_sim_oss.time, "sleep", lambda _seconds: None)

    nodes = minwam_sim_oss._wait_for_live_ray_nodes(fake_ray)

    assert [node["NodeID"] for node in nodes] == ["node-1", "node-2"]
    assert fake_ray.calls == 2


def test_oss_hf_download_retries_rate_limits(monkeypatch, tmp_path) -> None:
    minwam_sim_oss = _load_oss_minwam_sim()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("429 Client Error: Too Many Requests")
        return kwargs["local_dir"]

    monkeypatch.setenv("GROOT_HF_DOWNLOAD_RETRIES", "2")
    monkeypatch.setenv("GROOT_HF_DOWNLOAD_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(minwam_sim_oss.time, "sleep", lambda _seconds: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    downloaded_path = minwam_sim_oss._download_hf_artifact(
        repo_id="org/model",
        repo_type="model",
        revision="main",
        local_dir=str(tmp_path / "model"),
        token="hf-secret",
    )

    assert downloaded_path == str(tmp_path / "model")
    assert len(calls) == 2
    assert calls[0]["token"] == "hf-secret"


def _load_oss_minwam_sim():
    module_path = _package_root() / "groot/core/training/scripts/n2/golden/minwam_sim_oss.py"
    spec = importlib.util.spec_from_file_location("oss_minwam_sim_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_required_training_env(monkeypatch, *, pretrain_checkpoint: bool = True) -> None:
    if pretrain_checkpoint:
        monkeypatch.setenv("PRETRAIN_CHECKPOINT", "/tmp/pretrain-checkpoint")
    else:
        monkeypatch.delenv("PRETRAIN_CHECKPOINT", raising=False)
    monkeypatch.setenv("COSMOS_MODEL_PATH", "/tmp/cosmos")
    monkeypatch.setenv("DIFFUSION_TRANSFORMER_PATH", "/tmp/diffusion-transformer")
    monkeypatch.setenv("VAE_PATH", "/tmp/vae.pth")


def _repo_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())


def _package_root() -> Path:
    repo_root = _repo_root()
    package_dir = os.environ.get("OSS_PACKAGE_DIR")
    if package_dir:
        package_root = Path(package_dir)
        if not package_root.is_absolute():
            package_root = repo_root / package_root
        return package_root

    oss_root = repo_root / "oss"
    if (oss_root / "groot").is_dir():
        return oss_root
    return repo_root
