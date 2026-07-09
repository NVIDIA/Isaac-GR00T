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

"""OSS-native Min-WAM simulation golden-run launcher helpers."""

from __future__ import annotations

# ruff: noqa: T201
from dataclasses import dataclass
import getpass
import pathlib
from importlib import resources
import math
import os
import re
import subprocess
import sys
import time
from typing import TextIO

import tyro

from groot.infra.job import DEFAULT_IMAGE, ClusterSpec, RayCluster, RayJob, RayJobSpec

_ENTRYPOINT_PREFIX = "cd /tmp"
_HF_ARTIFACT_ENV_VARS = (
    "GOLDEN_RUN_BASE_MODEL_REPO_ID",
    "GOLDEN_RUN_BASE_MODEL_REVISION",
    "GOLDEN_RUN_DATASET_REPO_ID",
    "GOLDEN_RUN_DATASET_REVISION",
    "GOLDEN_RUN_DATASET_LOCAL_NAME",
    "GOLDEN_RUN_FINAL_MODEL_REPO_ID",
    "GOLDEN_RUN_FINAL_MODEL_REVISION",
    "GROOT_HF_ARTIFACT_CACHE_ROOT",
    "GROOT_HF_PREFETCH_MAX_CONCURRENT",
    "GROOT_HF_DOWNLOAD_RETRIES",
    "GROOT_HF_DOWNLOAD_BACKOFF_SECONDS",
)
_DEFAULT_HF_ARTIFACT_CACHE_ROOT = "/tmp/groot_hf_artifacts"  # noqa: S108
_DEFAULT_HF_PREFETCH_MAX_CONCURRENT = 1
_DEFAULT_HF_DOWNLOAD_RETRIES = 5
_DEFAULT_HF_DOWNLOAD_BACKOFF_SECONDS = 30.0
_DEFAULT_RAY_NODE_WAIT_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class Recipe:
    label: str
    data_yaml: str
    default_job_name: str
    max_n_cameras: int


@dataclass
class Submit:
    """Submit training job to a Ray cluster."""

    cluster_id: str | None = None
    compute_type: str = os.environ.get("CLUSTER_TYPE", "gb200")
    pool: str | None = None
    cluster_name: str | None = None
    image: str = os.environ.get("GROOT_TRAIN_IMAGE", DEFAULT_IMAGE)
    priority: str = os.environ.get("OSMO_PRIORITY", "high")
    num_nodes: int = int(os.environ.get("NUM_NODES", "1"))
    wait_for_cluster: bool = True
    tail_logs: bool = False
    log_file: str | None = None
    checkpoint_artifact_file: str | None = None
    status_poll_interval_seconds: float = 60.0


@dataclass
class Run:
    """Run training on the cluster."""


def main(recipe: Recipe, launcher_module: str) -> None:
    cmd = tyro.extras.subcommand_cli_from_dict(
        {
            "submit": Submit,
            "run": Run,
        }
    )
    if isinstance(cmd, Submit):
        _submit_to_cluster(recipe, launcher_module, cmd)
    elif isinstance(cmd, Run):
        _run_on_cluster(recipe)


def _cluster_type() -> str:
    return os.environ.get("CLUSTER_TYPE", "gb200")


def _job_name(recipe: Recipe) -> str:
    return os.environ.get("JOB_NAME", recipe.default_job_name)


def _user_slug() -> str:
    return os.environ.get("USER_SLUG") or getpass.getuser()


def _pretrain_checkpoint(recipe: Recipe) -> str:
    if checkpoint := os.environ.get("PRETRAIN_CHECKPOINT"):
        return checkpoint
    if repo_id := os.environ.get("GOLDEN_RUN_BASE_MODEL_REPO_ID"):
        revision = os.environ.get("GOLDEN_RUN_BASE_MODEL_REVISION", "main")
        local_dir = _hf_model_local_dir(repo_id, revision)
        return _snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=local_dir,
        )
    raise ValueError("PRETRAIN_CHECKPOINT or GOLDEN_RUN_BASE_MODEL_REPO_ID must be set")


def _dataset_root(recipe: Recipe) -> str:
    if repo_id := os.environ.get("GOLDEN_RUN_DATASET_REPO_ID"):
        revision = os.environ.get("GOLDEN_RUN_DATASET_REVISION", "main")
        local_name = os.environ.get("GOLDEN_RUN_DATASET_LOCAL_NAME")
        if not local_name:
            raise ValueError("GOLDEN_RUN_DATASET_LOCAL_NAME must be set with HF dataset artifacts")
        local_dir = _hf_dataset_local_dir(repo_id, revision, local_name)
        _snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
        return str(_hf_dataset_root(repo_id, revision))
    if dataset_root := os.environ.get("LFS_CACHE_PATH"):
        return dataset_root
    raise ValueError(
        "LFS_CACHE_PATH or GOLDEN_RUN_DATASET_REPO_ID and GOLDEN_RUN_DATASET_LOCAL_NAME must be set"
    )


def _data_yaml_path(recipe: Recipe) -> str:
    prefix = "groot/core/training/"
    if not recipe.data_yaml.startswith(prefix):
        return recipe.data_yaml
    package_relative_path = recipe.data_yaml.removeprefix(prefix)
    return str(resources.files("groot.core.training").joinpath(package_relative_path))


def _max_steps() -> str:
    return os.environ.get("MAX_STEPS", "10000")


def _training_output_dir(recipe: Recipe) -> str:
    if output_dir := os.environ.get("TRAINING_OUTPUT_DIR"):
        return output_dir
    return f"/tmp/groot_outputs/{_user_slug()}/{_job_name(recipe)}"  # noqa: S108


def _training_args(recipe: Recipe, pretrain_checkpoint: str | None = None) -> list[str]:
    cosmos_model_path = os.environ.get("COSMOS_MODEL_PATH", "")
    if not cosmos_model_path:
        raise ValueError("COSMOS_MODEL_PATH must be set")
    diffusion_transformer_path = os.environ.get("DIFFUSION_TRANSFORMER_PATH")
    if not diffusion_transformer_path:
        raise ValueError("DIFFUSION_TRANSFORMER_PATH must be set")
    vae_path = os.environ.get("VAE_PATH")
    if not vae_path:
        raise ValueError("VAE_PATH must be set")
    return [
        "--load_config_path",
        _data_yaml_path(recipe),
        "--data.seed",
        "24",
        "--data.mode",
        "single_turn",
        "--data.no-download-cache",
        "--data.allow_padding",
        "--data.video_backend",
        "torchcodec",
        "--data.override_pretraining_statistics",
        "--training.output_dir",
        _training_output_dir(recipe),
        "--training.learning_rate",
        "5e-5",
        "--training.weight_decay",
        "1e-5",
        "--training.logging_steps",
        "10",
        "--training.save_steps",
        "1000",
        "--training.save_total_limit",
        "5",
        "--training.max_steps",
        _max_steps(),
        "--training.warmup_ratio",
        "0.05",
        "--training.bf16",
        "--training.tf32",
        "--training.optim",
        "adamw_torch",
        "--training.lr_scheduler_type",
        "cosine",
        *_wandb_args(),
        "--training.per_device_batch_size",
        os.environ.get("BATCH_SIZE", "12"),
        "--training.num_gpus",
        os.environ.get("NUM_GPUS", "32"),
        "--training.dataloader_num_workers",
        os.environ.get("DATALOADER_NUM_WORKERS", "4"),
        "--training.start_from_checkpoint",
        pretrain_checkpoint or _pretrain_checkpoint(recipe),
        "--training.eval_strategy",
        "no",
        "model:GrootN2",
        "--model.model_dtype",
        "bfloat16",
        "--model.model_size",
        "2B",
        "--model.train_architecture",
        "full",
        "--model.image_target_size",
        "224",
        "320",
        "--model.tokenizer_path",
        cosmos_model_path,
        "--model.cond_timestep",
        "0.1",
        "--model.no-tiled",
        "--model.view_embed_dim",
        "7",
        "--model.max_n_cameras",
        str(recipe.max_n_cameras),
        "--model.video_scheduler_type",
        "beta",
        "--model.video_noise_beta_alpha",
        "7.0",
        "--model.video_noise_beta_beta",
        "1.0",
        "--model.video_noise_s",
        "0.999",
        "model.diffusion-model-cfg:diffusion-model-config",
        "--model.diffusion_model_cfg.diffusion_model_pretrained_path",
        diffusion_transformer_path,
        "--model.diffusion_model_cfg.model_size",
        "2B",
        "model.text-encoder-cfg:text-encoder-config",
        "--model.text_encoder_cfg.text_encoder_pretrained_path",
        cosmos_model_path,
        "model.vae-cfg:vae-config",
        "--model.vae_cfg.vae_pretrained_path",
        vae_path,
        "model.action-cfg:action-config",
        "--model.action-cfg.max-action-dim",
        "128",
        "--model.action-cfg.max-state-dim",
        "128",
        "--model.action-cfg.action-horizon",
        "40",
        "--model.action-cfg.max-action-horizon",
        "40",
        "--model.action-cfg.num-embodiments",
        "1",
        "--model.action-cfg.use_percentiles",
        "--model.action-cfg.clip_outliers",
        "--model.action-cfg.no-apply-sincos-state-encoding",
        "--model.action-cfg.state-dropout-prob",
        "0.5",
        "--model.action-cfg.attention-mask-pattern",
        "bidirectional_video_action_full",
        "1.0",
    ]


def _run_on_cluster(recipe: Recipe) -> None:
    num_gpus = int(os.environ.get("NUM_GPUS", "32"))
    per_device_batch = int(os.environ.get("BATCH_SIZE", "12"))
    global_batch = num_gpus * per_device_batch
    _prefetch_hf_artifacts_on_ray_nodes(recipe)
    dataset_root = _dataset_root(recipe)
    pretrain_checkpoint = _pretrain_checkpoint(recipe)

    print(f"\n{'=' * 70}")
    print(recipe.label.upper())
    print(
        f"  {_max_steps()} steps | "
        f"{num_gpus} GPUs | batch {global_batch} ({per_device_batch}/GPU) | lr 5e-5"
    )
    print(f"  Cluster: {_cluster_type()}")
    print(f"  Data config: {_data_yaml_path(recipe)}")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Checkpoint: {pretrain_checkpoint}")
    print(f"{'=' * 70}")
    os.environ["LFS_CACHE_PATH"] = dataset_root
    subprocess.run(
        [
            "python",
            "-m",
            "groot.core.training.experiment.experiment_ray",
            *_training_args(recipe, pretrain_checkpoint=pretrain_checkpoint),
        ],
        check=True,
    )
    _publish_final_checkpoint(recipe)


def _wandb_args() -> list[str]:
    if not os.environ.get("WANDB_API_KEY"):
        return []
    return [
        "--training.use_wandb",
        "--training.wandb_project",
        os.environ.get("WANDB_PROJECT", "groot_n2_posttrain"),
    ]


def _submit_to_cluster(recipe: Recipe, launcher_module: str, cmd: Submit) -> None:
    cluster_id = cmd.cluster_id or _launch_cluster(recipe, cmd)
    env_vars = _submit_env_vars(recipe, cmd.compute_type)
    job = RayJob(
        RayJobSpec(
            cluster_id=cluster_id,
            job_name=_job_name(recipe),
            entrypoint=f"{_ENTRYPOINT_PREFIX} && python -m {launcher_module} run",
            env_vars=env_vars,
            pip_packages=(),
            working_dir=None,
        )
    )
    result = job.submit()
    print(f"Submitted training job {result.submission_id} to Ray cluster {result.cluster_id}")
    for command in result.followup_commands:
        print(f"  {command}")
    if cmd.tail_logs:
        log_file = pathlib.Path(cmd.log_file) if cmd.log_file else None
        if log_file:
            with log_file.open("w", encoding="utf-8") as stream:
                job.tail_logs(result.submission_id, output=_RedactingTeeWriter(stream, sys.stdout))
        else:
            job.tail_logs(result.submission_id, output=_RedactingTeeWriter(sys.stdout))
        try:
            status = job.wait_until_terminal(
                result.submission_id,
                poll_interval_seconds=cmd.status_poll_interval_seconds,
            )
        except RuntimeError:
            info = job.info(result.submission_id)
            logs = job.logs(result.submission_id)
            detail = f"Ray job info:\n{info}\n\nRay job logs:\n{logs}"
            if log_file:
                with log_file.open("a", encoding="utf-8") as stream:
                    stream.write(detail)
            else:
                print(detail, end="")  # noqa: T201
            raise
        print(f"Ray job {result.submission_id} finished with status {status}")
        if cmd.checkpoint_artifact_file:
            checkpoint_path = _final_checkpoint_path(recipe)
            pathlib.Path(cmd.checkpoint_artifact_file).write_text(
                f"{checkpoint_path}\n", encoding="utf-8"
            )
            print(f"Final checkpoint path: {checkpoint_path}")
class _RedactingTeeWriter:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams
        self._secret_values = tuple(
            value
            for key in (
                "WANDB_API_KEY",
                "WANDB_ENTITY",
                "WANDB_BASE_URL",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HUGGINGFACE_TOKEN",
            )
            if (value := os.environ.get(key))
        )

    def write(self, chunk: str) -> int:
        redacted = chunk
        for value in self._secret_values:
            redacted = redacted.replace(value, "[REDACTED]")
        for stream in self._streams:
            stream.write(redacted)
            stream.flush()
        return len(chunk)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _final_checkpoint_path(recipe: Recipe) -> str:
    checkpoint_name = f"checkpoint-{_max_steps()}"
    if repo_id := os.environ.get("GOLDEN_RUN_FINAL_MODEL_REPO_ID"):
        revision = os.environ.get("GOLDEN_RUN_FINAL_MODEL_REVISION", "main")
        return f"hf://model/{repo_id}@{revision}/{checkpoint_name}"
    return str(pathlib.Path(_training_output_dir(recipe)) / checkpoint_name)


def _submit_env_vars(recipe: Recipe, compute_type: str) -> dict[str, str]:
    env_vars = {
        "JOB_NAME": _job_name(recipe),
        "USER_SLUG": _user_slug(),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", "12"),
        "NUM_GPUS": os.environ.get("NUM_GPUS", "32"),
        "NUM_NODES": os.environ.get("NUM_NODES", "1"),
        "MAX_STEPS": _max_steps(),
        "DATALOADER_NUM_WORKERS": os.environ.get("DATALOADER_NUM_WORKERS", "4"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "groot_n2_posttrain"),
        "CLUSTER_TYPE": compute_type,
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "TORCHINDUCTOR_CUDAGRAPH_TREES": "0",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_clean",  # noqa: S108
    }
    if os.environ.get("GOLDEN_RUN_DATASET_REPO_ID"):
        env_vars["LFS_CACHE_PATH"] = _hf_dataset_root_from_env()
    elif os.environ.get("LFS_CACHE_PATH"):
        env_vars["LFS_CACHE_PATH"] = os.environ["LFS_CACHE_PATH"]
    hf_token = _hf_token()
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    for key in (
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_BASE_URL",
        "HF_HUB_OFFLINE",
        "COSMOS_MODEL_PATH",
        "DIFFUSION_TRANSFORMER_PATH",
        "VAE_PATH",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        *_HF_ARTIFACT_ENV_VARS,
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


def _publish_final_checkpoint(recipe: Recipe) -> None:
    repo_id = os.environ.get("GOLDEN_RUN_FINAL_MODEL_REPO_ID")
    if not repo_id:
        return
    token = _hf_token()
    if not token:
        raise ValueError(
            "HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or HUGGINGFACE_TOKEN is required to publish final checkpoint"
        )
    revision = os.environ.get("GOLDEN_RUN_FINAL_MODEL_REVISION", "main")
    checkpoint_dir = pathlib.Path(_training_output_dir(recipe)) / f"checkpoint-{_max_steps()}"
    upload_dir = checkpoint_dir
    ignore_patterns: tuple[str, ...] = ()
    if not checkpoint_dir.is_dir():
        output_dir = pathlib.Path(_training_output_dir(recipe))
        if not output_dir.is_dir():
            raise FileNotFoundError(f"final checkpoint directory does not exist: {checkpoint_dir}")
        upload_dir = output_dir
        ignore_patterns = ("checkpoint-*", "profiling", "profiling/**")
        print(
            f"Final checkpoint directory was cleaned up after checkpoint upload; "
            f"publishing final model directory instead: {upload_dir}"
        )
    _create_hf_repo(repo_id=repo_id, repo_type="model", token=token)
    _upload_hf_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_in_repo=checkpoint_dir.name,
        token=token,
        ignore_patterns=ignore_patterns,
    )
    print(f"Published final checkpoint: {_final_checkpoint_path(recipe)}")


def _hf_cache_root() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("GROOT_HF_ARTIFACT_CACHE_ROOT", _DEFAULT_HF_ARTIFACT_CACHE_ROOT)
    )


def _hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN", "")
    )


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", value).strip("-") or "artifact"


def _hf_model_local_dir(repo_id: str, revision: str) -> pathlib.Path:
    return (
        _hf_cache_root() / "models" / _safe_path_component(repo_id) / _safe_path_component(revision)
    )


def _hf_dataset_local_dir(repo_id: str, revision: str, local_name: str) -> pathlib.Path:
    return _hf_dataset_root(repo_id, revision) / "datasets" / local_name


def _hf_dataset_root(repo_id: str, revision: str) -> pathlib.Path:
    return (
        _hf_cache_root()
        / "datasets"
        / _safe_path_component(repo_id)
        / _safe_path_component(revision)
    )


def _hf_dataset_root_from_env() -> str:
    repo_id = os.environ.get("GOLDEN_RUN_DATASET_REPO_ID", "")
    revision = os.environ.get("GOLDEN_RUN_DATASET_REVISION", "main")
    local_name = os.environ.get("GOLDEN_RUN_DATASET_LOCAL_NAME", "")
    if not repo_id or not local_name:
        raise ValueError(
            "GOLDEN_RUN_DATASET_REPO_ID and GOLDEN_RUN_DATASET_LOCAL_NAME must be set "
            "to derive the HF dataset root"
        )
    return str(_hf_dataset_root(repo_id, revision))


def _hf_artifacts_to_download(recipe: Recipe) -> list[tuple[str, str, str, pathlib.Path]]:
    artifacts = []
    if not os.environ.get("PRETRAIN_CHECKPOINT"):
        if repo_id := os.environ.get("GOLDEN_RUN_BASE_MODEL_REPO_ID"):
            revision = os.environ.get("GOLDEN_RUN_BASE_MODEL_REVISION", "main")
            artifacts.append((repo_id, "model", revision, _hf_model_local_dir(repo_id, revision)))
    if repo_id := os.environ.get("GOLDEN_RUN_DATASET_REPO_ID"):
        revision = os.environ.get("GOLDEN_RUN_DATASET_REVISION", "main")
        local_name = os.environ.get("GOLDEN_RUN_DATASET_LOCAL_NAME")
        if not local_name:
            raise ValueError("GOLDEN_RUN_DATASET_LOCAL_NAME must be set with HF dataset artifacts")
        artifacts.append(
            (
                repo_id,
                "dataset",
                revision,
                _hf_dataset_local_dir(repo_id, revision, local_name),
            )
        )
    return artifacts


def _prefetch_hf_artifacts_on_ray_nodes(recipe: Recipe) -> None:
    artifacts = _hf_artifacts_to_download(recipe)
    if not artifacts:
        return

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", ignore_reinit_error=True)
    try:
        nodes = _wait_for_live_ray_nodes(ray)
        max_concurrent = _hf_prefetch_max_concurrent()
        print(
            f"Prefetching {len(artifacts)} HF artifact(s) on {len(nodes)} Ray node(s) "
            f"under {_hf_cache_root()} with max {max_concurrent} concurrent HF download(s)"
        )
        remote_download = ray.remote(num_cpus=0)(_download_hf_artifact)
        pending_refs = []
        token = _hf_token()
        tasks = [
            (str(node["NodeID"]), repo_id, repo_type, revision, local_dir)
            for node in nodes
            for repo_id, repo_type, revision, local_dir in artifacts
        ]
        while tasks or pending_refs:
            while tasks and len(pending_refs) < max_concurrent:
                node_id, repo_id, repo_type, revision, local_dir = tasks.pop(0)
                strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                pending_refs.append(
                    remote_download.options(scheduling_strategy=strategy).remote(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        revision=revision,
                        local_dir=str(local_dir),
                        token=token,
                    )
                )
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1)
            for ref in ready_refs:
                downloaded_path = ray.get(ref)
                print(f"  HF artifact ready: {downloaded_path}")
    finally:
        ray.shutdown()


def _wait_for_live_ray_nodes(ray_module) -> list[dict]:
    expected_nodes = _expected_ray_nodes()
    timeout_seconds = _ray_node_wait_timeout_seconds()
    started_at = time.monotonic()
    while True:
        nodes = [node for node in ray_module.nodes() if node.get("Alive")]
        if len(nodes) >= expected_nodes:
            return nodes
        if time.monotonic() - started_at >= timeout_seconds:
            raise RuntimeError(
                f"Only {len(nodes)} live Ray node(s) are available for HF artifact prefetch; "
                f"expected {expected_nodes}."
            )
        print(
            f"Waiting for Ray nodes before HF artifact prefetch: "
            f"{len(nodes)}/{expected_nodes} live"
        )
        time.sleep(10)


def _download_hf_artifact(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: str,
    token: str,
) -> str:
    from huggingface_hub import snapshot_download

    path = pathlib.Path(local_dir)
    path.mkdir(parents=True, exist_ok=True)
    retries = _hf_download_retries()
    backoff_seconds = _hf_download_backoff_seconds()
    for attempt in range(retries + 1):
        try:
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(path),
                token=token or None,
            )
        except Exception as exc:
            if attempt >= retries or not _is_retryable_hf_download_error(exc):
                raise
            sleep_seconds = min(backoff_seconds * (2**attempt), 300.0)
            print(
                f"HF download for {repo_type} {repo_id}@{revision} failed with a retryable "
                f"error; retrying in {sleep_seconds:.0f}s ({attempt + 1}/{retries})"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def _hf_prefetch_max_concurrent() -> int:
    return max(
        1,
        int(
            os.environ.get("GROOT_HF_PREFETCH_MAX_CONCURRENT", _DEFAULT_HF_PREFETCH_MAX_CONCURRENT)
        ),
    )


def _hf_download_retries() -> int:
    return max(0, int(os.environ.get("GROOT_HF_DOWNLOAD_RETRIES", _DEFAULT_HF_DOWNLOAD_RETRIES)))


def _hf_download_backoff_seconds() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "GROOT_HF_DOWNLOAD_BACKOFF_SECONDS", _DEFAULT_HF_DOWNLOAD_BACKOFF_SECONDS
            )
        ),
    )


def _expected_ray_nodes() -> int:
    expected_nodes = max(1, int(os.environ.get("NUM_NODES", "1")))
    if num_gpus := os.environ.get("NUM_GPUS"):
        gpus_per_node = max(1, int(os.environ.get("GROOT_GPUS_PER_NODE", "8")))
        expected_nodes = min(expected_nodes, math.ceil(int(num_gpus) / gpus_per_node))
    return expected_nodes


def _ray_node_wait_timeout_seconds() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "GROOT_RAY_NODE_WAIT_TIMEOUT_SECONDS",
                _DEFAULT_RAY_NODE_WAIT_TIMEOUT_SECONDS,
            )
        ),
    )


def _is_retryable_hf_download_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        retryable in message
        for retryable in (
            "429",
            "too many requests",
            "rate limit",
            "connection",
            "timeout",
            "temporarily unavailable",
            "server error",
            "502",
            "503",
            "504",
        )
    )


def _snapshot_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: pathlib.Path,
) -> str:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        token=_hf_token() or None,
    )


def _create_hf_repo(*, repo_id: str, repo_type: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, private=True)


def _upload_hf_folder(
    *,
    folder_path: pathlib.Path,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    token: str,
    ignore_patterns: tuple[str, ...] = (),
) -> None:
    from huggingface_hub import HfApi

    HfApi(token=token).upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        ignore_patterns=ignore_patterns or None,
    )


def _launch_cluster(recipe: Recipe, cmd: Submit) -> str:
    pool = cmd.pool or os.environ.get("OSMO_POOL", "")
    if not pool:
        raise ValueError("--pool or OSMO_POOL is required when --cluster-id is omitted")
    cluster_name = cmd.cluster_name or _default_cluster_name(recipe, cmd.compute_type)
    result = RayCluster(
        ClusterSpec(
            cluster_name=cluster_name,
            pool=pool,
            gpu_type=cmd.compute_type,
            num_nodes=cmd.num_nodes,
            image=cmd.image,
            priority=cmd.priority,
        )
    ).launch(wait=cmd.wait_for_cluster)
    print(f"Started Ray cluster {result.workflow_id} for {recipe.label}")
    return result.cluster_id


def _default_cluster_name(recipe: Recipe, compute_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _job_name(recipe).lower()).strip("-")
    slug = slug or "n2-minwam-sim-oss-golden"
    return f"{slug[:40]}-{compute_type}"
