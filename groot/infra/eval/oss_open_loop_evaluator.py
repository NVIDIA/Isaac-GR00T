# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OSS post-training open-loop evaluation orchestration."""

from __future__ import annotations

import gc
import importlib
import logging
import math
import os
from pathlib import Path
import re
from typing import Any

from groot.core.utils.storage.msc import HF_MODEL_URI_PREFIX
from groot.core.utils.storage.msc import resolve_model_path as resolve_portable_model_path
from groot.infra.eval.eval_report import EvalReport
from groot.infra.eval.hf_artifact_cache import HfArtifactCache
from groot.infra.eval.hf_model_uri import HfModelUri
from groot.infra.eval.metric_summary import MetricSummary
from groot.infra.eval.oss_open_loop_eval_config import (
    DEFAULT_CHECKPOINT_NAME,
    OssOpenLoopEvalConfig,
)
from groot.infra.eval.policy_client_adapter import PolicyClientAdapter
from groot.infra.eval.policy_server_process import PolicyServerProcess
from groot.infra.eval.resolved_artifacts import ResolvedArtifacts


class OssOpenLoopEvaluator:
    MODEL_WEIGHT_FILENAMES = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
    )
    CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

    def __init__(self, config: OssOpenLoopEvalConfig):
        self.config = config
        self.cache = HfArtifactCache.init_from_config(config)

    def resolve_model_path(self) -> Path:
        checkpoint = self.config.checkpoint or os.environ.get("FINAL_CHECKPOINT_PATH")
        if checkpoint:
            if not checkpoint.startswith(HF_MODEL_URI_PREFIX):
                return Path(checkpoint).expanduser()
            artifact = HfModelUri.init_from_uri(checkpoint)
        else:
            repo_id = os.environ.get("GOLDEN_RUN_FINAL_MODEL_REPO_ID")
            if not repo_id:
                raise ValueError(
                    "Set --checkpoint, FINAL_CHECKPOINT_PATH, or GOLDEN_RUN_FINAL_MODEL_REPO_ID"
                )
            artifact = HfModelUri(
                repo_id=repo_id,
                revision=os.environ.get("GOLDEN_RUN_FINAL_MODEL_REVISION", "main"),
                checkpoint_name=self.checkpoint_name(),
            )
            checkpoint = f"{HF_MODEL_URI_PREFIX}{artifact.repo_id}@{artifact.revision}/{artifact.checkpoint_name}"

        preferred = Path(
            resolve_portable_model_path(
                checkpoint,
                "checkpoint",
                cache_root=self.config.cache_root,
            )
        )
        return self.resolve_loadable_checkpoint(preferred.parent, artifact.checkpoint_name)

    def resolve_loadable_checkpoint(self, root: Path, checkpoint_name: str) -> Path:
        preferred = root / checkpoint_name
        if not preferred.exists() or self.has_model_weights(preferred):
            return preferred

        requested_step = self.checkpoint_step(checkpoint_name)
        candidates = [
            candidate
            for candidate in root.iterdir()
            if candidate.is_dir()
            and self.has_model_weights(candidate)
            and (
                requested_step is None
                or self.checkpoint_step(candidate.name) is None
                or self.checkpoint_step(candidate.name) <= requested_step
            )
        ]
        if not candidates:
            return preferred

        selected = max(candidates, key=lambda path: self.checkpoint_sort_key(path.name))
        logging.warning(
            "Requested checkpoint %s has no Hugging Face model weights; using %s instead.",
            preferred,
            selected,
        )
        return selected

    def has_model_weights(self, checkpoint_path: Path) -> bool:
        for filename in self.MODEL_WEIGHT_FILENAMES:
            if (checkpoint_path / filename).exists():
                return True
        return any(checkpoint_path.glob("model-*.safetensors"))

    def checkpoint_step(self, checkpoint_name: str) -> int | None:
        match = self.CHECKPOINT_RE.match(checkpoint_name)
        if not match:
            return None
        return int(match.group(1))

    def checkpoint_sort_key(self, checkpoint_name: str) -> tuple[int, str]:
        step = self.checkpoint_step(checkpoint_name)
        if step is None:
            return (-1, checkpoint_name)
        return (step, checkpoint_name)

    def resolve_dataset_path(self) -> Path:
        if self.config.dataset_path:
            return Path(self.config.dataset_path).expanduser()

        repo_id = self.config.dataset_repo_id or os.environ.get("GOLDEN_RUN_DATASET_REPO_ID")
        local_name = self.config.dataset_local_name or os.environ.get(
            "GOLDEN_RUN_DATASET_LOCAL_NAME"
        )
        if not repo_id or not local_name:
            raise ValueError(
                "Set --dataset-path or both GOLDEN_RUN_DATASET_REPO_ID and "
                "GOLDEN_RUN_DATASET_LOCAL_NAME"
            )
        revision = self.config.dataset_revision or os.environ.get(
            "GOLDEN_RUN_DATASET_REVISION", "main"
        )
        return self.cache.download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )

    def resolve_artifacts(self) -> ResolvedArtifacts:
        return ResolvedArtifacts(
            model_path=self.resolve_model_path(),
            dataset_path=self.resolve_dataset_path(),
        )

    def evaluate_policy(self, policy: Any, dataset_path: Path) -> MetricSummary:
        from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from groot.core.training.open_loop.open_loop_eval import evaluate_single_trajectory

        modality = self.modality_config_from_policy(policy)
        loader = LeRobotEpisodeLoader(
            dataset_path=str(dataset_path),
            modality_configs=modality,
            video_backend=self.config.video_backend,
            video_backend_kwargs=None,
        )
        embodiment_tag = self.embodiment_tag()
        per_trajectory: list[dict[str, float | int]] = []
        per_metric: dict[str, list[float]] = {}

        for traj_id in self.config.traj_ids:
            if traj_id >= len(loader):
                logging.warning(
                    "Trajectory ID %d is out of range for dataset length %d",
                    traj_id,
                    len(loader),
                )
                continue
            save_plot_path = None
            if self.config.save_plot_dir:
                save_plot_path = str(Path(self.config.save_plot_dir) / f"traj_{traj_id}.png")
            metrics = evaluate_single_trajectory(
                policy=policy,
                loader=loader,
                traj_id=traj_id,
                embodiment_tag=embodiment_tag,
                steps=self.config.steps,
                action_horizon=self.config.action_horizon,
                save_plot_path=save_plot_path,
                enable_frechet=True,
                enable_reverse_kl=False,
            )
            row: dict[str, float | int] = {"traj_id": traj_id}
            row.update({key: float(value) for key, value in metrics.items()})
            per_trajectory.append(row)
            for key, value in metrics.items():
                per_metric.setdefault(key, []).append(float(value))

        if not per_trajectory:
            raise RuntimeError("No valid trajectories were evaluated.")
        return MetricSummary.init_from_trajectory_metrics(per_trajectory, per_metric)

    def modality_config_from_policy(self, policy: Any) -> dict[str, Any]:
        from groot.core.data.types import ModalityConfig

        modality = policy.get_modality_config()
        return {
            key: ModalityConfig(**value) if isinstance(value, dict) else value
            for key, value in modality.items()
        }

    def evaluate_with_policy_server(self, model_path: Path, dataset_path: Path) -> MetricSummary:
        from groot.core.policy.server_client import PolicyClient

        with PolicyServerProcess(model_path, self.config, self.device()) as server:
            client = PolicyClient(
                address=server.address,
                timeout_ms=self.config.server_timeout_ms,
                jpeg_enabled=True,
            )
            adapter = PolicyClientAdapter(client, ready=not self.config.use_cuda_graph)
            try:
                return self.evaluate_policy(adapter, dataset_path)
            finally:
                client.close()

    def close_policy(self, policy: Any) -> None:
        close = getattr(policy, "close", None)
        if close is not None:
            close()

    def release_device_cache(self) -> None:
        gc.collect()
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_results(self, results: dict[str, MetricSummary]) -> None:
        for label, summary in results.items():
            self.check_finite_metrics(label, summary)
            if (
                self.config.max_mse is not None
                and summary.average.get("mse", 0.0) > self.config.max_mse
            ):
                raise RuntimeError(
                    f"{label} average mse {summary.average['mse']:.6f} exceeds "
                    f"--max-mse {self.config.max_mse:.6f}"
                )

        if "local" not in results or "server" not in results:
            return

        local = results["local"].average
        server = results["server"].average
        for metric in sorted(set(local) & set(server)):
            delta = abs(local[metric] - server[metric])
            if delta > self.config.server_parity_tolerance:
                raise RuntimeError(
                    f"local/server {metric} delta {delta:.6f} exceeds "
                    f"--server-parity-tolerance {self.config.server_parity_tolerance:.6f}"
                )

    def run(self) -> EvalReport:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        artifacts = self.resolve_artifacts()
        logging.info("Resolved model path: %s", artifacts.model_path)
        logging.info("Resolved dataset path: %s", artifacts.dataset_path)

        results: dict[str, MetricSummary] = {}
        local_policy = self.make_local_policy(artifacts.model_path)
        try:
            results["local"] = self.evaluate_policy(local_policy, artifacts.dataset_path)
        finally:
            self.close_policy(local_policy)
            local_policy = None
            self.release_device_cache()

        if not self.config.local_only:
            results["server"] = self.evaluate_with_policy_server(
                artifacts.model_path,
                artifacts.dataset_path,
            )

        self.validate_results(results)
        report = EvalReport(
            artifacts=artifacts,
            traj_ids=self.config.traj_ids,
            steps=self.config.steps,
            action_horizon=self.config.action_horizon,
            results=results,
        )
        output_path = Path(self.config.output_json)
        report.write_json(output_path)
        logging.info("Wrote eval report: %s", output_path)
        for label, summary in results.items():
            logging.info("%s averages: %s", label, summary.average)
        return report

    def checkpoint_name(self) -> str:
        return (
            self.config.checkpoint_name
            or os.environ.get("GOLDEN_RUN_CHECKPOINT_NAME")
            or DEFAULT_CHECKPOINT_NAME
        )

    def embodiment_tag(self):
        from groot.core.data.schema import EmbodimentTag

        try:
            return EmbodimentTag(self.config.embodiment_tag)
        except ValueError:
            return EmbodimentTag[self.config.embodiment_tag]

    def device(self) -> str:
        if self.config.device:
            return self.config.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def make_local_policy(self, model_path: Path):
        from groot.core.policy.n2_policy import Gr00tN2Policy

        return Gr00tN2Policy(
            embodiment_tag=self.embodiment_tag(),
            model_path=str(model_path),
            device=self.device(),
            num_inference_timesteps=self.config.denoising_steps,
            trt_engine_dir=self.config.trt_engine_dir,
            use_cuda_graph=self.config.use_cuda_graph,
        )

    def check_finite_metrics(self, label: str, summary: MetricSummary) -> None:
        for metric, value in summary.average.items():
            if not math.isfinite(value):
                raise RuntimeError(f"{label} average {metric} is not finite: {value}")
