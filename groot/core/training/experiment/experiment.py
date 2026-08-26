#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unified Training Script for ICRT v2.

Supports:
- ICRT training (multi-turn with demonstrations)
- Behavior Cloning with trajectory mode (full trajectory as conversation)
- Behavior Cloning with single_turn mode (state→action mapping)

Usage:
    # ICRT training
    python train.py --data.task icrt --training.num_gpus 8

    # BC trajectory mode
    python train.py --data.task bc --data.mode trajectory --training.num_gpus 8

    # BC single_turn mode
    python train.py --data.task bc --data.mode single_turn --training.num_gpus 8
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Literal, cast, get_args
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import (
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import wandb

from groot.core.data.dataset import deltacat_sampled_stats
from groot.core.data.dataset.build_artifacts import DatasetBuildArtifacts
from groot.core.data.dataset.factory import DatasetFactory as _ValDatasetFactory
from groot.core.training.configs.base_config import Config
from groot.core.training.experiment.callbacks.msc_checkpoint import MSCCheckpointCallback
from groot.core.training.experiment.callbacks.validation_loss import ValidationLossCallback
from groot.core.training.experiment.gear_registration import (
    append_training_run_status_event,
    dashboard_checkpoint_path_to_msc_url,
    one_launcher_resume_checkpoint,
    preserve_checkpoint_root_enabled,
    register_experiment,
    resolve_training_output_dir,
    update_training_run,
)
from groot.core.training.experiment.tracking_policy import (
    should_fallback_register_experiment,
)

# Use custom trainer that profiles data loading & forward times
from groot.core.training.experiment.trainer import GR00TTrainer, ProfCallback
from groot.core.training.experiment.utils import (
    BestMetricCheckpointCallback,
    CheckpointFormatCallback,
    CheckpointUploadCallback,
    OpenLoopEvalCallback,
    RLCallback,
    SaveVlModelCallback,
    StragglerDetectionCallback,
    UnfreezeBackboneCallback,
)
from groot.core.training.models import MODEL_REGISTRY
from groot.core.training.models.identity.model import IdentityModel
from groot.core.training.models.n2.compile_diagnostics import N2CompileDiagnosticsCallback
from groot.core.utils.state_action.initial_actions import (
    INITIAL_ACTIONS_FILENAME,
    save_initial_actions,
)
from groot.core.utils.storage.data_paths import resolve_all_dataset_paths
from groot.core.utils.storage.msc import (
    DataSyncConfigBuilder,
    download_remote_checkpoint_for_resume,
    get_remote_last_checkpoint,
    resolve_checkpoint_path,
    upload_dir,
)
from groot.core.version import VERSION

TorchrunBackend = Literal["nccl", "gloo"]
_TORCHRUN_BACKENDS = get_args(TorchrunBackend)


def _resolve_torchrun_backend(config: Config) -> TorchrunBackend:
    backend = (config.training.distributed_backend or "").strip().lower()
    if backend:
        if backend not in _TORCHRUN_BACKENDS:
            raise ValueError(
                "training.distributed_backend must be one of "
                f"{list(_TORCHRUN_BACKENDS)}, got {backend!r}"
            )
        return cast(TorchrunBackend, backend)
    if config.training.num_gpus > 0:
        return "nccl"
    return "gloo"


def update_training_run_profile_job_id(
    *,
    global_rank: int,
    experiment_id: str,
    training_run_id: str,
    profile_job_id: str,
) -> None:
    """Mirror the PyTorch profiler job id onto the dashboard training run.

    No-op unless this process is rank 0 and the launcher registered a training
    run. Call this after rank 0 successfully uploads its profiler trace to MSC.
    The dashboard can derive the fixed profile root from ``profile_job_id``:
    ``GearCheckpoints/profiles/profiling/<profile_job_id>``.
    """
    if global_rank != 0 or not experiment_id or not training_run_id or not profile_job_id:
        return
    try:
        update_training_run(
            experiment_id,
            training_run_id,
            profile_job_id=profile_job_id,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Dashboard profile_job_id PATCH failed (non-fatal): %s", exc)


class OLTrainingProgressCallback(TrainerCallback):
    """Emit structured progress lines for one-launcher terminal monitoring."""

    _TRAINING_METRIC_KEYS = frozenset(
        {
            "train/loss",
            "training_step_time",
            "model_forward_time",
        }
    )

    def __init__(self, *, global_rank: int):
        self.global_rank = global_rank
        self.start_time: float | None = None
        self.start_step = 0
        self.last_emitted_step = -1

    def on_train_begin(self, _args, state, _control, **_kwargs):
        if self.global_rank != 0:
            return
        self.start_time = time.monotonic()
        self.start_step = int(state.global_step or 0)
        sys.stdout.write(
            "OL_PHASE name=model_loading status=completed message=model and trainer are ready\n"
            "OL_PHASE name=training status=started message=optimizer loop started\n"
        )
        sys.stdout.flush()

    def on_log(self, args, state, _control, logs=None, **_kwargs):
        if self.global_rank != 0 or not logs:
            return
        if not any(key in logs for key in self._TRAINING_METRIC_KEYS):
            return
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.start_step = max(0, int(state.global_step or 0) - 1)
            sys.stdout.write(
                "OL_PHASE name=model_loading status=completed message=model and trainer are ready\n"
                "OL_PHASE name=training status=started message=optimizer loop started\n"
            )
            sys.stdout.flush()
        step = int(state.global_step or 0)
        total = int(args.max_steps)
        if step <= 0 or total <= 0:
            return
        elapsed_seconds = time.monotonic() - self.start_time
        self.emit(step=step, total=total, elapsed_seconds=elapsed_seconds)

    def on_step_end(self, args, state, _control, **_kwargs):
        if self.global_rank != 0 or self.start_time is None:
            return
        step = int(state.global_step or 0)
        total = int(args.max_steps)
        logging_steps = max(1, int(args.logging_steps or 1))
        first_step = self.start_step + 1
        if step not in {first_step, total} and step % logging_steps != 0:
            return
        elapsed_seconds = time.monotonic() - self.start_time
        self.emit(step=step, total=total, elapsed_seconds=elapsed_seconds)

    def emit(self, *, step: int, total: int, elapsed_seconds: float) -> None:
        if step == self.last_emitted_step:
            return
        completed_steps = max(0, step - self.start_step)
        avg_step_seconds = elapsed_seconds / completed_steps if completed_steps else 0.0
        sys.stdout.write(
            "OL_TRAIN_PROGRESS "
            f"step={step} total={total} "
            f"elapsed_s={elapsed_seconds:.1f} avg_step_s={avg_step_seconds:.3f}\n"
        )
        sys.stdout.flush()
        self.last_emitted_step = step


@dataclass
class OmniTrainingArguments(TrainingArguments):
    use_muon: bool = field(default=False)
    muon_impl: str = field(default="dist")  # "dist", "single_batched"
    muon_lr: float = field(default=0.05)
    use_legacy_wd_application: bool = field(default=False)
    log_distributed_aux_losses: bool = field(default=True)
    enable_nan_loss_diagnostics: bool = field(default=False)
    nan_loss_diagnostics_dir: str | None = field(default=None)
    nan_loss_diagnostics_max_dumps: int = field(default=4)
    log_distributed_grad_norm: bool = field(default=True)
    sync_seq_len: bool = field(default=False)
    seq_len_handle_strategy: str = field(default="pad")  # "pad", "truncate", or "fixed"
    use_truncated_bptt: bool = field(default=False)
    truncated_bptt_chunk_size: int = field(default=10000)
    use_async_checkpoint_save: bool = field(default=True)
    use_local_state_dict_for_zero2: bool = field(default=True)

    # WSD scheduler
    _use_wsd_scheduler: bool = field(default=False)
    wsd_stable_ratio: float = field(default=0.8)
    wsd_decay_type: str = field(default="cosine")


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def _setup_msc_config(config: Config):
    """Setup MSC configuration for remote storage access.

    Generates MSC config using DataSyncConfigBuilder and sets MSC_CONFIG environment variable.
    Credentials are read from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
    or from a credentials file specified by config.
    """
    credentials_file = str(Path.home() / ".gear" / "data_credentials")
    profile = "default"
    ckpt_bucket_name = getattr(config.training, "msc_ckpt_bucket", "GearCheckpoints")
    ckpt_use_rust_client = getattr(config.training, "msc_ckpt_use_rust_client", True)

    try:
        builder = DataSyncConfigBuilder(
            credentials_file=credentials_file,
            profile=profile,
            use_rust_client=ckpt_use_rust_client,
        )

        # Add checkpoint storage profile
        builder.add_profile(profile=ckpt_bucket_name, bucket=ckpt_bucket_name)

        # Dashboard event outbox writes to msc://GearHome/... on API fallback.
        if ckpt_bucket_name != "GearHome":
            builder.add_profile(profile="GearHome", bucket="GearHome")

        # Add file profile for local filesystem access
        builder.config["profiles"]["file"] = {
            "storage_provider": {"type": "file", "options": {"base_path": "/"}}
        }

        # Build config (creates temp file and sets MSC_CONFIG env var)
        builder.build()

        logging.info(f"MSC config initialized with bucket: {ckpt_bucket_name}")
    except Exception:
        logging.exception("Failed to setup MSC config")
        raise

    # Dsiable msc sync progress bar for better logging
    os.environ["SUPPRESS_PROGRESS_BAR"] = "True"


def _materialize_model_source_checkpoint(config: Config, global_rank: int) -> None:
    """Resolve only the checkpoint source that the model pipeline will consume."""
    start_from_checkpoint = config.training.start_from_checkpoint
    if start_from_checkpoint is not None:
        if start_from_checkpoint.startswith("msc://"):
            _setup_msc_config(config)
            config.training.start_from_checkpoint = resolve_checkpoint_path(
                checkpoint_path=start_from_checkpoint,
                output_dir=config.training.output_dir,
                global_rank=global_rank,
            )
        return

    pretrained_base_model_path = getattr(config.model, "pretrained_base_model_path", None)
    if isinstance(pretrained_base_model_path, str) and pretrained_base_model_path.startswith(
        "msc://"
    ):
        _setup_msc_config(config)
        config.model.pretrained_base_model_path = resolve_checkpoint_path(
            checkpoint_path=pretrained_base_model_path,
            output_dir=config.training.output_dir,
            global_rank=global_rank,
        )


def warn_configs(config: Config):
    if config.data.video_backend != "torchcodec":
        warnings.warn(
            "video_backend is not torchcodec. Only torchcodec will be supported in the future.",
            stacklevel=2,
        )

    if config.training.use_legacy_wd_application:
        warnings.warn(
            "use_legacy_wd_application is deprecated and should not be used. "
            "The legacy weight decay application behavior is not recommended for new experiments.",
            stacklevel=2,
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use "
            "warmup_ratio instead. For now, this will override warmup_ratio.",
            stacklevel=2,
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future.",
            stacklevel=2,
        )

    if (
        hasattr(config.model, "use_albumentations_transforms")
        and not config.model.use_albumentations_transforms
    ):
        warnings.warn(
            "use_albumentations_transforms is not True. This will be deprecated in the future.",
            stacklevel=2,
        )

    if (
        hasattr(config.model, "image_crop_size")
        and hasattr(config.model, "image_target_size")
        and (config.model.image_crop_size is not None or config.model.image_target_size is not None)
    ):
        assert (
            config.model.image_crop_size is not None and config.model.image_target_size is not None
        ), "image_crop_size and image_target_size must be set together"
        warnings.warn(
            "image_crop_size and image_target_size will be deprecated in the future. "
            "Please use shortest_image_edge and crop_fraction instead.",
            stacklevel=2,
        )
        if hasattr(config.model, "shortest_image_edge") and hasattr(config.model, "crop_fraction"):
            assert (
                config.model.shortest_image_edge is None and config.model.crop_fraction is None
            ), (
                "Do not set shortest_image_edge and crop_fraction "
                "together with image_crop_size and image_target_size"
            )

    if (
        hasattr(config.model, "shortest_image_edge")
        and hasattr(config.model, "crop_fraction")
        and (config.model.shortest_image_edge is not None or config.model.crop_fraction is not None)
    ):
        assert (
            config.model.use_albumentations_transforms
        ), "use_albumentations_transforms must be True when shortest_image_edge and crop_fraction are set"


def fetch_latest_checkpoint_from_remote_if_available(
    local_checkpoint_dir: Path,
    msc_checkpoint_url: str,
    global_rank: int,
) -> None:
    """Fetch the latest checkpoint from remote storage if available and newer than local.

    Only rank 0 performs the actual checks and downloads. All ranks synchronize using a barrier.

    Args:
        local_checkpoint_dir: Local directory where checkpoints are stored
        msc_checkpoint_url: Remote storage URL for checkpoints
        global_rank: Rank of the current process in distributed training
    """
    if global_rank == 0:
        # Step 1: Get latest local checkpoint
        local_checkpoints = sorted(
            local_checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        latest_local_step = -1
        latest_local_valid = False

        if local_checkpoints:
            latest_local = local_checkpoints[-1]
            latest_local_step = int(latest_local.name.split("-")[-1])

            # Verify checkpoint has either DeepSpeed state (global_step*/) or DCP state (dcp_state/)
            # Keep both checks until legacy DeepSpeed and early DCP checkpoints age out.
            has_deepspeed = len(list(latest_local.glob("global_step*"))) > 0
            has_dcp = (latest_local / "dcp_state" / ".metadata").exists()
            latest_local_valid = has_deepspeed or has_dcp

            if latest_local_valid:
                logging.info(
                    f"Rank 0 found valid local checkpoint: {latest_local.name} (step {latest_local_step})"
                )
            else:
                logging.warning(
                    f"Rank 0 found incomplete local checkpoint: {latest_local.name}, will ignore it"
                )

        # Step 2: Get latest remote checkpoint
        try:
            latest_checkpoint_url = get_remote_last_checkpoint(msc_checkpoint_url)
            if latest_checkpoint_url:
                checkpoint_name = latest_checkpoint_url.rstrip("/").split("/")[-1]
                latest_remote_step = int(checkpoint_name.split("-")[-1])
                logging.info(
                    f"Rank 0 found remote checkpoint: {checkpoint_name} (step {latest_remote_step})"
                )

                # Step 3: Compare and decide whether to download
                should_download = False
                if not latest_local_valid:
                    # No valid local checkpoint, download remote
                    should_download = True
                    logging.info(
                        f"Rank 0 no valid local checkpoint, will download remote checkpoint-{latest_remote_step}"
                    )
                elif latest_remote_step > latest_local_step:
                    # Remote is newer, download it
                    should_download = True
                    logging.info(
                        f"Rank 0 remote checkpoint is newer "
                        f"(step {latest_remote_step} > {latest_local_step}), will download"
                    )
                else:
                    # Local is up-to-date or newer
                    logging.info(
                        f"Rank 0 local checkpoint is up-to-date "
                        f"(step {latest_local_step} >= {latest_remote_step}), skipping download"
                    )

                # Step 4: Download if needed
                if should_download:
                    local_checkpoint_path = local_checkpoint_dir / checkpoint_name
                    logging.info(f"Rank 0 downloading to shared Lustre: {local_checkpoint_path}")

                    success = download_remote_checkpoint_for_resume(
                        remote_checkpoint_url=latest_checkpoint_url,
                        local_checkpoint_dir=local_checkpoint_path,
                    )

                    if success:
                        logging.info(
                            "Rank 0 ✓ checkpoint download successful, "
                            f"training will resume from step {latest_remote_step}"
                        )
                    else:
                        logging.warning(
                            "Rank 0 ✗ checkpoint download failed, will use local checkpoint or start from scratch"
                        )
            else:
                logging.info(f"Rank 0 no remote checkpoint found in {msc_checkpoint_url}")
                if latest_local_valid:
                    logging.info(f"Rank 0 will use local checkpoint-{latest_local_step}")
                else:
                    logging.info("Rank 0 starting from scratch")

        except Exception as e:  # noqa: BLE001
            logging.warning(f"Rank 0 error during checkpoint handling: {e}")
            if latest_local_valid:
                logging.info(f"Rank 0 will use local checkpoint-{latest_local_step}")
            else:
                logging.info("Rank 0 starting from scratch")
    else:
        # Other ranks: Wait for rank 0 to complete checkpoint resolution
        logging.info(f"Rank {global_rank} waiting for rank 0 to resolve checkpoint")

    # Synchronize all ranks using barrier
    if dist.is_initialized():
        dist.barrier()
        if global_rank != 0:
            logging.info(
                f"Rank {global_rank} checkpoint resolution complete, proceeding with training"
            )


def _build_in_process_open_loop_callback(
    *,
    in_process_cfg,
    data_cfg,
    processor,
    action_horizon: int,
    video_backend: str,
    trainer,
):
    """Build the in-process open-loop eval callback or return None.

    Returns ``None`` when ``in_process_cfg.enabled`` is False or no eval
    specs resolve. Otherwise constructs the callback, sets
    ``cb.trainer = trainer`` so it can route metrics through HF logging,
    and returns it. The caller is responsible for ``trainer.add_callback``.

    Extracted from inline wiring so the test
    ``tests/groot/core/training/experiment/test_experiment_open_loop_wiring.py``
    can call it directly with stub fixtures (no full experiment context).
    """
    if not in_process_cfg.enabled:
        return None

    # Lazy import — pulls in matplotlib, torchcodec, etc.
    from groot.core.training.open_loop.in_process_eval_callback import (
        EvalDatasetSpec,
        InProcessOpenLoopEvalCallback,
    )

    if not data_cfg.datasets:
        raise ValueError(
            "Cannot enable in-process open-loop evaluation without "
            "training datasets in DataConfig.datasets."
        )

    in_proc_specs: list[EvalDatasetSpec] = []
    ds_filter = in_process_cfg.dataset_indices
    for idx, dataset_config in enumerate(data_cfg.datasets):
        if ds_filter is not None and idx not in ds_filter:
            continue
        embodiment_tag = dataset_config.embodiment_tag
        if not embodiment_tag:
            logging.warning(
                f"[in-process open-loop eval] dataset idx={idx} missing "
                "embodiment_tag — skipping."
            )
            continue

        train_path = str(dataset_config.dataset_paths[0]) if dataset_config.dataset_paths else None
        val_path = dataset_config.val_dataset_path

        if val_path:
            in_proc_specs.append(
                EvalDatasetSpec(
                    name=f"ds{idx}",
                    dataset_path=str(val_path),
                    embodiment_tag=embodiment_tag,
                    eval_type="val",
                )
            )
        if (in_process_cfg.use_train_path or not val_path) and train_path:
            in_proc_specs.append(
                EvalDatasetSpec(
                    name=f"ds{idx}",
                    dataset_path=train_path,
                    embodiment_tag=embodiment_tag,
                    eval_type="train",
                )
            )

    if not in_proc_specs:
        logging.warning(
            "[in-process open-loop eval] enabled but no eval datasets resolved "
            "(check DataConfig.datasets / val_dataset_path)."
        )
        return None

    logging.info(
        "[in-process open-loop eval] enabled — every %d steps, %d eval target(s):",
        in_process_cfg.interval_steps,
        len(in_proc_specs),
    )
    for spec in in_proc_specs:
        logging.info(
            "  - %s [%s/%s]: %s",
            spec.name,
            spec.embodiment_tag,
            spec.eval_type,
            spec.dataset_path,
        )

    cb = InProcessOpenLoopEvalCallback(
        eval_datasets=in_proc_specs,
        processor=processor,
        traj_ids=in_process_cfg.traj_ids,
        steps_per_traj=in_process_cfg.steps_per_traj,
        action_horizon=action_horizon,
        eval_steps=in_process_cfg.interval_steps,
        eval_at_start=in_process_cfg.eval_at_start,
        save_plot_dir=in_process_cfg.save_plot_dir,
        video_backend=video_backend,
        num_inference_timesteps=in_process_cfg.num_inference_timesteps,
        sample_mode=in_process_cfg.sample_mode,
        num_random_samples=in_process_cfg.num_random_samples,
        random_seed=in_process_cfg.random_seed,
        enable_frechet=in_process_cfg.enable_frechet,
        enable_reverse_kl=in_process_cfg.enable_reverse_kl,
        rkl_num_samples=in_process_cfg.rkl_num_samples,
    )
    cb.trainer = trainer
    return cb


def _build_open_loop_eval_callback(
    *,
    legacy_cfg,
    data_cfg,
    action_horizon: int,
    video_backend: str,
    trainer,
):
    """Build the legacy on-checkpoint-save open-loop eval callback or return None.

    Returns ``None`` when ``legacy_cfg.enabled`` is False or no eval configs
    resolve (no dataset whose embodiment is in the hardcoded ``eval_embodiments``
    set). Otherwise returns ``OpenLoopEvalCallback`` ready for
    ``trainer.add_callback``.
    """
    if not legacy_cfg.enabled:
        return None

    logging.info("Open-loop evaluation enabled")
    logging.info(f"   Policy type: {legacy_cfg.policy_type}")

    if not data_cfg.datasets:
        raise ValueError("Cannot enable open-loop evaluation without training datasets.")

    eval_embodiments = {
        "agibot",
        "xdof",
        "xdof_dream",
        "sim_behavior_r1_pro",
        "agibot_dream",
        "oxe_droid_dream",
    }
    logging.info(f"   Evaluating only on embodiments: {eval_embodiments}")

    eval_configs: list[dict] = []
    eval_dataset_index = legacy_cfg.dataset_index
    if eval_dataset_index is not None:
        logging.info(f"   Filtering to dataset index: {eval_dataset_index}")

    for idx, dataset_config in enumerate(data_cfg.datasets):
        embodiment_tag = dataset_config.embodiment_tag
        if not embodiment_tag:
            raise ValueError(f"Dataset {dataset_config.dataset_paths} missing embodiment_tag.")
        if embodiment_tag not in eval_embodiments:
            continue
        if eval_dataset_index is not None and idx != eval_dataset_index:
            continue

        train_dataset_path = dataset_config.dataset_paths[0]
        eval_configs.append(
            {
                "dataset_path": train_dataset_path,
                "embodiment_tag": embodiment_tag,
                "eval_type": "train",
            }
        )
        val_dataset_path = dataset_config.val_dataset_path or train_dataset_path
        eval_configs.append(
            {"dataset_path": val_dataset_path, "embodiment_tag": embodiment_tag, "eval_type": "val"}
        )

    if not eval_configs:
        logging.warning(
            "[open-loop eval] enabled but no eval configs resolved "
            "(no dataset with embodiment_tag in the legacy eval_embodiments set)."
        )
        return None

    logging.info(f"   Trajectory IDs: {legacy_cfg.traj_ids}")
    logging.info(f"   Steps per trajectory: {legacy_cfg.steps_per_traj}")
    logging.info(f"   Evaluation configs ({len(eval_configs)} total):")
    for eval_config in eval_configs:
        logging.info(
            f"     - {eval_config['embodiment_tag']}/{eval_config['eval_type']}: "
            f"{eval_config['dataset_path']}"
        )

    return OpenLoopEvalCallback(
        eval_configs=eval_configs,
        traj_ids=legacy_cfg.traj_ids,
        action_horizon=action_horizon,
        steps_per_traj=legacy_cfg.steps_per_traj,
        plot_indices=legacy_cfg.plot_indices,
        video_backend=video_backend,
        policy_type=legacy_cfg.policy_type,
        trainer=trainer,
    )


def run(
    config: Config,
    dataset_build_artifacts: DatasetBuildArtifacts | None = None,
):
    warn_configs(config)

    """Main training function."""
    # If using distributed training, initialize the process group
    if dist.is_initialized():
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Explicitly set CUDA device for the worker. For Ray with 1 GPU per worker,
        # local_rank is always 0 (single visible GPU), but this is harmless and ensures
        # correct device binding regardless of the distributed setup.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logging.info(f"Rank {global_rank}: CUDA device set to {torch.cuda.current_device()}")
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        # Backend choice keyed off the launcher's declared intent
        # (config.training.num_gpus) rather than torch.cuda.is_available():
        #   num_gpus > 0  -> "this is a GPU run, use NCCL" — applies even to
        #                    the GPU dry-run with IdentityModel, where the
        #                    model has 1 param but we still want NCCL
        #                    collectives for symmetry with real training.
        #   num_gpus == 0 -> CPU dry-run (Ray TorchTrainer with no GPU
        #                    bundles); fall through to Gloo.
        # DCP async checkpoint uses a dedicated Gloo process group (created
        # in GR00TTrainer), so the default backend can stay NCCL-only.
        backend = _resolve_torchrun_backend(config)
        if config.training.num_gpus > 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        distributed_timeout_minutes = int(os.environ.get("GROOT_DISTRIBUTED_TIMEOUT_MINUTES", "25"))
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=distributed_timeout_minutes),
        )  # relative stats calculation and synchronous checkpoint upload may take a while before barriers
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    # Setup
    setup_logging()
    set_seed(config.data.seed)

    # Resolve dataset paths using LFS_CACHE_PATH environment variable
    # This must happen AFTER cache download to avoid double slash in S3 URLs
    resolve_all_dataset_paths(config)
    config.data.materialize_deltacat_as_of_timestamps()

    # Validate config
    config.validate()
    if dataset_build_artifacts is None:
        dataset_build_artifacts = DatasetBuildArtifacts(
            deltacat_stats_by_embodiment=deltacat_sampled_stats.compute_from_config(
                config.data,
                model_config=config.model,
                allow_ray=False,
            )
        )

    preserve_checkpoint_root = preserve_checkpoint_root_enabled()
    api_checkpoint_path = os.environ.get("GEAR_CHECKPOINT_PATH", "")
    api_msc_checkpoint_url = (
        ""
        if preserve_checkpoint_root
        else dashboard_checkpoint_path_to_msc_url(api_checkpoint_path)
    )
    configured_msc_checkpoint_url = getattr(config.training, "msc_checkpoint_url", None)
    if configured_msc_checkpoint_url and api_msc_checkpoint_url:
        logging.info(
            "Overriding training.msc_checkpoint_url on worker from %s to dashboard-allocated %s",
            configured_msc_checkpoint_url,
            api_msc_checkpoint_url,
        )
        config.training.msc_checkpoint_url = api_msc_checkpoint_url

    resolved_output_dir = resolve_training_output_dir(config.training.output_dir)
    if resolved_output_dir != config.training.output_dir:
        logging.info(
            "Resolved canonical training.output_dir on worker: %s",
            resolved_output_dir,
        )
        config.training.output_dir = resolved_output_dir
    if preserve_checkpoint_root:
        requested_msc_checkpoint_url = os.environ.get("TRAINING_MSC_CHECKPOINT_URL", "")
        if (
            requested_msc_checkpoint_url
            and config.training.msc_checkpoint_url != requested_msc_checkpoint_url
        ):
            logging.info(
                "Preserving launcher training.msc_checkpoint_url on worker: %s",
                requested_msc_checkpoint_url,
            )
            config.training.msc_checkpoint_url = requested_msc_checkpoint_url
    api_wandb_run_name = os.environ.get("GEAR_WANDB_RUN_NAME", "")

    # Create output directory and check if using remote MSC upload
    msc_checkpoint_url = getattr(config.training, "msc_checkpoint_url", None)
    msc_profile_url = getattr(config.training, "msc_profile_url", None)

    # Setup MSC config if either checkpoint or profile upload is enabled
    if msc_checkpoint_url or msc_profile_url:
        try:
            _setup_msc_config(config)
            if msc_checkpoint_url:
                logging.info(f"MSC checkpoint storage enabled: {msc_checkpoint_url}")
            if msc_profile_url:
                logging.info(f"MSC profile storage enabled: {msc_profile_url}")
        except Exception as e:
            if msc_checkpoint_url:
                # Checkpoint upload was explicitly requested — fail hard
                raise
            # Only profiling triggered MSC setup; disable upload gracefully
            logging.warning(
                f"MSC config setup failed ({e}). "
                f"Profiling will still run locally but MSC upload is disabled."
            )
            msc_profile_url = None

    # If submit.py registered the experiment, the dashboard returned three
    # different canonical strings:
    #   * checkpoint_path     = "GearCheckpoints/.../{ref}_{name}" -- the
    #     remote checkpoint root. Override config.training.msc_checkpoint_url
    #     so MSC uploads land at the same root that evals and dashboard lineage
    #     use.
    #   * lfs_checkpoint_path = "{lfs_prefix}/{ref}_{name}" -- the on-disk
    #     output directory. Override config.training.output_dir so on-disk
    #     writes and the dashboard record agree on a single Lustre path.
    #   * wandb_run_name      = "{name}" -- the human-friendly identifier we
    #     want on W&B and in saved configs. Crucially this is NOT the
    #     basename of lfs_checkpoint_path: lfs basename is "{ref}_{name}",
    #     so deriving experiment_name from output_dir.name (the legacy
    #     fallback below) produces "{ref}_{name}" and the W&B run ends up
    #     named "00042_my_job" instead of "my_job". Always prefer the
    #     server-supplied wandb_run_name when available.
    # Both fall back to the launcher-provided values when the env vars are
    # absent (local runs, dashboard outage, backfill mode).

    # Setup output directory (always use local Lustre path, not ephemeral storage)
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        # Server-supplied name beats output_dir basename so the W&B run name
        # is "my_job", not "00042_my_job".
        experiment_name = api_wandb_run_name or output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name
    # Write the resolved name back so it propagates to W&B config / saved YAML.
    config.training.experiment_name = experiment_name

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    wandb_config_file = output_dir / "wandb_config.json"

    # Read experiment IDs that the launcher may have injected via env vars
    # (submit.py forwards GEAR_EXPERIMENT_ID / GEAR_EXPERIMENT_REF to every Ray
    # worker). All ranks read them so downstream code on any rank sees the same
    # values on config.training, but only rank 0 writes to disk or calls the API.
    experiment_id = os.environ.get("GEAR_EXPERIMENT_ID", "")
    experiment_ref = os.environ.get("GEAR_EXPERIMENT_REF", "")
    training_run_id = os.environ.get("GEAR_TRAINING_RUN_ID", "")

    def _emit_run_event(
        status: str,
        *,
        event: str,
        message: str | None = None,
        metadata: dict | None = None,
        set_status: bool = True,
    ) -> None:
        """Append a training-run status event to the dashboard.

        No-op unless we are rank 0 AND the launcher pre-registered the run.
        Always graceful: registration helpers swallow network/HTTP errors.
        """
        if global_rank != 0 or not experiment_id or not training_run_id:
            return
        meta = {"event": event, **(metadata or {})}
        try:
            append_training_run_status_event(
                experiment_id,
                training_run_id,
                status=status,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message=message,
                metadata=meta,
                set_status=set_status,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Dashboard status-event append failed (non-fatal): %s", exc)

    # Capture start-of-post-registration time so any failure in the broader
    # try below (open-loop callback construction, msc-checkpoint fetch, …)
    # can still report a sensible duration. The "running" event is still
    # emitted only when Trainer.train() is entered (see _running_emitted).
    _train_started_at = datetime.now(timezone.utc)

    if global_rank == 0:
        # Create output directories (rank 0 is the sole writer).
        output_dir.mkdir(parents=True, exist_ok=True)
        save_cfg_dir.mkdir(parents=True, exist_ok=True)
        processor_dir.mkdir(parents=True, exist_ok=True)

        # Build the canonical dataset payload from the resolved training
        # config. The launcher didn't have this — it registered a skeleton
        # record.
        _datasets = [
            {
                "paths": [str(p) for p in ds.dataset_paths],
                "embodiment_tag": ds.embodiment_tag,
                "mix_ratio": ds.mix_ratio,
            }
            for ds in config.data.datasets
        ]

        if not experiment_id and os.environ.get("SKIP_EXPERIMENT_REGISTRATION", "0") != "1":
            # No launcher registration (local run, or launcher's API call
            # failed). Register from inside the training job as a fallback.
            # Dry-run / sim launchers set SKIP_EXPERIMENT_REGISTRATION=1 to
            # bypass this entirely and avoid junk dashboard entries; for
            # real runs that reach here, the guardrail decides whether to
            # actually register.
            should_register, reason = should_fallback_register_experiment(config)
            if should_register:
                try:
                    lfs_prefix = str(Path(config.training.output_dir).parent) or None
                    record = register_experiment(
                        experiment_name=experiment_name,
                        wandb_project=config.training.wandb_project,
                        lfs_prefix=lfs_prefix,
                        groot_commit=os.environ.get("GROOT_COMMIT_HASH", ""),
                        start_from_checkpoint=config.training.start_from_checkpoint,
                        datasets=_datasets,
                    )
                    experiment_id = record.get("experiment_id", "")
                    experiment_ref = record.get("experiment_ref", "")
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Experiment registration failed (non-fatal): %s", exc)
            else:
                logging.info("Skipping fallback experiment registration: %s", reason)

        config.training.experiment_id = experiment_id
        config.training.experiment_ref = experiment_ref
        config.training.training_run_id = training_run_id

        # Save config AFTER experiment IDs are assigned so the frozen YAML on
        # disk carries the experiment identity.
        config.save(save_cfg_dir / "config.yaml")
        # this is to be compatible with groot.vla.eval.sim.evaluator_daemon.py, which reads conf.yaml
        omegaconf_config = OmegaConf.create(config.__dict__)
        omegaconf_config["max_steps"] = config.training.max_steps
        omegaconf_config["save_steps"] = config.training.save_steps
        OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)

        # Write wandb_config.json (rank 0 only — prevents other ranks from
        # racing and overwriting with empty IDs when they don't participate in
        # registration).
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": config.training.wandb_project,
                    "run_id": experiment_name,
                    "experiment_name": experiment_name,
                    "experiment_id": experiment_id,
                    "experiment_ref": experiment_ref,
                    "training_run_id": training_run_id,
                    "groot_commit": os.environ.get("GROOT_COMMIT_HASH", ""),
                    "branch": os.environ.get("GROOT_BRANCH", ""),
                },
                f,
            )

        logging.info(f"Saved config to {save_cfg_dir}")
    else:
        # Non-rank-0 workers still need the IDs on their local config for any
        # downstream code that inspects config.training.
        config.training.experiment_id = experiment_id
        config.training.experiment_ref = experiment_ref
        config.training.training_run_id = training_run_id

    # Initialize wandb if configured, but only on the main process
    if config.training.use_wandb and global_rank == 0:
        # Add git commit hash and version info to config
        config_dict = {
            **config.__dict__,
            "experiment_id": experiment_id,
            "experiment_ref": experiment_ref,
            "training_run_id": training_run_id,
            "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
            "groot_version": VERSION,
        }

        _wandb_tags = [config.data.mode]
        if experiment_ref:
            _wandb_tags.append(f"exp_ref:{experiment_ref}")
        if experiment_id:
            _wandb_tags.append(f"exp_id:{experiment_id}")

        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config=config_dict,
            tags=_wandb_tags,
        )

        # Mirror W&B identifiers onto the training run so retries stay unambiguous.
        try:
            run = wandb.run
            wandb_run_id = getattr(run, "id", "") if run is not None else ""
            wandb_run_url = getattr(run, "url", "") if run is not None else ""
            if wandb_run_id and experiment_id and training_run_id:
                update_training_run(
                    experiment_id,
                    training_run_id,
                    wandb_run_id=wandb_run_id,
                    wandb_run_url=wandb_run_url,
                )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Dashboard W&B-metadata PATCH failed (non-fatal): %s", exc)

    # Track whether we already emitted "running" so the failure-path event
    # name distinguishes setup-time crashes (training_setup_failed) from
    # crashes inside Trainer.train() (training_failed). Without this the
    # dashboard could leave the run stuck in an earlier state (e.g.
    # ray_submitted) when post-registration setup blows up.
    _running_emitted = False

    # Wrap ALL post-registration setup — model pipeline, trainer construction,
    # every callback build, msc-checkpoint fetch — in this try, not just
    # Trainer.train() and the open-loop callback work. Any crash here moves
    # the dashboard run to "failed" via the except branch below instead of
    # leaving it stuck in an earlier state.
    try:
        if global_rank == 0:
            sys.stdout.write(
                "OL_PHASE name=model_loading status=started message=building model and trainer\n"
            )
            sys.stdout.flush()

        _materialize_model_source_checkpoint(config, global_rank)

        # Configure structlog BEFORE pipeline.setup() so events emitted from
        # inside `factory.build()` (e.g., V2 grouped factory's
        # `step_priority_mapping`) propagate. Without this call, structlog
        # is still at the import-time WARNING filter and all info/debug
        # events from the factory build are silently dropped. Trainer's
        # own configure_logging in __init__ runs later, after .build() has
        # already executed.
        if config.training.enable_struct_logs and config.training.training_log_dir:
            os.makedirs(config.training.training_log_dir, exist_ok=True)
            from groot.core.utils.distributed import worker_logging

            worker_logging.configure_logging(
                worker_type="training_actor",
                level=config.training.training_logging_level,
                log_file=f"{config.training.training_log_dir}/training_actor_{global_rank}.jsonl",
                force=True,
            )

        # Setup model training pipeline. When the launch_identity_*_dry_run
        # launchers set _GROOT_DRY_RUN_STUB_MODEL=1, shadow _create_model on
        # the constructed pipeline instance so .setup() skips the cuda-bound
        # model constructor (WanVideoVAE etc.) and yields an IdentityModel;
        # the rest of pipeline.setup() — including _create_dataset, which
        # builds the real processor + collator — runs unchanged. Done here,
        # after torch.cuda.set_device(local_rank), to avoid the "Duplicate
        # GPU detected" NCCL error you'd hit if model imports forced
        # torch.cuda init before the per-rank device was bound (Ray's
        # worker_process_setup_hook fires too early for that).
        pipeline = MODEL_REGISTRY.get(type(config.model))(
            config,
            save_cfg_dir,
            dataset_build_artifacts=dataset_build_artifacts,
        )
        if os.environ.get("_GROOT_DRY_RUN_STUB_MODEL") == "1":
            pipeline._create_model = lambda: IdentityModel()  # noqa: SLF001
        pipeline.setup()
        model = pipeline.return_model()
        train_dataset, eval_dataset = pipeline.return_dataset()
        data_collator = pipeline.return_collator()
        processor = pipeline.return_processor()
        processor.save_pretrained(processor_dir)

        # deepspeed config
        if config.training.num_gpus > 1 and not config.training.use_ddp:
            deepspeed_config = config.get_deepspeed_config()
        else:
            deepspeed_config = None

        # per_device_batch_size is the single source of truth.
        # Effective global batch = num_gpus * per_device_batch_size * gradient_accumulation_steps.
        assert (
            config.training.per_device_batch_size is not None
        ), "training.per_device_batch_size must be set."
        per_device_train_batch_size = config.training.per_device_batch_size

        # these training settings require sync_seq_len
        sync_seq_len = config.training.deepspeed_stage == 3 or config.training.use_truncated_bptt

        # WSD scheduler: "wsd" is not a valid HF lr_scheduler_type, so we intercept it
        # and pass a dummy value to avoid validation errors.
        use_wsd = config.training.lr_scheduler_type == "wsd"
        hf_lr_scheduler_type = "cosine" if use_wsd else config.training.lr_scheduler_type

        # Create training arguments
        # CPU-only path (launch_identity_cpu_dry_run): force HF Trainer's
        # device picking to "cpu" and skip its `is_torch_xla_available()`
        # → `is_cuda_available()` autodetect chain. Required even when
        # CUDA_VISIBLE_DEVICES="" because the Ray worker may still see
        # nvidia drivers in /dev and pick the wrong device class.
        # NOTE: HF Trainer raises if use_cpu=True is set with
        # bf16=True or fp16=True — keep --training.no-bf16 /
        # no-fp16 / deepspeed_stage 0 on the launch script.
        use_cpu = config.training.num_gpus == 0

        training_args = OmniTrainingArguments(
            output_dir=str(output_dir),
            use_cpu=use_cpu,
            max_steps=config.training.max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=config.training.eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=hf_lr_scheduler_type,
            weight_decay=config.training.weight_decay,
            adam_beta1=config.training.adam_beta1,
            adam_beta2=config.training.adam_beta2,
            adam_epsilon=config.training.adam_epsilon,
            warmup_ratio=config.training.warmup_ratio,
            max_grad_norm=config.training.max_grad_norm,
            logging_steps=config.training.logging_steps,
            log_distributed_aux_losses=config.training.log_distributed_aux_losses,
            enable_nan_loss_diagnostics=config.training.enable_nan_loss_diagnostics,
            nan_loss_diagnostics_dir=config.training.nan_loss_diagnostics_dir,
            nan_loss_diagnostics_max_dumps=config.training.nan_loss_diagnostics_max_dumps,
            log_distributed_grad_norm=config.training.log_distributed_grad_norm,
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            save_strategy="no" if not config.training.save_final_model else "steps",
            fp16=config.training.fp16,
            bf16=config.training.bf16,
            tf32=config.training.tf32,
            torch_compile=config.training.torch_compile,
            torch_compile_mode=(
                config.training.torch_compile_mode if config.training.torch_compile else None
            ),
            gradient_checkpointing=config.training.gradient_checkpointing,
            optim=config.training.optim,
            dataloader_num_workers=config.training.dataloader_num_workers,
            dataloader_pin_memory=config.training.dataloader_pin_memory,
            report_to="wandb" if config.training.use_wandb else "none",
            seed=config.data.seed,
            deepspeed=deepspeed_config,
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
            use_muon=config.training.use_muon,
            muon_impl=config.training.muon_impl,
            muon_lr=config.training.muon_lr,
            eval_strategy="no",  # val loss is handled by ValidationLossCallback
            batch_eval_metrics=True,
            remove_unused_columns=config.training.remove_unused_columns,
            use_legacy_wd_application=config.training.use_legacy_wd_application,
            ignore_data_skip=True,
            sync_seq_len=sync_seq_len,
            use_truncated_bptt=config.training.use_truncated_bptt,
            truncated_bptt_chunk_size=config.training.truncated_bptt_chunk_size,
            _use_wsd_scheduler=use_wsd,
            wsd_stable_ratio=config.training.wsd_stable_ratio,
            wsd_decay_type=config.training.wsd_decay_type,
            use_async_checkpoint_save=getattr(config.training, "use_async_checkpoint_save", True),
        )

        if hasattr(config.model, "collator_seq_len_handle_strategy"):
            training_args.seq_len_handle_strategy = config.model.collator_seq_len_handle_strategy

        assert training_args.muon_impl in (
            "dist",
            "single_batched",
        ), "muon_impl must be 'dist' or 'single_batched'"

        # Create trainer
        trainer_class = pipeline.get_trainer_class(GR00TTrainer)
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            multiprocessing_context=config.data.multiprocessing_context,
            straggler_config=config.training.straggler,
            data_loading_enable_struct_logs=config.data.enable_struct_logs,
            data_loading_logging_level=config.data.data_loading_logging_level,
            data_loading_log_dir=config.data.data_loading_log_dir,
            dataloader_prefetch_factor=config.data.dataloader_prefetch_factor,
            training_enable_struct_logs=config.training.enable_struct_logs,
            training_logging_level=config.training.training_logging_level,
            training_log_dir=config.training.training_log_dir,
            **pipeline.trainer_init_kwargs(),
        )

        # Override save_steps after resume: HuggingFace Trainer restores save_steps from
        # trainer_state.json during resume, overriding CLI args. This callback re-applies
        # the config value so CLI args always take precedence over checkpoint state.
        class _OverrideSaveStepsCallback(TrainerCallback):
            def on_train_begin(self, args, state, _control, **_kwargs):
                if args.save_steps != config.training.save_steps:
                    logging.info(
                        f"Overriding save_steps from checkpoint state ({args.save_steps}) "
                        f"with config value ({config.training.save_steps})"
                    )
                    args.save_steps = config.training.save_steps
                    state.save_steps = config.training.save_steps

        trainer.add_callback(_OverrideSaveStepsCallback())
        trainer.add_callback(OLTrainingProgressCallback(global_rank=global_rank))
        for callback in pipeline.create_trainer_callbacks(trainer):
            trainer.add_callback(callback)

        model_type = getattr(config.model, "model_type", type(config.model).__name__)
        if model_type in {"GrootN2", "GrootN2TTT", "GrootN2Nano"}:
            n2_compile_diagnostics = N2CompileDiagnosticsCallback.from_env()
            if n2_compile_diagnostics.enabled:
                trainer.add_callback(n2_compile_diagnostics)

        # Add per-dataset validation loss callback if eval_strategy was "steps"
        if config.training.eval_strategy == "steps":
            val_datasets = _ValDatasetFactory(
                config.data,
                experiment_name=config.training.experiment_name,
            ).build_val_datasets(processor)
            if val_datasets:
                val_callback = ValidationLossCallback(
                    val_datasets=val_datasets,
                    processor=processor,
                    data_collator=data_collator,
                    eval_steps=config.training.eval_steps,
                    val_total_samples=config.training.val_total_samples,
                    eval_batch_size=config.training.eval_batch_size,
                    val_loss_seed=config.training.val_loss_seed,
                )
                val_callback.trainer = trainer
                trainer.add_callback(val_callback)
            else:
                logging.warning(
                    "eval_strategy='steps' but no val_dataset_path configured. "
                    "Skipping validation loss callback."
                )

        if config.training.add_rl_callback:
            trainer.add_callback(RLCallback())

        if getattr(config.model, "freeze_backbone_steps", 0) > 0:
            trainer.add_callback(UnfreezeBackboneCallback(model=model))

        # Add straggler detection callback if enabled
        straggler_cfg = config.training.straggler
        if straggler_cfg.enabled:
            logging.info("Straggler detection enabled")
            trainer.add_callback(
                StragglerDetectionCallback(
                    report_time_interval=straggler_cfg.report_time_interval_sec,
                    scores_to_compute=straggler_cfg.scores_to_compute,
                    gather_on_rank0=straggler_cfg.gather_on_rank0,
                    profiling_interval=straggler_cfg.profiling_interval,
                    perf_threshold=straggler_cfg.perf_threshold,
                )
            )

        # Conditionally add SaveVlModelCallback based on config
        if config.training.save_vl_model:
            trainer.add_callback(SaveVlModelCallback(processor=processor))

        trainer.add_callback(
            CheckpointFormatCallback(
                run_name=experiment_name,
                exp_cfg_dir=save_cfg_dir,
                processor_dir=processor_dir,
            )
        )

        # Add checkpoint upload callback
        if msc_checkpoint_url:
            # Use MSC for direct remote storage upload
            # Note: MSCCheckpointCallback uploads every checkpoint (controlled by save_steps),
            # unlike CheckpointUploadCallback which only selectively upload a subset of checkpoints.

            # "local" = sequential upload, "ray" = distributed upload using Ray workers
            execution_mode = getattr(config.training, "msc_execution_mode", "local")
            # "async" = background uploads, "sync" = blocking uploads
            callback_mode = getattr(config.training, "msc_callback_mode", "async") == "sync"
            delete_checkpoint_after_upload = getattr(
                config.training, "msc_delete_checkpoint_after_upload", True
            )
            msc_callback = MSCCheckpointCallback(
                remote_checkpoint_dir=msc_checkpoint_url,  # Remote MSC/S3 URL
                local_checkpoint_dir=output_dir,  # Local shared directory where checkpoints are saved
                exp_cfg_dir=save_cfg_dir,
                wandb_config_file=wandb_config_file,
                max_concurrent_uploads=config.training.max_concurrent_uploads,
                sync_mode=callback_mode,
                delete_after_upload=delete_checkpoint_after_upload,
                execution_mode=execution_mode,
                max_retries=getattr(config.training, "msc_max_retries", 3),
            )
            msc_callback.trainer = trainer
            trainer.add_callback(msc_callback)
            logging.info(
                f"Using MSCCheckpointCallback: local={output_dir}, remote={msc_checkpoint_url}, "
                f"callback_mode={callback_mode}, execution_mode={execution_mode}"
            )
        elif config.training.upload_checkpoints:
            # Use traditional CheckpointUploadCallback for Lustre->S3 workflow
            upload_callback = CheckpointUploadCallback(
                run_name=experiment_name,
                exp_cfg_dir=save_cfg_dir,
                pdx_checkpoint_root=api_checkpoint_path if not preserve_checkpoint_root else None,
                upload_every=config.training.upload_every,
                last_n_checkpoints=config.training.upload_last_n_checkpoints,
                max_concurrent_uploads=config.training.max_concurrent_uploads,
            )
            upload_callback.trainer = trainer
            trainer.add_callback(upload_callback)
        if config.training.save_best_eval_metric_name != "":
            trainer.add_callback(
                BestMetricCheckpointCallback(
                    metric_name=config.training.save_best_eval_metric_name,
                    greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                    exp_cfg_dir=save_cfg_dir,
                )
            )

        # In-process open-loop evaluation (groot.core only — no checkpoint reload).
        # action_horizon lives at different paths per model family:
        #   N1.x: config.model.action_horizon (direct attr)
        #   N2:   config.model.action_cfg["action_horizon"] (TypedDict on action_cfg)
        _model_action_horizon = getattr(config.model, "action_horizon", None) or (
            getattr(config.model, "action_cfg", {}) or {}
        ).get("action_horizon")
        in_proc_cb = _build_in_process_open_loop_callback(
            in_process_cfg=config.training.open_loop_eval_in_process,
            data_cfg=config.data,
            processor=processor,
            action_horizon=(
                config.training.open_loop_eval_in_process.action_horizon or _model_action_horizon
            ),
            video_backend=config.data.video_backend,
            trainer=trainer,
        )
        if in_proc_cb is not None:
            trainer.add_callback(in_proc_cb)

        # Open-loop evaluation (requires groot.control at runtime — lazy imports)
        legacy_cb = _build_open_loop_eval_callback(
            legacy_cfg=config.training.open_loop_eval,
            data_cfg=config.data,
            action_horizon=_model_action_horizon,
            video_backend=config.data.video_backend,
            trainer=trainer,
        )
        if legacy_cb is not None:
            trainer.add_callback(legacy_cb)

        if hasattr(train_dataset, "get_initial_actions"):
            initial_actions = train_dataset.get_initial_actions()
            if initial_actions:
                initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
                save_initial_actions(initial_actions, initial_actions_path)
                logging.info(
                    f"Saved {len(initial_actions)} initial actions to {initial_actions_path}"
                )

        # Resume always reads from the local Lustre checkpoint dir
        # (`output_dir`). The pre-download from MSC was added in
        # 95bdcb9c88 (core open loop eval) and caused issues — it conflated
        # "resume" with "fetch newer checkpoint from remote storage", which
        # is unexpected when Lustre already has the run's own checkpoints.
        # `trainer.train(resume_from_checkpoint=True)` below falls through
        # to `get_last_checkpoint(args.output_dir)` and resumes from Lustre
        # directly. MSC is still used for uploads (downstream) — only the
        # download-before-resume step is dropped here.
        #
        # If you ever need the MSC pre-download back, call
        # `fetch_latest_checkpoint_from_remote_if_available(output_dir,
        #  msc_checkpoint_url, global_rank)` here (the helper is still
        # defined above for that purpose).

        # Train
        logging.info("🚀 Starting training...")
        _emit_run_event(
            "running",
            event="training_started",
            message="Trainer.train() entered",
            set_status=True,
        )
        _running_emitted = True

        if config.training.enable_profiling:
            logging.info(f"Rank {global_rank}: Starting training with profiling...")

            # Create unique job ID for organizing profiles across different training runs
            # Format: {experiment_name}_{timestamp}
            # Broadcast timestamp from rank 0 so all ranks use the same profiling_job_id.
            # Use a temporary Gloo process group instead of the default NCCL group.
            # NCCL broadcasts can hang if channels haven't fully initialized yet
            # (e.g. when stats generation completes quickly and the broadcast fires
            # before NCCL channel setup is done on all ranks).
            if dist.is_initialized():
                gloo_group = dist.new_group(backend="gloo")
                timestamp_tensor = torch.tensor([int(time.time())], dtype=torch.long)
                dist.broadcast(timestamp_tensor, src=0, group=gloo_group)
                dist.destroy_process_group(gloo_group)
                profiling_timestamp = int(timestamp_tensor.item())
            else:
                profiling_timestamp = int(time.time())
            profiling_job_id = f"{experiment_name}_{profiling_timestamp}"

            # Write to Lustre first (existing behavior)
            lustre_profile_dir = output_dir / "profiling" / profiling_job_id / f"rank_{global_rank}"
            lustre_profile_dir.mkdir(parents=True, exist_ok=True)

            logging.info(
                f"Rank {global_rank}: Profiler traces will be saved to {lustre_profile_dir}"
            )
            logging.info(f"Rank {global_rank}: Profiling job ID: {profiling_job_id}")

            # Get profiling configuration
            profile_start_step = getattr(config.training, "profile_start_step", 50)
            profile_warmup_steps = getattr(config.training, "profile_warmup_steps", 1)
            profile_active_steps = getattr(config.training, "profile_active_steps", 5)
            profile_delete_after_upload = getattr(
                config.training, "profile_delete_after_upload", True
            )
            profile_record_shapes = getattr(config.training, "profile_record_shapes", False)
            profile_with_stack = getattr(config.training, "profile_with_stack", False)
            profile_memory = getattr(config.training, "profile_memory", False)

            # Calculate end step
            profile_end_step = profile_start_step + profile_warmup_steps + profile_active_steps

            logging.info(
                f"Rank {global_rank}: Profiling enabled - will collect traces for session steps "
                f"{profile_start_step} to {profile_end_step} "
                f"({profile_warmup_steps + profile_active_steps} steps total)"
            )

            # Define upload function that will be called after profiling completes
            def upload_profile_async():
                """Background thread to upload profile and clean up."""
                if not msc_profile_url:
                    return

                try:
                    remote_profile_url = (
                        f"{msc_profile_url}/profiling/{profiling_job_id}/rank_{global_rank}"
                    )

                    # Check if directory exists and has files
                    if not lustre_profile_dir.exists():
                        logging.error(
                            f"Rank {global_rank}: Profile directory does not exist: {lustre_profile_dir}"
                        )
                        return

                    trace_files = list(lustre_profile_dir.glob("*.pt.trace.json*"))
                    if not trace_files:
                        logging.warning(
                            f"Rank {global_rank}: No trace files found in {lustre_profile_dir}. "
                            f"Profiling may have failed to capture traces."
                        )
                        return

                    logging.info(
                        f"Rank {global_rank}: Uploading {len(trace_files)} trace file(s) to {remote_profile_url}"
                    )
                    upload_start = time.time()

                    # Use upload_dir wrapper from msc_utils
                    upload_dir(
                        local_path=str(lustre_profile_dir),
                        remote_url=remote_profile_url,
                    )

                    upload_duration = time.time() - upload_start
                    logging.info(
                        f"Rank {global_rank}: Successfully uploaded profiler traces to S3 "
                        f"in {upload_duration:.2f}s"
                    )
                    logging.info(
                        f"Rank {global_rank}: Remote profile location: {remote_profile_url}"
                    )
                    # Surface the profile only after rank 0 has uploaded a real trace.
                    update_training_run_profile_job_id(
                        global_rank=global_rank,
                        experiment_id=experiment_id,
                        training_run_id=training_run_id,
                        profile_job_id=profiling_job_id,
                    )

                    # Clean up local profiles after successful upload
                    if profile_delete_after_upload:
                        try:
                            shutil.rmtree(lustre_profile_dir)
                            logging.info(
                                f"Rank {global_rank}: ✓ Cleaned up local traces at {lustre_profile_dir}"
                            )
                        except Exception as cleanup_error:  # noqa: BLE001
                            logging.warning(
                                f"Rank {global_rank}: Failed to clean up local traces: {cleanup_error}"
                            )

                except Exception:
                    logging.exception(
                        "Rank %s: Failed to upload profiler traces to S3", global_rank
                    )
                    logging.warning(
                        f"Rank {global_rank}: Profiler traces remain available on Lustre at "
                        f"{lustre_profile_dir}"
                    )

            # Store upload thread reference so we can wait for it to complete
            upload_thread_ref = [None]

            def trigger_upload():
                """Wrapper to start upload in background thread with staggered delay."""

                def upload_with_stagger():
                    # Stagger uploads by rank to avoid network/IO congestion
                    # Each rank waits rank * 0.5 seconds before starting
                    stagger_delay = global_rank * 0.5
                    if stagger_delay > 0:
                        logging.info(
                            f"Rank {global_rank}: Waiting {stagger_delay:.1f}s before upload to avoid congestion"
                        )
                        time.sleep(stagger_delay)
                    upload_profile_async()

                upload_thread = threading.Thread(target=upload_with_stagger, daemon=False)
                upload_thread.start()
                upload_thread_ref[0] = upload_thread

            # Add profiling callback that will dynamically start/stop profiler during training
            # This avoids wrapping the entire training in profiler context which adds overhead
            # The callback will remove itself after profiling completes to ensure zero overhead
            trainer.add_callback(
                ProfCallback(
                    profile_dir=lustre_profile_dir,
                    upload_callback=trigger_upload if msc_profile_url else None,
                    profile_start_step=profile_start_step,
                    profile_end_step=profile_end_step,
                    warmup_steps=profile_warmup_steps,
                    active_steps=profile_active_steps,
                    trainer=trainer,
                    record_shapes=profile_record_shapes,
                    with_stack=profile_with_stack,
                    profile_memory=profile_memory,
                )
            )
            trainer.train(resume_from_checkpoint=one_launcher_resume_checkpoint())

            # Wait for upload thread to complete before exiting
            if msc_profile_url and upload_thread_ref[0] is not None:
                logging.info(f"Rank {global_rank}: Waiting for upload to complete...")
                upload_thread_ref[0].join(timeout=300)
                if upload_thread_ref[0].is_alive():
                    logging.warning(
                        f"Rank {global_rank}: Upload timeout - check {lustre_profile_dir}"
                    )
        else:
            # No profiling - just train normally
            trainer.train(resume_from_checkpoint=one_launcher_resume_checkpoint())

        # Save final model
        if config.training.save_final_model:
            trainer.save_model()
            logging.info(f"Model saved to {output_dir}")

        if config.training.assert_loss_less_than is not None:
            final_loss = trainer.loss
            if final_loss.item() > config.training.assert_loss_less_than:
                raise AssertionError(
                    f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
                )

        # Cleanup
        if hasattr(pipeline, "stop"):
            pipeline.stop()
        if hasattr(train_dataset, "close"):
            train_dataset.close()
        logging.info("Training completed!")
    except Exception as _train_exc:
        # _running_emitted distinguishes failures inside Trainer.train()
        # ("training_failed") from failures during post-registration setup
        # — open-loop callback construction, msc-checkpoint fetch, etc. —
        # which would otherwise leave the dashboard run stuck in an
        # earlier state ("training_setup_failed").
        _failure_phase = "training" if _running_emitted else "setup"
        logging.exception("Training failed during %s phase", _failure_phase)
        _emit_run_event(
            "failed",
            event="training_failed" if _running_emitted else "training_setup_failed",
            message=f"{type(_train_exc).__name__}: {_train_exc}",
            metadata={
                "exception_type": type(_train_exc).__name__,
                "exception_message": str(_train_exc),
                "phase": _failure_phase,
                "duration_seconds": int(
                    (datetime.now(timezone.utc) - _train_started_at).total_seconds()
                ),
            },
            set_status=True,
        )
        raise
    else:
        _emit_run_event(
            "completed",
            event="training_completed",
            message="Trainer.train() returned and final save succeeded",
            metadata={
                "duration_seconds": int(
                    (datetime.now(timezone.utc) - _train_started_at).total_seconds()
                ),
            },
            set_status=True,
        )
