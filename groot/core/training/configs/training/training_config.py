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
import dataclasses
from dataclasses import dataclass

from groot.core.training.configs.training.open_loop_eval_config import (
    OpenLoopEvalConfig,
    OpenLoopEvalInProcessConfig,
)


@dataclass
class StragglerConfig:
    """Straggler detection configuration (nvidia_resiliency_ext).

    Monitors per-rank performance in distributed training to identify slow ranks.
    Enabled by default. Requires nvidia-resiliency-ext (declared in pyproject.toml).
    """

    enabled: bool = True
    report_time_interval_sec: int = 60

    # Scores to compute:
    # - relative_perf_scores: rank performance relative to other ranks (cross-rank comparison).
    # - individual_perf_scores: rank performance relative to its own history (self-comparison).
    scores_to_compute: list[str] = dataclasses.field(
        default_factory=lambda: ["relative_perf_scores", "individual_perf_scores"]
    )

    gather_on_rank0: bool = True
    profiling_interval: int = 1
    profile_cuda: bool = False
    perf_threshold: float = 0.7


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Sequence model related
    use_truncated_bptt: bool = False
    truncated_bptt_chunk_size: int = 300

    # Output
    output_dir: str = "./outputs"
    experiment_name: str | None = None

    # Basic training
    max_steps: int = 30000  # this will override num_epochs
    # Per-device batch size. Effective global batch = num_gpus * per_device_batch_size.
    per_device_batch_size: int | None = None
    gradient_accumulation_steps: int = 1

    # Muon
    use_muon: bool = False
    muon_impl: str = "dist"  # "dist", "single_batched"
    muon_lr: float = 0.005

    # Legacy optimizer
    use_legacy_wd_application: bool = False

    # torch.compile
    torch_compile: bool = False
    torch_compile_mode: str = "default"

    # Optimization
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # this will override warmup_ratio
    max_grad_norm: float = 1.0

    # WSD (Warmup-Stable-Decay) scheduler options (used when lr_scheduler_type="wsd")
    wsd_stable_ratio: float = 0.8
    wsd_decay_type: str = "cosine"  # "cosine", "linear", or "sqrt"

    # Optimizer choice (huggingface TrainingArguments.optim)
    # Options include: 'adamw_torch', 'adamw_torch_fused', 'paged_adamw_32bit',
    # 'paged_adamw_8bit' (requires bitsandbytes), 'adafactor', etc.
    optim: str = "adamw_torch_fused"

    start_from_checkpoint: str | None = None

    # Mixed precision
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    log_distributed_aux_losses: bool = True
    enable_nan_loss_diagnostics: bool = False
    nan_loss_diagnostics_dir: str | None = None
    nan_loss_diagnostics_max_dumps: int = 4
    log_distributed_grad_norm: bool = True
    save_steps: int = 1000
    save_total_limit: int = 5

    # Model saving
    save_final_model: bool = (
        True  # Save model after training completes (disable for CI smoke tests)
    )
    save_vl_model: bool = False  # Control whether to save VL model and processor in callbacks

    # Checkpoint uploading
    upload_checkpoints: bool = False
    upload_every: int = 1000
    upload_last_n_checkpoints: int = 5
    max_concurrent_uploads: int = 2

    # MSC (multi-storage-client) checkpoint options (for remote storage like s3:// or msc://)
    msc_checkpoint_url: str | None = None
    msc_profile_url: str | None = (
        "msc://GearCheckpoints/profiles"  # Separate URL for profiler traces
    )
    msc_callback_mode: str = "async"  # "async" = background uploads, "sync" = blocking uploads
    msc_execution_mode: str = (
        "local"  # "local" = sequential upload, "ray" = distributed parallel upload
    )
    msc_delete_checkpoint_after_upload: bool = (
        True  # Delete local checkpoint after successful upload
    )

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: int = 500
    eval_set_split_ratio: float = 0.1
    eval_batch_size: int = 2
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_better: bool = True

    # Validation loss (distributed val loss during training)
    val_total_samples: int = 1024  # Fixed total eval samples (distributed across ranks)
    val_loss_seed: int = 12345  # Separate seed for val shard sampling

    # DeepSpeed (default)
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3)
    deepspeed_offload_optimizer: bool = False  # Offload optimizer states to CPU (pinned memory)
    deepspeed_reduce_bucket_size: float = (
        0  # Override reduce/allgather bucket size (bytes). 0 = use JSON default.
    )
    gradient_checkpointing: bool = False

    use_async_checkpoint_save: bool = True

    # Transformers loading parameters
    # TODO: once we have hierarchat config system, put these into a group  # noqa: TD002, TD003
    # called transformers_kwargs and do AutoModel.from_pretrained(**transformers_kwargs)
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = True
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  # Access token for HuggingFace Hub

    # DDP
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100
    # None preserves the default torchrun behavior: NCCL for GPU runs, Gloo for CPU runs.
    distributed_backend: str | None = None

    # Hardware
    num_gpus: int = 1
    # CPU-mode worker count — used only when num_gpus == 0. One Ray
    # TorchTrainer worker per value. >1 spreads workers across nodes
    # via STRICT_SPREAD so each pod runs one worker; must match the
    # cluster's num_nodes when launched via SkyPilot.
    num_cpu_workers: int = 1
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    enable_struct_logs: bool = False
    training_logging_level: str = "INFO"
    training_log_dir: str | None = None

    # Data handling
    remove_unused_columns: bool = False

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "groot_omni"
    # Controls rank-0 fallback dashboard registration when the launcher did not
    # inject GEAR_EXPERIMENT_ID. Valid values: "auto", "enabled", "disabled".
    experiment_tracking: str = "auto"

    # GEAR Dashboard experiment IDs (populated at launch time via gear_registration.py)
    experiment_id: str = ""
    experiment_ref: str = ""
    training_run_id: str = ""

    # Profiling
    enable_profiling: bool = True
    profile_skip_first: int = 50  # Steps to skip before profiling (relative to session start)
    profile_warmup_steps: int = 1  # Warmup steps before active profiling
    profile_active_steps: int = 3  # Number of active profiling steps
    profile_with_stack: bool = True  # Record Python stack traces (enables source-level attribution)
    profile_delete_after_upload: bool = True  # Delete local profiles after S3 upload

    # Worker-level profiling (runs torch.profiler inside DataLoader worker processes)
    enable_worker_profiling: bool = False
    worker_profile_skip_shards: int = 5  # Shards to skip before starting worker profiler
    worker_profile_active_samples: int = 2000  # Samples to profile per worker

    # Max number of retries in training for fault tolerance
    max_retries: int = 3

    # Max number of Ray Train worker failures before the job is terminated
    max_failures: int = 50

    # For testing.
    assert_loss_less_than: float | None = None

    # RL
    add_rl_callback: bool = False

    # Straggler Detection (nvidia_resiliency_ext)
    straggler: "StragglerConfig" = dataclasses.field(default_factory=lambda: StragglerConfig())

    # Open-loop evaluation (two nested configs).
    open_loop_eval: OpenLoopEvalConfig = dataclasses.field(
        default_factory=OpenLoopEvalConfig
    )
    open_loop_eval_in_process: OpenLoopEvalInProcessConfig = dataclasses.field(
        default_factory=OpenLoopEvalInProcessConfig
    )

    def __post_init__(self) -> None:
        """Coerce dict -> nested dataclass for YAML-loaded configs.

        ``Config.load_dict`` in ``groot/core/training/configs/base_config.py``
        does ``TrainingConfig(**data["training"])``. With nested dataclass
        fields, YAML delivers ``data["training"]["open_loop_eval"]`` as a
        ``dict``. Without coercion,
        ``config.training.open_loop_eval.enabled`` raises ``AttributeError``
        on ``dict.enabled``. Mirrors ``DataConfig.__post_init__``.
        """
        if isinstance(self.open_loop_eval, dict):
            self.open_loop_eval = OpenLoopEvalConfig(**self.open_loop_eval)
        if isinstance(self.open_loop_eval_in_process, dict):
            self.open_loop_eval_in_process = OpenLoopEvalInProcessConfig(
                **self.open_loop_eval_in_process
            )
