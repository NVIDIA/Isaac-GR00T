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

from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Output
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # Basic training
    max_steps: int = 30000  # this will override num_epochs

    global_batch_size: int = 1024
    """Total batch summed across all GPUs in one forward/backward, BEFORE
    gradient accumulation. See :attr:`accumulated_batch_size` for the
    post-accumulation per-optimizer-step value."""

    per_gpu_batch_size: Optional[int] = None
    """Deprecated. When set, overrides ``global_batch_size`` as the per-GPU batch."""

    gradient_accumulation_steps: int = 1
    """Forward passes per optimizer step. Passed through to HuggingFace
    ``TrainingArguments``; see :attr:`accumulated_batch_size`."""

    # Optimization
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # this will override warmup_ratio
    max_grad_norm: float = 1.0

    # Optimizer choice (huggingface TrainingArguments.optim)
    # Options include: 'adamw_torch', 'adamw_torch_fused', 'paged_adamw_32bit',
    # 'paged_adamw_8bit' (requires bitsandbytes), 'adafactor', etc.
    optim: str = "adamw_torch_fused"

    start_from_checkpoint: Optional[str] = None
    skip_weight_loading: bool = False  # skip loading checkpoint weights (architecture only)

    # Mixed precision
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 5

    # Model saving
    save_vl_model: bool = False  # Control whether to save VL model and processor in callbacks
    save_only_model: bool = False  # Skip optimizer/scheduler/RNG states — cannot resume training

    # Default False so a rerun against an existing output_dir starts fresh.
    resume_from_checkpoint: bool = False

    # Checkpoint uploading
    upload_checkpoints: bool = False
    upload_every: int = 1000
    upload_last_n_checkpoints: int = 5
    max_concurrent_uploads: int = 2

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: int = 500
    eval_set_split_ratio: float = 0.1
    eval_batch_size: int = 2
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_better: bool = True

    # DeepSpeed (default)
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3)
    gradient_checkpointing: bool = False

    # Transformers loading parameters
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = False
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  # Access token for HuggingFace Hub

    # DDP
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100

    # Hardware
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    # Data handling
    remove_unused_columns: bool = False

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"

    # Profiling
    enable_profiling: bool = False

    # Max number of retries in training for fault tolerance
    max_retries: int = 3

    # For testing.
    assert_loss_less_than: float | None = None

    # RL
    add_rl_callback: bool = False

    # Open-loop evaluation
    enable_open_loop_eval: bool = False
    """Enable open-loop evaluation on saved checkpoints."""

    open_loop_eval_traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    open_loop_eval_steps_per_traj: int = 100
    """Number of steps to evaluate per trajectory."""

    open_loop_eval_plot_indices: Optional[list[int]] = None
    """List of action indices to plot. If None, plots all indices."""

    @property
    def accumulated_batch_size(self) -> int:
        """Total samples per optimizer step, after gradient accumulation."""
        if self.per_gpu_batch_size is not None:
            global_batch = self.per_gpu_batch_size * self.num_gpus
        else:
            global_batch = self.global_batch_size
        return global_batch * self.gradient_accumulation_steps

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.gradient_accumulation_steps > 1 and self.per_gpu_batch_size is None:
            warnings.warn(
                f"global_batch_size={self.global_batch_size} is pre-accumulation; "
                f"accumulated_batch_size={self.accumulated_batch_size} "
                f"(× gradient_accumulation_steps={self.gradient_accumulation_steps}).",
                stacklevel=2,
            )


def check_resume_compatibility(training: TrainingConfig) -> None:
    """Reject ``save_only_model=True`` + ``resume_from_checkpoint=True``.

    HF Trainer would otherwise restore ``global_step`` from
    ``trainer_state.json`` and silently re-init optimizer / LR schedule.
    """
    if training.save_only_model and training.resume_from_checkpoint:
        raise ValueError(
            "save_only_model=True is incompatible with resume_from_checkpoint=True: "
            "the checkpoint lacks optimizer/scheduler/RNG state, so resuming "
            "silently re-initializes the optimizer and LR schedule, degrading "
            "training quality. Either disable save_only_model, or start fresh "
            "(set resume_from_checkpoint=False or use a different output_dir)."
        )
