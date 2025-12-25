from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Output
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # Basic training
    max_steps: int = 30000  # this will override num_epochs
    global_batch_size: int = 1024
    batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1

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
    wandb_project: str = "finetune-gr00t-n1d6"

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
