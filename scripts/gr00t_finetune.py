# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# Helper functions
#####################################################################################


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """
    Copy weights with partial dimension matching for action_dim changes.
    NOTE(Youliang): this is a very experimental implementation to handle action_dim changes. TODO: improve this.
    """
    total_params = copied_params = random_params = 0

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue

        new_tensor = new_dict[key]
        total_params += new_tensor.numel()

        if old_tensor.shape == new_tensor.shape:
            # Same shape: direct copy
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            # Input dimension change: copy [:, :old_dim]
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            # Output dimension change: copy first old_dim elements of last dimension
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            # Incompatible shape: keep random initialization
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random ({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim+1}-{new_dim} will be learned from scratch")
    return new_dict


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Assert that the last transform is a GR00TTransform and has max_action_dim
    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
    data_max_action_dim = last_transform.max_action_dim

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            print(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
