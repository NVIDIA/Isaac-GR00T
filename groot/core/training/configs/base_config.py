# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import yaml

from groot.core.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
from groot.core.training.configs.data.data_config import DataConfig, SingleDatasetConfig
from groot.core.training.configs.data.registry import (
    create_data_config,
    resolve_data_validation_inputs,
    validate_data_training_contract,
)
from groot.core.training.configs.model import create_model_union_type
from groot.core.training.configs.model.n2 import GrootN2Config
from groot.core.training.configs.training.registry import create_training_config
from groot.core.training.configs.training.training_config import TrainingConfig

ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration."""

    load_config_path: str | None = None
    model: ModelUnionType = field(default_factory=lambda: GrootN2Config())
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self, f)

    def load(self, path: Path):
        """Load configuration from YAML file."""
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        if isinstance(data, dict):  # for training
            self.load_dict(data)
        elif isinstance(data, self.__class__):
            self = data  # noqa: PLW0642
        else:
            raise ValueError(f"Invalid config file: {path}")  # noqa: TRY004
        return self

    def load_dict(self, data: dict):
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            data_data = dict(data["data"])
            # Pop modality_configs overrides before creating DataConfig
            # so DataConfig gets the full MODALITY_CONFIGS default.
            yaml_modality_configs = data_data.pop("modality_configs", {})
            self.data = create_data_config(data_data)
            # Ensure nested datasets are converted to dataclass instances
            converted: list[SingleDatasetConfig] = []
            for ds in self.data.datasets:
                if isinstance(ds, dict):
                    converted.append(SingleDatasetConfig(**ds))
                else:
                    converted.append(ds)
            self.data.datasets = converted
            # Merge YAML modality_configs overrides with defaults (only override specified fields)
            if yaml_modality_configs:
                from groot.core.data.types import ModalityConfig

                for embodiment_tag, modalities in yaml_modality_configs.items():
                    if embodiment_tag not in self.data.modality_configs:
                        continue
                    for modality_name, modality_cfg in modalities.items():
                        if isinstance(modality_cfg, dict):
                            default_cfg = self.data.modality_configs[embodiment_tag].get(
                                modality_name
                            )
                            if default_cfg is not None and isinstance(default_cfg, ModalityConfig):
                                self.data.modality_configs[embodiment_tag][modality_name] = replace(
                                    default_cfg, **modality_cfg
                                )
                            else:
                                self.data.modality_configs[embodiment_tag][modality_name] = (
                                    ModalityConfig(**modality_cfg)
                                )
                        else:
                            self.data.modality_configs[embodiment_tag][modality_name] = modality_cfg
        if "training" in data:
            self.training = create_training_config(data["training"])
        return self

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        return data

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        stage = self.training.deepspeed_stage
        offload = self.training.deepspeed_offload_optimizer

        omni_dir = Path(__file__).parent.parent
        if stage == 2 and offload:
            config_path = omni_dir / "configs/deepspeed/zero2_offload_config.json"
        elif stage == 2:
            config_path = omni_dir / "configs/deepspeed/zero2_config.json"
        elif stage == 3:
            config_path = omni_dir / "configs/deepspeed/zero3_config.json"
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")
        with config_path.open() as config_file:
            config = json.load(config_file)

        bucket = self.training.deepspeed_reduce_bucket_size
        if bucket > 0:
            config["zero_optimization"]["reduce_bucket_size"] = bucket
            config["zero_optimization"]["allgather_bucket_size"] = bucket

        return config

    def validate(self):
        """Validate configuration."""
        validate_data_training_contract(self.data, self.model, self.training)
        validation_inputs = resolve_data_validation_inputs(self.data)

        # Check dataset path(s)
        embodiment_tags = set()
        for d_cfg in validation_inputs.datasets:
            if d_cfg.dataset_type == "physical_embodiment" and not d_cfg.embodiment_tag:
                raise ValueError(f"Embodiment tag is empty for dataset {d_cfg.dataset_path}")
            if d_cfg.embodiment_tag is not None:
                embodiment_tags.add(d_cfg.embodiment_tag)
        embodiment_tags.update(validation_inputs.embodiment_tags)

        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            stripped_modality_configs[embodiment_tag] = self.data.modality_configs[embodiment_tag]
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        total_ratio = sum(d.mix_ratio for d in validation_inputs.datasets)
        if total_ratio <= 0 and not validation_inputs.permits_empty_dataset_mix:
            raise ValueError("Sum of mix_ratio must be greater than zero")

        # Fill in default values for action configs
        for embodiment_tag in self.data.modality_configs:
            # Fill in default values for action representation, type and format
            if self.data.modality_configs[embodiment_tag]["action"].action_configs is None:
                self.data.modality_configs[embodiment_tag]["action"].action_configs = [
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ] * len(self.data.modality_configs[embodiment_tag]["action"].modality_keys)

        # Check GPU availability, skip if running under Ray
        import ray
        import torch

        if not ray.is_initialized() and self.training.num_gpus > torch.cuda.device_count():
            raise ValueError(
                f"Requested {self.training.num_gpus} GPUs but only {torch.cuda.device_count()} available"
            )
        # Validate precision settings
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
