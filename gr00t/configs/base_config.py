from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional

import yaml

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

from .data.data_config import DataConfig, SingleDatasetConfig
from .model import create_model_union_type
from .model.gr00t_n1d6 import Gr00tN1d6Config
from .training.training_config import TrainingConfig


ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration."""

    load_config_path: Optional[str] = None
    model: ModelUnionType = field(default_factory=lambda: Gr00tN1d6Config())
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
            self = data
        else:
            raise ValueError(f"Invalid config file: {path}")
        # config = cls(**config) # if yaml.dump(self.__dict__, ...) is used
        return self

    def load_dict(self, data: dict):
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
            # Ensure nested datasets are converted to dataclass instances
            converted: List[SingleDatasetConfig] = []
            for ds in self.data.datasets:
                if isinstance(ds, dict):
                    converted.append(SingleDatasetConfig(**ds))
                else:
                    converted.append(ds)
            self.data.datasets = converted
        if "training" in data:
            self.training = TrainingConfig(**data["training"])
        return self

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        return data

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration."""
        stage = self.training.deepspeed_stage

        gr00t_dir = Path(__file__).parent.parent
        if stage == 2:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero2_config.json"))
        elif stage == 3:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero3_config.json"))
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")

        return config

    def validate(self):
        """Validate configuration."""
        # Check dataset path(s)
        embodiment_tags = set()
        for d_cfg in self.data.datasets:
            # (Disable missing data check because we now support caching PDX data sources.)
            # if not Path(d_cfg.dataset_path).exists():
            #     raise ValueError(f"Dataset path does not exist: {d_cfg.dataset_path}")
            if d_cfg.dataset_type == "physical_embodiment" and not d_cfg.embodiment_tag:
                raise ValueError(f"Embodiment tag is empty for dataset {d_cfg.dataset_path}")
            if d_cfg.embodiment_tag is not None:
                embodiment_tags.add(d_cfg.embodiment_tag)

        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            stripped_modality_configs[embodiment_tag] = self.data.modality_configs[embodiment_tag]
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        total_ratio = sum(d.mix_ratio for d in self.data.datasets)
        if total_ratio <= 0:
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

        if isinstance(self.model, Gr00tN1d6Config):
            import warnings

            if self.model.eagle_collator:
                warnings.warn(
                    'eagle_collator is deprecated. Please use backbone_model_type "eagle" in the future.',
                    DeprecationWarning,
                )
                self.model.backbone_model_type = "eagle"
            assert self.model.backbone_model_type in [
                "eagle",
            ], f"Invalid backbone model type: {self.model.backbone_model_type}"

        # Validate precision settings
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
