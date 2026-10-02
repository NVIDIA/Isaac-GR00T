# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel

from groot.core.data.collator import BasicDataCollator
from groot.core.data.dataset.build_artifacts import DatasetBuildArtifacts
from groot.core.data.dataset.factory import DatasetFactory
from groot.core.data.types import BaseProcessor
from groot.core.training.configs.base_config import Config


class ModelPipeline:
    def __init__(
        self,
        config: Config,
        *,
        dataset_build_artifacts: DatasetBuildArtifacts | None = None,
    ):
        self.config = config
        self.dataset_build_artifacts = dataset_build_artifacts or DatasetBuildArtifacts()
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None
        self._dataset_factory = None
        # Checkpoint that training continues from (None for a fresh run); set by
        # experiment.run before setup().
        self.resume_checkpoint: str | None = None

    def setup(self):
        pass

    def return_model(self):
        return self.model

    def return_dataset(self):
        return self.train_dataset, self.eval_dataset

    def return_collator(self):
        return self.data_collator

    def return_processor(self):
        return self.processor

    def create_trainer_callbacks(self, trainer) -> list:  # noqa: ARG002
        """Return model-family callbacks to install after Trainer construction."""
        return []

    def get_trainer_class(self, default_trainer_class):
        """Return the trainer class for this model family."""
        return default_trainer_class

    def trainer_init_kwargs(self) -> dict:
        """Return model-family keyword arguments for trainer construction."""
        return {}

    def stop(self) -> None:
        """Stop dataset/hydrator resources (e.g. IPC server). Call when training ends."""
        if self._dataset_factory is not None:
            self._dataset_factory.stop()
            self._dataset_factory = None


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class BasicPipeline(ModelPipeline):
    """A simple pipeline that works for diffusion and flowmatching-based models."""

    model_class: type[PreTrainedModel]
    processor_class: type[BaseProcessor]
    data_collator_class: type[BasicDataCollator] = BasicDataCollator

    def __init__(
        self,
        config: Config,
        save_cfg_dir: Path,
        *,
        dataset_build_artifacts: DatasetBuildArtifacts | None = None,
    ):
        super().__init__(
            config,
            dataset_build_artifacts=dataset_build_artifacts,
        )
        self.save_cfg_dir = save_cfg_dir

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        # Load model
        model = self.model_class(self.config.model)
        logging.info("Model Config: %s", model.config)

        # unfreeze the model first
        for _name, param in model.named_parameters():
            param.requires_grad = True

        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)"
        )
        return model

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=None,  # This will be computed and set later.
            **self.config.model.processor_kwargs,
        )
        self._dataset_factory: DatasetFactory = DatasetFactory(
            self.config.data,
            experiment_name=self.config.training.experiment_name,
            dataset_build_artifacts=self.dataset_build_artifacts,
        )
        train_dataset, eval_dataset = self._dataset_factory.build(self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.data_collator_class()
        return data_collator
