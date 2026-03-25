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

import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.data.collator import BasicDataCollator
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.interfaces import BaseProcessor
import numpy as np
import torch
from transformers import PreTrainedModel


class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None

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


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class BasicPipeline(ModelPipeline):
    """A simple pipeline that works for diffusion and flowmatching-based models."""

    model_class: type[PreTrainedModel]
    processor_class: type[BaseProcessor]
    data_collator_class: type[BasicDataCollator] = BasicDataCollator

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        # Load model
        model = self.model_class(self.config.model)
        print("Model Config: ", model.config)

        # unfreeze the model first
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        return model

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=None,  # This will be computed and set later.
            **self.config.model.processor_kwargs,
        )
        dataset_factory = DatasetFactory(self.config)
        train_dataset, eval_dataset = dataset_factory.build(self.processor)

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
