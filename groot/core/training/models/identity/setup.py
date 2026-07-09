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
import logging
from pathlib import Path

from groot.core.data.dataset.build_artifacts import DatasetBuildArtifacts
from groot.core.data.dataset.factory import get_dataset_factory
from groot.core.training.configs.base_config import Config
from groot.core.training.configs.model.identity import IdentityConfig
from groot.core.training.models.base.model_pipeline import ModelPipeline
from groot.core.training.models.identity.model import IdentityModel
from groot.core.training.models.identity.processor import (
    IdentityCollator,
    IdentityProcessor,
)
from groot.core.training.models.registry import register_model


class IdentityPipeline(ModelPipeline):
    """Pipeline that wires the dry-run identity stack into the standard
    ``experiment.run`` flow: ``IdentityModel`` + ``IdentityProcessor`` +
    ``IdentityCollator``, no tokenizer or transforms."""

    model_class = IdentityModel
    processor_class = IdentityProcessor
    data_collator_class = IdentityCollator

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
        self.train_dataset, self.eval_dataset = self._create_dataset()
        self.data_collator = self._create_collator()

    def _create_model(self):
        model = self.model_class()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)"
        )
        return model

    def _create_dataset(self):
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
        )
        # Respect the YAML's `dataset_factory_type` so quality_weighted
        # configs route through QualityPriorityGrouper (and emit
        # `step_priority_mapping` events). Hardcoding DatasetFactory here
        # bypassed the V2 factory regardless of what the config asked for.
        self._dataset_factory = get_dataset_factory(
            self.config.data,
            experiment_name=self.config.training.experiment_name,
            dataset_build_artifacts=self.dataset_build_artifacts,
        )
        return self._dataset_factory.build(self.processor)

    def _create_collator(self):
        return self.processor.collator


register_model(IdentityConfig, IdentityPipeline)
