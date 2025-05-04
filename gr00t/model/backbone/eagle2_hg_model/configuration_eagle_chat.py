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


import copy

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Eagle2ChatConfig(PretrainedConfig):
    model_type = "eagle_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        mlp_checkpoint=True,
        pre_feature_reduction=False,
        keep_aspect_ratio=False,
        vocab_size=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        attn_impl = kwargs.get("attn_implementation", None)
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing Vision Encoders with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LLM config with default values")

        if attn_impl:
            logger.info(f"attn_impl: {attn_impl}")
            vision_config["_attn_implementation"] = attn_impl
            llm_config["attn_implementation"] = attn_impl

        if vision_config["model_type"] == "siglip_vision_model":
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            raise ValueError("Unsupported model_type: {}".format(vision_config["model_type"]))

        if llm_config["architectures"][0] == "LlamaForCausalLM":
            self.llm_config = LlamaConfig(**llm_config)
        else:
            raise ValueError("Unsupported architecture: {}".format(llm_config["architectures"][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.mlp_checkpoint = mlp_checkpoint
        self.pre_feature_reduction = pre_feature_reduction
        self.keep_aspect_ratio = keep_aspect_ratio
        self.vocab_size = self.llm_config.vocab_size
        logger.info(f"keep_aspect_ratio: {self.keep_aspect_ratio}")
        logger.info(f"vision_select_layer: {self.select_layer}")
        logger.info(f"min_dynamic_patch: {self.min_dynamic_patch}")
        logger.info(f"max_dynamic_patch: {self.max_dynamic_patch}")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch
        output["keep_aspect_ratio"] = self.keep_aspect_ratio

        return output
