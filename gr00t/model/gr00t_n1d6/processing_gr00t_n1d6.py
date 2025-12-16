import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Literal
import warnings

import albumentations as A
from gr00t.configs.data.embodiment_configs import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.utils import parse_modality_configs, to_json_serializable
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

from .image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)


# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

### Mapping from embodiment tag to projector index.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    ##### Pretrain embodiment ids #####
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    ##### Pre-registered posttrain embodiment ids #####
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    assert model_name == "nvidia/Eagle-Block2A-2B-v2", f"Processor for {model_name} not supported"
    eagle_path = os.path.join(
        os.path.dirname(__file__), "..", "modules", "nvidia", "Eagle-Block2A-2B-v2"
    )
    return AutoProcessor.from_pretrained(eagle_path, **transformers_loading_kwargs)


class Gr00tN1d6DataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict = {},
    ):
        ### We need to use the same  processor for padding input ids and concat
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # Handle vlm_content specially - extract text and images
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                # NOTE: some VLMs need this, others don't.
                if self.model_type == "eagle":
                    image_inputs, _ = self.processor.process_vision_info(
                        [v["conversation"] for v in values]
                    )
                vlm_inputs = self.processor(
                    text=text_list, images=image_inputs, return_tensors="pt", padding=True
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d6DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d6Processor(BaseProcessor):
    data_collator_class = Gr00tN1d6DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 512,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        model_type: Literal["eagle"] = "eagle",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        apply_sincos_state_encoding: bool = False,
        max_action_horizon: int = 40,
        use_albumentations: bool = False,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        # Save image processing settings
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # handle the case where the fine-tuning embodiment tag is not in the pre-trained embodiment tag mapping
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        # Choose between torchvision and albumentations transforms
        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size, image_crop_size, random_rotation_angle, color_jitter_params
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(
                embodiment_tag
            )

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        # Use StateActionProcessor to handle relative conversion and normalization
        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            # Concatenate actions
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys], dim=-1
            )  # (t, d)
            action_dim = normalized_actions.shape[1]
            # Pad action to max_action_dim
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            # Pad action to max_action_horizon
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            # Create action mask
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        # Concatenate states
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        normalized_states = torch.cat(
            [torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1
        )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1]
                ),
            ],
            dim=-1,
        )

        # Crop and resize images.
        if self.training:
            image_transform = self.train_image_transform
        else:
            image_transform = self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        # Add VLM inputs
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        image_transform: transforms.Compose | A.Compose,
        language: str,
    ):
        temporal_stacked_images = {}

        if self.use_albumentations:
            # Use albumentations transforms
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                # Apply transforms with replay for consistency
                transformed_images, replay = apply_with_replay(
                    image_transform, images[view], replay
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)
        else:
            # Use torchvision transforms
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
            .flatten(0, 1)
            .numpy()
        )  # (T*V, C, H, W), Eagle processor expects numpy array

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        # dump modality configs to dict using the recursive function
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = Path(save_directory) / "processor_config.json"
        statistics_file = Path(save_directory) / "statistics.json"
        embodiment_id_file = Path(save_directory) / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                # Image processing settings
                "image_crop_size": self.image_crop_size,
                "image_target_size": self.image_target_size,
                "use_albumentations": self.use_albumentations,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                "shortest_image_edge": self.shortest_image_edge,
                "crop_fraction": self.crop_fraction,
                # VLM settings
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                # State action dimensions
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                # StateActionProcessor settings
                "use_percentiles": self.use_percentiles,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        # Save statistics
        with open(statistics_file, "w") as f:
            json.dump(to_json_serializable(self.state_action_processor.statistics), f, indent=2)
        # Save embodiment id mapping
        with open(embodiment_id_file, "w") as f:
            json.dump(self.embodiment_id_mapping, f, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(cached_file(pretrained_model_name_or_path, "processor_config.json"))
            statistics_file = Path(cached_file(pretrained_model_name_or_path, "statistics.json"))
            embodiment_id_file = Path(
                cached_file(pretrained_model_name_or_path, "embodiment_id.json")
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file, "r") as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping
        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs while keeping pretrained embodiment configs
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


AutoProcessor.register("Gr00tN1d6", Gr00tN1d6Processor)
