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

import argparse
import copy
import os
import time
from typing import Dict, Optional

# Global attention implementation setting
# Read from environment variable first, otherwise default to "eager". 
ATTN_IMPLEMENTATION = os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")

import modelopt.torch.quantization as mtq
import numpy as np
import torch
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
)

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH, EagleBackbone
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values


def no_batch_collate_fn(batch):
    """Collate function that returns the first item without adding batch dimension."""
    return batch[0]


class ViTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for ViT calibration data.
    This provides realistic calibration data for the vision transformer.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
        use_position_ids: bool = True,
    ):
        """
        Initialize the ViT calibration dataset.

        Args:
            dataset_path: Path to the LeRobot dataset
            modality_configs: Modality configuration for the dataset
            embodiment_tag: Embodiment tag for the dataset
            policy: Gr00tPolicy instance for using apply_transforms()
            calib_size: Number of calibration samples to use
            video_backend: Video backend for loading videos
        """
        self.calib_size = calib_size
        self.policy = policy
        self.use_position_ids = use_position_ids
        # Initialize the LeRobot dataset
        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        # Use sequential indices for calibration
        self.dataset_size = len(self.lerobot_dataset)
        print(f"ViT Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        data = self.lerobot_dataset[idx]

        # Process the data to get pixel_values and position_ids for ViT
        processed_data = self._process_vit_data(data)
        return processed_data

    def _process_vit_data(self, data):
        """
        Process LeRobot data to extract pixel_values and position_ids for ViT calibration.
        """
        try:
            # Ensure data is in the correct format for apply_transforms
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            # Apply the same transforms as used in training/inference
            transformed_data = self.policy.apply_transforms(data)

            # Check if we have eagle pixel values
            if "eagle_pixel_values" in transformed_data:
                pixel_values = transformed_data["eagle_pixel_values"]
                batch_size = pixel_values.shape[0]
                # Generate position_ids for the patches
                num_patches = (
                    self.policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
                )
                position_ids = torch.arange(
                    num_patches, dtype=torch.long, device=pixel_values.device
                ).expand((batch_size, -1))
                if self.use_position_ids:
                    return {
                    "pixel_values": pixel_values,
                        "position_ids": position_ids,
                    }
                else:
                    return {
                        "pixel_values": pixel_values,
                    }
            else:
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )
        except Exception as e:
            print(f"Warning: ViT data processing failed: {e}, using dummy data")
            raise RuntimeError(f"apply_transforms() failed: {e}")


class LLMCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for LLM calibration data.
    This provides more realistic calibration data compared to random data.
    Uses apply_transforms() for consistent data processing.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
        enable_comparison: bool = False,
    ):
        """
        Initialize the LeRobot calibration dataset.

        Args:
            dataset_path: Path to the LeRobot dataset
            modality_configs: Modality configuration for the dataset
            embodiment_tag: Embodiment tag for the dataset
            policy: Gr00tPolicy instance for using apply_transforms()
            calib_size: Number of calibration samples to use
            video_backend: Video backend for loading videos
            enable_comparison: Whether to enable input_embeds comparison with saved file
        """
        self.calib_size = calib_size
        self.policy = policy
        self.enable_comparison = enable_comparison

        # Initialize the LeRobot dataset
        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        # Use sequential indices for calibration
        self.dataset_size = len(self.lerobot_dataset)
        print(f"Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        data = self.lerobot_dataset[idx]

        # Process the data to get input_ids, vit_embeds, and attention_mask
        processed_data = self._process_llm_data(data)

        return processed_data

    def _process_llm_data(self, data):
        """
        Process LLM data to extract input_ids, vit_embeds, and attention_mask
        for LLM calibration using apply_transforms() for consistent data processing.
        """
        # Use apply_transforms() for consistent data processing
        try:
            # Ensure data is in the correct format for apply_transforms
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            # Apply the same transforms as used in training/inference
            transformed_data = self.policy.apply_transforms(data)

            # Check if we have eagle data (from apply_transforms)
            if "eagle_input_ids" in transformed_data and "eagle_pixel_values" in transformed_data:
                # Use the already processed eagle data
                pixel_values = transformed_data["eagle_pixel_values"]
                input_ids = transformed_data["eagle_input_ids"]
                attention_mask = transformed_data["eagle_attention_mask"]

                # Extract vit_embeds using the actual ViT model from the policy
                try:
                    # Use the policy's eagle model for consistent feature extraction
                    with torch.no_grad():
                        # Move pixel_values to CUDA before feature extraction
                        pixel_values = pixel_values.to("cuda")
                        # Extract vit_embeds using the policy's eagle model
                        if self.policy.model.backbone.eagle_model.select_layer == -1:
                            vit_embeds = self.policy.model.backbone.eagle_model.vision_model(
                                pixel_values=pixel_values,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                            if hasattr(vit_embeds, "last_hidden_state"):
                                vit_embeds = vit_embeds.last_hidden_state
                        else:
                            vit_embeds = self.policy.model.backbone.eagle_model.vision_model(
                                pixel_values=pixel_values,
                                output_hidden_states=True,
                                return_dict=True,
                            ).hidden_states[self.policy.model.backbone.eagle_model.select_layer]

                except Exception as e:
                    raise RuntimeError(f"Policy eagle model failed: {e}")

                vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])

                if self.policy.model.backbone.eagle_model.use_pixel_shuffle:
                    h = w = int(vit_embeds.shape[1] ** 0.5)
                    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                    vit_embeds = self.policy.model.backbone.pixel_shuffle(
                        vit_embeds, scale_factor=self.policy.model.backbone.downsample_ratio
                    )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
                    vit_embeds = vit_embeds.reshape(
                        vit_embeds.shape[0], -1, vit_embeds.shape[-1]
                    )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])

                if (
                    self.policy.model.backbone.eagle_model.mlp_checkpoint
                    and vit_embeds.requires_grad
                ):
                    vit_embeds = cp.checkpoint(
                        self.policy.model.backbone.eagle_model.mlp1, vit_embeds
                    )
                else:
                    vit_embeds = self.policy.model.backbone.eagle_model.mlp1(vit_embeds)

                # Move input_ids to the same device as the model
                input_ids = input_ids.to(next(self.policy.model.parameters()).device)

                # Get input_ids from vl_input and convert to embeddings
                input_embeds = (
                    self.policy.model.backbone.eagle_model.language_model.get_input_embeddings()(
                        input_ids
                    )
                )

                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                input_ids_flat = input_ids.reshape(B * N)
                selected = (
                    input_ids_flat == self.policy.model.backbone.eagle_model.image_token_index
                )
                try:
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                        -1, C
                    )
                except Exception as e:
                    vit_embeds = vit_embeds.reshape(-1, C)
                    print(
                        f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                        f"vit_embeds.shape={vit_embeds.shape}"
                    )
                    n_token = selected.sum()
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

                input_embeds = input_embeds.reshape(B, N, C)

                # Convert to float16 for quantization compatibility
                input_embeds = input_embeds.to(torch.float16)

                return {
                    "inputs_embeds": input_embeds,
                    "attention_mask": attention_mask,
                }
            else:
                # If eagle data is not available, raise an error
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )

        except Exception as e:
            raise RuntimeError(f"apply_transforms() failed: {e}")


class DiTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for DiT calibration data.
    This provides realistic calibration data for the diffusion transformer.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
    ):
        """
        Initialize the DiT calibration dataset.

        Args:
            dataset_path: Path to the LeRobot dataset
            modality_configs: Modality configuration for the dataset
            embodiment_tag: Embodiment tag for the dataset
            policy: Gr00tPolicy instance for using apply_transforms()
            calib_size: Number of calibration samples to use
            video_backend: Video backend for loading videos
        """
        self.calib_size = calib_size
        self.policy = policy

        # Initialize the LeRobot dataset
        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        # Use sequential indices for calibration
        self.dataset_size = len(self.lerobot_dataset)
        print(f"DiT Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

        # Get dimensions from policy
        self.input_embedding_dim = policy.model.action_head.config.input_embedding_dim
        self.backbone_embedding_dim = policy.model.action_head.config.backbone_embedding_dim
        self.action_horizon = policy.model.action_head.config.action_horizon
        self.num_target_vision_tokens = policy.model.action_head.config.num_target_vision_tokens

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        data = self.lerobot_dataset[idx]

        # Process the data to get DiT inputs
        processed_data = self._process_dit_data(data)

        return processed_data

    def _process_dit_data(self, data):
        """
        Process LeRobot data to prepare inputs for DiT calibration.
        Returns the necessary inputs for running the denoising loop.
        """
        try:
            # Ensure data is in the correct format for apply_transforms
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            # Apply the same transforms as used in training/inference
            transformed_data = self.policy.apply_transforms(data)

            # Use the model's prepare_input method which returns backbone_inputs and action_inputs
            backbone_inputs, action_inputs = self.policy.model.prepare_input(transformed_data)
            backbone_outputs = self.policy.model.backbone(backbone_inputs)

            backbone_output = self.policy.model.action_head.process_backbone_output(
                backbone_outputs
            )

            # Get vision and language embeddings.
            vl_embs = backbone_output["backbone_features"]
            embodiment_id = action_inputs["embodiment_id"]

            # Embed state.
            state_features = self.policy.model.action_head.state_encoder(
                action_inputs["state"], embodiment_id
            )

            # Set initial actions as the sampled noise.
            batch_size = vl_embs.shape[0]
            device = vl_embs.device
            actions = torch.randn(
                size=(
                    batch_size,
                    self.policy.model.action_head.config.action_horizon,
                    self.policy.model.action_head.config.action_dim,
                ),
                dtype=vl_embs.dtype,
                device=device,
            )

            return {
                "vl_embs": vl_embs,  # Remove batch dimension
                "state_features": state_features,  # Remove batch dimension
                "actions": actions,  # Remove batch dimension
                "embodiment_id": embodiment_id,  # Remove batch dimension
            }
        except Exception as e:
            raise RuntimeError(f"DiT data processing failed: {e}")


def _quantize_model(model, calib_dataloader, quant_cfg):
    """
    The calibration loop for the model can be setup using the modelopt API.
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            data = {k: v.to(next(model.parameters()).device) for k, v in data.items()}
            model(**data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model


def _quantize_dit_model(model, calib_dataloader, quant_cfg, action_head):
    """
    Custom calibration loop for DiT model that runs the full denoising process.
    DiT requires multiple forward passes (typically 4 steps) for proper calibration.
    """

    def calibrate_loop(model):
        """Run the denoising loop for DiT calibration."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating DiT batch {idx}...")

            # Move data to device
            device = next(model.parameters()).device
            vl_embs = data["vl_embs"].to(device)
            state_features = data["state_features"].to(device)
            actions = data["actions"].to(device)
            embodiment_id = data["embodiment_id"].to(device)

            batch_size = vl_embs.shape[0]
            num_steps = action_head.num_inference_timesteps
            dt = 1.0 / num_steps

            # Run denoising steps (typically 4 steps)
            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * action_head.num_timestep_buckets)

                # Embed noised action trajectory
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = action_head.action_encoder(
                    actions, timesteps_tensor, embodiment_id
                )

                # Maybe add position embedding
                if action_head.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                # Join state, future tokens, and action embeddings
                future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                # Run DiT model forward (this is what we're calibrating)
                sa_embs_float16 = sa_embs.to(torch.float16)
                vl_embs_float16 = vl_embs.to(torch.float16)

                # Forward pass through model being calibrated (output not used, just for calibration)
                _ = model(
                    hidden_states=sa_embs_float16,
                    encoder_hidden_states=vl_embs_float16,
                    timestep=timesteps_tensor,
                )

                # Forward pass through original model
                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )

                # Decode predictions
                pred = action_head.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -action_head.config.action_horizon :]

                # Update actions using euler integration
                actions = actions + dt * pred_velocity

    print("Starting DiT quantization with multi-step denoising...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"DiT quantization finishes in {end_time - start_time}s.")

    return model


def quantize_vit(
    model,
    precision="fp8",
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    use_position_ids=True,
    policy=None,
    compare_accuracy=True,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
):
    """
    Quantize the ViT model using FP8 quantization.

    Args:
        model: The ViT model to quantize
        precision: Quantization precision (fp8, fp16, etc.)
        calib_size: Number of calibration samples
        batch_size: Batch size for calibration
        dataset_path: Path to LeRobot dataset
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        video_backend: Video backend
        policy: Gr00tPolicy instance
        compare_accuracy: Whether to compare accuracy before/after quantization

    Returns:
        Quantized model
    """
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in [
        "fp8",
        "fp16",
    ], f"Only fp8 and fp16 are supported for ViT. You passed: {precision}."

    # FP8 quantization configuration
    quant_cfg = mtq.FP8_DEFAULT_CFG

    # Disable Conv to avoid accuracy degradation.
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    # Create the dataset and dataloader
    if dataset_path is None or modality_configs is None or policy is None:
        raise ValueError(
            "ViT quantization requires valid dataset_path, modality_configs, and policy."
        )

    print(f"Using LeRobot dataset for ViT calibration: {dataset_path}")
    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()
    device = "cuda"
    policy_copy2 = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    dataset = ViTCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        policy=policy_copy2,
        calib_size=calib_size,
        video_backend=video_backend,
        use_position_ids=use_position_ids,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    # Quantize the model if quantization config is provided
    if quant_cfg is not None:
        quantized_model = _quantize_model(model, data_loader, quant_cfg)
        mtq.print_quant_summary(quantized_model)

        return quantized_model
    else:
        print("No quantization applied to ViT model")
        return model


def quantize_dit(
    model,
    precision="fp8",
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    attention_mask=None,
    input_state=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
):
    """
    Quantize the DiT (Diffusion Transformer) model using FP8 quantization.

    Args:
        model: The DiT model to quantize
        action_head: The action head containing encoder/decoder and other components
        precision: Quantization precision (fp8, fp16, etc.)
        calib_size: Number of calibration samples
        batch_size: Batch size for calibration
        dataset_path: Path to LeRobot dataset
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        video_backend: Video backend
        policy: Gr00tPolicy instance
        compare_accuracy: Whether to compare accuracy before/after quantization
        attention_mask: Attention mask for sequence length
        input_state: Input state for dimensions

    Returns:
        Quantized model
    """
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in [
        "fp8",
        "fp16",
    ], f"Only fp8 and fp16 are supported for DiT. You passed: {precision}."

    # FP8 quantization configuration for DiT
    quant_cfg = mtq.FP8_DEFAULT_CFG
    quant_cfg["quant_cfg"]["*[qkv]_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None}
    quant_cfg["quant_cfg"]["*softmax_quantizer"] = {"num_bits": (4, 3), "axis": None}

    # Create the dataset and dataloader
    if dataset_path is None or modality_configs is None or policy is None:
        raise ValueError(
            "DiT quantization requires valid dataset_path, modality_configs, and policy."
        )

    print(f"Using LeRobot dataset for DiT calibration: {dataset_path}")
    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()
    device = "cuda"
    policy_copy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    dataset = DiTCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        policy=policy_copy,
        calib_size=calib_size,
        video_backend=video_backend,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    # Quantize the model if quantization config is provided
    if quant_cfg is not None:
        # Use custom DiT calibration function that runs the denoising loop
        quantized_model = _quantize_dit_model(
            model, data_loader, quant_cfg, policy_copy.model.action_head
        )
        mtq.print_quant_summary(quantized_model)

        return quantized_model
    else:
        print("No quantization applied to DiT model")
        return model


def compare_model_accuracy(original_model, quantized_model, test_data, device="cuda"):
    """
    Compare accuracy between original and quantized models.

    Args:
        original_model: The original model before quantization
        quantized_model: The quantized model after quantization
        test_data: Test data for comparison
        device: Device to run comparison on

    Returns:
        dict: Comparison results including mean difference, max difference, etc.
    """
    print("\n🔍 COMPARING MODEL ACCURACY BEFORE AND AFTER QUANTIZATION...")

    original_model.eval()
    quantized_model.eval()

    differences = []
    max_diffs = []

    with torch.no_grad():
        for i, data in enumerate(test_data):
            if i >= 5:  # Limit to 5 samples for comparison
                break

            # Move data to device
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            try:
                # Get outputs from both models
                original_output = original_model(**data)
                quantized_output = quantized_model(**data)

                # Calculate difference
                if isinstance(original_output, torch.Tensor) and isinstance(
                    quantized_output, torch.Tensor
                ):
                    diff = torch.abs(original_output - quantized_output)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()
                    # Compute cosine similarity
                    orig_flat = original_output.flatten()
                    quant_flat = quantized_output.flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
                    ).item()
                    print(f"Cosine similarity: {cos_sim:.6f}")

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)
                    # INSERT_YOUR_CODE
                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
                    # Print the position and the original value of the max diff
                    max_diff_pos = (diff == torch.max(diff)).nonzero(as_tuple=True)
                    if len(max_diff_pos[0]) > 0:
                        idx = tuple(pos[0].item() for pos in max_diff_pos)
                        print(f"Position of max diff: {idx}")
                        print(
                            f"Original value at this position: {original_output[idx] if isinstance(original_output, torch.Tensor) else original_output.logits[idx]}"
                        )

                elif hasattr(original_output, "logits") and hasattr(quantized_output, "logits"):
                    # Handle model outputs with logits
                    diff = torch.abs(original_output.logits - quantized_output.logits)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)

                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")

            except Exception as e:
                print(f"Error comparing sample {i+1}: {e}")
                continue

    if differences:
        avg_mean_diff = sum(differences) / len(differences)
        avg_max_diff = sum(max_diffs) / len(max_diffs)
        overall_max_diff = max(max_diffs)

        print("\n📊 QUANTIZATION ACCURACY SUMMARY:")
        print(f"Average mean difference: {avg_mean_diff:.6f}")
        print(f"Average max difference: {avg_max_diff:.6f}")
        print(f"Overall max difference: {overall_max_diff:.6f}")

        # Determine if quantization is acceptable
        if avg_mean_diff < 0.01:
            print("✅ Quantization accuracy is excellent (< 0.01)")
        elif avg_mean_diff < 0.1:
            print("✅ Quantization accuracy is good (< 0.1)")
        elif avg_mean_diff < 0.5:
            print("⚠️  Quantization accuracy is acceptable (< 0.5)")
        else:
            print("❌ Quantization accuracy is poor (> 0.5)")

        return {
            "avg_mean_diff": avg_mean_diff,
            "avg_max_diff": avg_max_diff,
            "overall_max_diff": overall_max_diff,
            "num_samples": len(differences),
        }
    else:
        print("❌ No valid comparisons could be made")
        return None


def quantize_llm(
    model,
    precision,
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    full_layer_quant=False,
):
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in [
        "nvfp4",
        "fp8",
    ], f"Only nvfp4 (W4A4) and fp8 are supported. You passed an unsupported precision: {precision}."

    # Configure quantization based on precision
    if precision == "nvfp4":
        # NVFP4_AWQ configs usually have better accuracy. You may also try other configs.
        assert hasattr(mtq, "NVFP4_AWQ_LITE_CFG")
        quant_cfg = mtq.NVFP4_AWQ_FULL_CFG
    else:  # fp8
        # FP8 quantization configuration
        quant_cfg = mtq.FP8_DEFAULT_CFG

    # Apply layer-specific configurations for nvfp4
    if precision == "nvfp4" and not full_layer_quant:
        # Disable quantization for specific layers when full_layer_quant is False
        print("Using selective layer quantization (disabling down_proj and o_proj)")
        quant_cfg["quant_cfg"][
            "eagle_model.language_model.model.layers.*.mlp.down_proj.*_quantizer"
        ] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        }
        quant_cfg["quant_cfg"][
            "eagle_model.language_model.model.layers.*.self_attn.o_proj.*_quantizer"
        ] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        }
    elif precision == "nvfp4" and full_layer_quant:
        print("Using full layer quantization (all layers enabled)")

    print(f"Quantization configuration: {quant_cfg}")

    # Create the dataset and dataloader
    if dataset_path is None or modality_configs is None:
        raise ValueError("LLM quantization requires valid dataset_path and modality_configs.")

    print(f"Using LeRobot dataset for calibration: {dataset_path}")
    # Deep copy the policy to avoid any modifications during calibration
    # policy_copy = copy.deepcopy(policy)
    # load the policy
    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()
    device = "cuda"
    policy_copy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    dataset = LLMCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        policy=policy_copy,
        calib_size=calib_size,
        video_backend=video_backend,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    # Store original model for comparison
    original_model = None
    if compare_accuracy:
        print("📋 Creating backup of original model for accuracy comparison...")
        original_model = copy.deepcopy(model)
        original_model.eval()

    # Quantize the model
    quantized_model = _quantize_model(model, data_loader, quant_cfg)
    mtq.print_quant_summary(quantized_model)
    # Compare accuracy if requested
    if compare_accuracy and original_model is not None:
        print("\n🔍 Starting accuracy comparison...")
        try:
            # Create a test dataset (use first few samples from calibration data)
            test_data = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=no_batch_collate_fn
            )
            comparison_results = compare_model_accuracy(original_model, quantized_model, test_data)

            if comparison_results:
                print("\n📈 FINAL QUANTIZATION RESULTS:")
                print("✅ Quantization completed successfully!")
                print(f"📊 Accuracy comparison: {comparison_results['num_samples']} samples tested")
                print(f"📊 Average mean difference: {comparison_results['avg_mean_diff']:.6f}")
                print(f"📊 Overall max difference: {comparison_results['overall_max_diff']:.6f}")
            else:
                print("⚠️  Accuracy comparison failed, but quantization completed")

        except Exception as e:
            print(f"⚠️  Accuracy comparison failed: {e}")
            print("✅ Quantization completed successfully despite comparison failure")

    return quantized_model


def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    normalized_input = unsqueeze_dict_values
    # Apply transforms
    normalized_input = policy.apply_transforms(observations)

    return normalized_input["eagle_attention_mask"], normalized_input["state"]


def export_eagle2_vit(
    vision_model,
    output_dir,
    vit_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    policy=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
):
    class SiglipVisionEmbeddingsOpt(SiglipVisionEmbeddings):
        def __init__(self, config):
            super().__init__(config)

        def forward(
            self,
            pixel_values: torch.FloatTensor,
            position_ids: torch.LongTensor,  # position_ids is now an input
            interpolate_pos_encoding=False,
        ) -> torch.Tensor:
            _, _, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            patch_embeds = self.patch_embedding(
                pixel_values.to(dtype=target_dtype)
            )  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

    class SiglipVisionTransformerOpt(SiglipVisionTransformer):
        def __init__(self, config: SiglipVisionConfig):
            config._attn_implementation = ATTN_IMPLEMENTATION
            super().__init__(config)
            self.embeddings = SiglipVisionEmbeddingsOpt(config)

        def forward(
            self,
            pixel_values,
            position_ids,  # Pass position_ids as input
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = False,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            hidden_states = self.embeddings(
                pixel_values,
                position_ids=position_ids,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state

    model = SiglipVisionTransformerOpt(vision_model.config).to(torch.float16)
    model.load_state_dict(vision_model.state_dict())
    model.eval().cuda()

    # Quantize ViT if requested
    if vit_dtype == "fp8":
        print("Quantizing Eagle2 ViT to fp8")
        model = quantize_vit(
            model,
            precision="fp8",
            calib_size=calib_size,
            dataset_path=calib_dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
        )

    # Get the number of video views from modality_configs
    num_video_views = 1
    if modality_configs is not None and "video" in modality_configs:
        num_video_views = len(modality_configs["video"].modality_keys)
        print(f"Number of video views detected from modality config: {num_video_views}")
    else:
        print(f"Using default number of video views: {num_video_views}")

    pixel_values = torch.randn(
        (
            num_video_views,
            model.config.num_channels,
            model.config.image_size,
            model.config.image_size,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    position_ids = torch.arange(model.embeddings.num_patches, device="cuda").expand(
        (num_video_views, -1)
    )

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, position_ids),  # Include position_ids in ONNX export
            f"{output_dir}/eagle2/vit_{vit_dtype}.onnx",
            input_names=["pixel_values", "position_ids"],  # Add position_ids to input names
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )


def export_eagle2_llm(
    backbone_model,
    backbone_config,
    output_dir,
    attention_mask,
    llm_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    policy=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
    full_layer_quant=False,
):
    class EagleBackboneOpt(EagleBackbone):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Modify LlamamModel architecture for ONNX export
            config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
            config._attn_implementation = ATTN_IMPLEMENTATION

            assert config.text_config.architectures[0] == "Qwen3ForCausalLM"
            self.eagle_model.language_model = Qwen3ForCausalLM(config.text_config)

            # # remove parts of the LLM
            while len(self.eagle_model.language_model.model.layers) > kwargs["select_layer"]:
                self.eagle_model.language_model.model.layers.pop(-1)

        def forward(self, inputs_embeds, attention_mask):
            outputs = self.eagle_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            eagle_features = outputs.hidden_states[self.select_layer]

            eagle_features = self.eagle_linear(eagle_features)
            return eagle_features

    model = EagleBackboneOpt(**backbone_config).to(torch.float16)
    model.load_state_dict(backbone_model.state_dict())
    model.eval().cuda()

    if llm_dtype in ["nvfp4", "fp8"]:
        print(f"Quantizing Eagle2 LLM to {llm_dtype}")

        model = quantize_llm(
            model,
            llm_dtype,
            calib_size=calib_size,
            dataset_path=calib_dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
            full_layer_quant=full_layer_quant,
        )

        # This is required for nvfp4 ONNX export
        if llm_dtype == "nvfp4":
            from modelopt.torch.quantization.utils import is_quantized_linear

            for module in model.modules():
                assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
                if isinstance(module, torch.nn.Linear):
                    module.input_quantizer._trt_high_precision_dtype = "Half"
                    module.input_quantizer._onnx_quantizer_type = "dynamic"
                    module.weight_quantizer._onnx_quantizer_type = "static"

    inputs_embeds = torch.randn(
        (
            1,
            attention_mask.shape[1],
            model.eagle_model.language_model.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    attention_mask = torch.ones((1, attention_mask.shape[1]), dtype=torch.int64).cuda()

    # Use different filename for full layer quantization
    llm_dtype_suffix = (
        f"{llm_dtype}_full" if (llm_dtype == "nvfp4" and full_layer_quant) else llm_dtype
    )
    onnx_path = f"{output_dir}/eagle2/llm_{llm_dtype_suffix}.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    start_time = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inputs_embeds, attention_mask),
            onnx_path,
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
        )
    end_time = time.time()
    print(
        f"Eagle2 LLM ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {onnx_path}."
    )

    if llm_dtype == "nvfp4":
        print("Converting nvfp4 ONNX model to 2dq")
        import onnx
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)

        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"llm_{llm_dtype}.onnx_data",
            convert_attribute=True,
        )


class VLLN_VLSelfAttention(torch.nn.Module):
    def __init__(self, vlln, vl_self_attention):
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features):
        x = self.vlln(backbone_features)
        x = self.vl_self_attention(x)
        return x


def export_action_head(
    policy,
    ONNX_export_path,
    input_state,
    attention_mask,
    dit_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
):
    """
    Export the action head models to ONNX format with optional DiT quantization.

    Args:
        policy: Gr00tPolicy instance
        ONNX_export_path: Path to save ONNX models
        input_state: Input state tensor
        attention_mask: Attention mask tensor
        dit_dtype: Data type for DiT export (fp16 or fp8 for quantization)
        calib_dataset_path: Path to LeRobot dataset for calibration
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        calib_size: Number of calibration samples
    """
    process_backbone_model = (
        VLLN_VLSelfAttention(
            policy.model.action_head.vlln, policy.model.action_head.vl_self_attention
        )
        .to(torch.float16)
        .cuda()
    )
    backbone_features = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=torch.float16,
    ).cuda()

    torch.onnx.export(
        process_backbone_model,
        (backbone_features),
        os.path.join(ONNX_export_path, "action_head/vlln_vl_self_attention.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["backbone_features"],
        output_names=["output"],
        dynamic_axes={
            "backbone_features": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )

    state_encoder = policy.model.action_head.state_encoder.to(torch.float16)

    state_tensor = torch.randn(
        (1, input_state.shape[1], input_state.shape[2]), dtype=torch.float16
    ).cuda()
    embodiment_id_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/state_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_encoder = policy.model.action_head.action_encoder.to(torch.float16)
    actions_tensor = torch.randn(
        (
            1,
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.action_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    timesteps_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # DiT model with optional FP8 quantization
    DiT = policy.model.action_head.model.to(torch.float16).cuda()

    # Quantize DiT if requested
    if dit_dtype == "fp8":
        print("Quantizing DiT to fp8")
        # Use a default dataset path if None
        dataset_path_for_calib = (
            calib_dataset_path if calib_dataset_path is not None else "dummy_path"
        )
        DiT = quantize_dit(
            DiT,
            precision="fp8",
            calib_size=calib_size,
            dataset_path=dataset_path_for_calib,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            attention_mask=attention_mask,
            input_state=input_state,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
        )

    sa_embs_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.input_embedding_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    vl_embs_tensor = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=torch.float16,
    ).cuda()

    onnx_path = os.path.join(ONNX_export_path, f"action_head/DiT_{dit_dtype}.onnx")
    torch.onnx.export(
        DiT,
        (sa_embs_tensor, vl_embs_tensor, timesteps_tensor),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            "vl_embs": {0: "batch_size", 1: "sequence_length"},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"DiT ONNX exported to {onnx_path}")

    action_decoder = policy.model.action_head.action_decoder.to(torch.float16)
    model_output_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    torch.onnx.export(
        action_decoder,
        (model_output_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_decoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def run_groot_inference(
    dataset_path: str,
    model_path: str,
    onnx_model_path: str,
    data_config: str = "fourier_gr1_arms_only",
    embodiment_tag: str = "gr1",
    denoising_steps: int = 4,
    device: str = "cuda",
    llm_dtype: str = "fp16",
    vit_dtype: str = "fp16",
    dit_dtype: str = "fp16",
    calib_dataset_path: str = None,
    calib_size: int = 10,
    video_backend: str = "decord",
    full_layer_quant: bool = False,
) -> Dict[str, float]:

    # load the policy
    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    modality_config = policy.modality_config
    # load the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
    )

    step_data = dataset[0]
    # get the action
    predicted_action = policy.get_action(step_data)

    attention_mask, state = get_input_info(policy, step_data)
    # export onnx
    os.makedirs(onnx_model_path, exist_ok=True)
    os.makedirs(os.path.join(onnx_model_path, "eagle2"), exist_ok=True)
    os.makedirs(os.path.join(onnx_model_path, "action_head"), exist_ok=True)

    export_eagle2_vit(
        policy.model.backbone.eagle_model.vision_model.vision_model,
        onnx_model_path,
        vit_dtype=vit_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        policy=policy,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
    )
    export_eagle2_llm(
        policy.model.backbone,
        policy.model.config.backbone_cfg,
        onnx_model_path,
        attention_mask,
        llm_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        policy=policy,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
        full_layer_quant=full_layer_quant,
    )
    export_action_head(
        policy,
        onnx_model_path,
        state,
        attention_mask,
        dit_dtype=dit_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
    )

    return predicted_action


if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--onnx-model-path",
        type=str,
        help="Path where the ONNX model will be stored",
        default=os.path.join(os.getcwd(), "gr00t_onnx"),
    )

    parser.add_argument(
        "--data-config",
        type=str,
        help="The name of the data config to use (e.g. fourier_gr1_arms_only, fourier_gr1_arms_waist, unitree_g1, etc.) or a path to a custom data config file",
        default="fourier_gr1_arms_only",
    )

    parser.add_argument(
        "--embodiment-tag",
        type=str,
        help="The embodiment tag for the model (e.g. gr1, g1, so100, etc.)",
        default="gr1",
    )

    parser.add_argument(
        "--llm-dtype",
        type=str,
        choices=["fp16", "nvfp4", "fp8"],
        help="Data type for LLM export (fp16, nvfp4, or fp8 for quantization)",
        default="nvfp4",
    )

    parser.add_argument(
        "--vit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="Data type for ViT export (fp16 or fp8 for quantization)",
        default="fp8",
    )

    parser.add_argument(
        "--dit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="Data type for DiT (Diffusion Transformer) export (fp16 or fp8 for quantization)",
        default="fp8",
    )

    parser.add_argument(
        "--calib-dataset-path",
        type=str,
        help="Path to the LeRobot dataset for calibration (if different from dataset_path)",
        default=None,
    )

    parser.add_argument(
        "--calib-size",
        type=int,
        help="Number of calibration samples to use",
        default=10,
    )

    parser.add_argument(
        "--denoising-steps",
        type=int,
        help="Number of denoising steps for diffusion model inference",
        default=4,
    )

    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["decord", "torchcodec"],
        help="Video backend to use for loading video frames",
        default="decord",
    )

    parser.add_argument(
        "--full-layer-quant",
        action="store_true",
        help="Enable full layer nvfp4 quantization for LLM (default: False, which disables down_proj and o_proj layers)",
    )

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"ONNX model path: {args.onnx_model_path}")
    print(f"Data config: {args.data_config}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Denoising steps: {args.denoising_steps}")
    print(f"Video backend: {args.video_backend}")
    print(f"LLM data type: {args.llm_dtype}")
    print(f"ViT data type: {args.vit_dtype}")
    print(f"DiT data type: {args.dit_dtype}")
    print(f"Calibration dataset path: {args.calib_dataset_path or args.dataset_path}")
    print(f"Calibration size: {args.calib_size}")
    print(f"Full layer quantization: {args.full_layer_quant}")

    predicted_action = run_groot_inference(
        args.dataset_path,
        args.model_path,
        args.onnx_model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        llm_dtype=args.llm_dtype,
        vit_dtype=args.vit_dtype,
        dit_dtype=args.dit_dtype,
        calib_dataset_path=args.calib_dataset_path,
        calib_size=args.calib_size,
        video_backend=args.video_backend,
        full_layer_quant=args.full_layer_quant,
    )

    for key, value in predicted_action.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
