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

"""TensorRT forward functions for GR00T N1.6 full-pipeline inference.

This module provides TRT-accelerated forward functions that replace the
PyTorch backbone and action head during inference. Adapted from N1.5
deployment_scripts/trt_model_forward.py for the N1.6 architecture.

Architecture (full_pipeline mode):
  Backbone: ViT (TRT) → pixel_shuffle_back (PyTorch) → MLP1 (PyTorch)
            → embed + scatter (PyTorch) → LLM (TRT)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]

Note: ViT TRT is optional — if vit_bf16.engine is not found, falls back to
PyTorch ViT.
"""

from functools import partial
import logging
import os
import sys

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


# Ensure sibling modules are importable (scripts/deployment is not a package)
_deploy_dir = os.path.dirname(os.path.abspath(__file__))
if _deploy_dir not in sys.path:
    sys.path.insert(0, _deploy_dir)
from trt_torch import Engine  # noqa: E402


# ============================================================
# Backbone TRT Forward
# ============================================================


def eagle3_tensorrt_forward(self, vl_input):
    """Replace EagleBackbone.forward() with TRT-accelerated ViT + LLM.

    When a ViT TRT engine is available, it replaces the PyTorch ViT.
    pixel_shuffle_back and MLP1 remain in PyTorch (small ops).
    The LLM is always replaced with a TRT engine.

    Args:
        self: EagleBackbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values
    """
    self.set_frozen_modules_to_eval_mode()

    eagle_model = self.model  # Eagle3_VLForConditionalGeneration

    # Extract only the keys we need
    keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    pixel_values = vl_input["pixel_values"]
    engine_dtype = torch.bfloat16

    if hasattr(self, "vit_engine") and self.vit_engine is not None:
        # --- ViT TRT Engine ---
        # pixel_values is a list of tensors from the processor; stack into single tensor
        if isinstance(pixel_values, (list, tuple)):
            pv = torch.cat(pixel_values, dim=0)  # (B, 3, H, H)
        else:
            pv = pixel_values
        if pv.dtype != engine_dtype:
            pv = pv.to(engine_dtype)

        self.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
        vit_embeds = self.vit_engine(pv)["vit_embeds"]  # (B, num_patches, 1152)

        # pixel_shuffle_back + MLP1 (stay in PyTorch — small ops)
        # Derive patches_per_side from pixel_values shape and saved patch_size
        patches_per_side = pv.shape[-1] // self.patch_size  # e.g. 252/14=18
        spatial_shapes = torch.tensor(
            [[patches_per_side, patches_per_side]] * vit_embeds.shape[0],
            device=vit_embeds.device,
            dtype=torch.int32,
        )
        vit_embeds, spatial_shapes = self._pixel_shuffle_back(vit_embeds, spatial_shapes)
        vit_embeds = self._mlp1(vit_embeds)
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)
    else:
        # --- ViT + pixel_shuffle_back + MLP1 (all PyTorch) ---
        vit_embeds = eagle_model.extract_feature(pixel_values)
    # vit_embeds: [B*N, C] (flattened across batch and spatial dims)

    # --- Create input embeddings and scatter vision tokens ---
    input_ids = vl_input["input_ids"]
    input_embeds = self.embedding_layer(input_ids)

    # Ensure consistent dtype for TRT engine
    engine_dtype = torch.bfloat16
    if input_embeds.dtype != engine_dtype:
        input_embeds = input_embeds.to(engine_dtype)
    if vit_embeds.dtype != engine_dtype:
        vit_embeds = vit_embeds.to(engine_dtype)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids_flat = input_ids.reshape(B * N)
    selected = input_ids_flat == self.image_token_index
    n_image_tokens = selected.sum().item()
    vit_embeds_flat = vit_embeds.reshape(-1, C)
    n_vit_tokens = vit_embeds_flat.shape[0]

    if n_image_tokens != n_vit_tokens:
        logger.warning(
            f"Vision token count mismatch: {n_image_tokens} image_token slots "
            f"vs {n_vit_tokens} ViT output tokens. Truncating to min."
        )
        n_fill = min(n_image_tokens, n_vit_tokens)
        input_embeds[selected] = input_embeds[selected] * 0.0
        input_embeds[selected][:n_fill] = vit_embeds_flat[:n_fill]
    else:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat

    input_embeds = input_embeds.reshape(B, N, C)

    # --- LLM TRT Engine ---
    attention_mask = vl_input["attention_mask"]
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)

    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", input_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    embeddings = self.llm_engine(input_embeds, attention_mask)["embeddings"]

    # Build output (same as original EagleBackbone.forward)
    image_mask = input_ids == self.image_token_index
    backbone_attention_mask = attention_mask == 1

    return BatchFeature(
        data={
            "backbone_features": embeddings,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
        }
    )


# ============================================================
# Action Head TRT Forward
# ============================================================


def action_head_tensorrt_forward(self, backbone_output, action_input):
    """Replace Gr00tN1d6ActionHead.get_action() with TRT-accelerated inference.

    VLLN (LayerNorm) stays in PyTorch. State Encoder, Action Encoder,
    DiT, and Action Decoder are replaced with TRT engines.

    Args:
        self: Gr00tN1d6ActionHead instance (monkey-patched)
        backbone_output: BatchFeature with backbone_features, backbone_attention_mask, image_mask
        action_input: BatchFeature with state, embodiment_id
    """
    # --- VLLN: LayerNorm in PyTorch (too small for TRT) ---
    backbone_features = backbone_output.backbone_features
    backbone_features = self.vlln(backbone_features)
    vl_embs = backbone_features

    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]
    device = vl_embs.device

    engine_dtype = torch.bfloat16

    # Ensure consistent dtypes
    if vl_embs.dtype != engine_dtype:
        vl_embs = vl_embs.to(engine_dtype)
    if action_input.state.dtype != engine_dtype:
        action_input.state = action_input.state.to(engine_dtype)
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    # --- State Encoder TRT ---
    self.state_encoder_engine.set_runtime_tensor_shape("state", action_input.state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(action_input.state, embodiment_id)["output"]

    # --- Initialize actions as random noise ---
    if hasattr(self, "init_actions"):
        actions = self.init_actions.expand((batch_size, -1, -1))
    else:
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=engine_dtype,
            device=device,
        )

    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    # --- Denoising loop ---
    for t in range(num_steps):
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * self.num_timestep_buckets)

        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.int64
        )

        # Action Encoder TRT
        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("timesteps", timesteps_tensor.shape)
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(
            actions.to(engine_dtype), timesteps_tensor, embodiment_id
        )["output"]

        # Maybe add position embedding (stays in PyTorch)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(engine_dtype)
            action_features = action_features + pos_embs

        # Concatenate state + action embeddings
        sa_embs = torch.cat((state_features, action_features), dim=1).to(engine_dtype)

        # DiT TRT
        self.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.dit_engine.set_runtime_tensor_shape("timestep", timesteps_tensor.shape)

        dit_kwargs = {}
        if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
            image_mask = backbone_output.image_mask
            self.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
            dit_kwargs["image_mask"] = image_mask

        if (
            hasattr(backbone_output, "backbone_attention_mask")
            and backbone_output.backbone_attention_mask is not None
        ):
            bb_mask = backbone_output.backbone_attention_mask
            self.dit_engine.set_runtime_tensor_shape("backbone_attention_mask", bb_mask.shape)
            dit_kwargs["backbone_attention_mask"] = bb_mask

        model_output = self.dit_engine(sa_embs, vl_embs, timesteps_tensor, **dit_kwargs)["output"]

        # Action Decoder TRT
        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        pred_velocity = pred[:, -self.action_horizon :]

        # Euler integration
        actions = actions + dt * pred_velocity

    return BatchFeature(data={"action_pred": actions})


# ============================================================
# Engine Setup
# ============================================================


def setup_tensorrt_engines(policy, trt_engine_path, mode="full_pipeline"):
    """Load TRT engines, delete PyTorch modules, and monkey-patch forward methods.

    Args:
        policy: Gr00tPolicy instance
        trt_engine_path: Path to directory containing TRT engine files
        mode: 'full_pipeline' or 'dit_only'
    """
    if mode == "full_pipeline":
        _setup_full_pipeline(policy, trt_engine_path)
    elif mode == "dit_only":
        _setup_dit_only(policy, trt_engine_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'full_pipeline' or 'dit_only'.")


def _setup_full_pipeline(policy, trt_engine_path):
    """Set up all TRT engines for full-pipeline acceleration."""
    backbone = policy.model.backbone
    eagle_model = backbone.model  # Eagle3_VLForConditionalGeneration
    action_head = policy.model.action_head

    # --- Backbone setup ---
    # Save embedding layer and image token index before deleting the language model
    backbone.embedding_layer = eagle_model.language_model.get_input_embeddings()
    backbone.image_token_index = eagle_model.image_token_index

    # Check if ViT TRT engine is available
    vit_engine_path = os.path.join(trt_engine_path, "vit_bf16.engine")

    # Save pixel_shuffle_back, mlp1, and patch_size before potentially deleting vision_model
    backbone._pixel_shuffle_back = eagle_model.pixel_shuffle_back
    backbone._mlp1 = eagle_model.mlp1
    backbone.patch_size = eagle_model.vision_model.vision_model.config.patch_size

    if os.path.exists(vit_engine_path):
        print(f"Loading ViT engine: {vit_engine_path}")
        backbone.vit_engine = Engine(vit_engine_path)

        del eagle_model.vision_model
        torch.cuda.empty_cache()
        print("  Deleted PyTorch ViT (replaced by TRT engine)")
    else:
        backbone.vit_engine = None
        print(f"  ViT engine not found at {vit_engine_path}, using PyTorch ViT")

    # Delete the language model to free GPU memory
    del eagle_model.language_model
    torch.cuda.empty_cache()

    # Load LLM TRT engine
    llm_engine_path = os.path.join(trt_engine_path, "llm_bf16.engine")
    print(f"Loading LLM engine: {llm_engine_path}")
    backbone.llm_engine = Engine(llm_engine_path)

    # --- Action head setup ---
    # Delete PyTorch modules that are replaced by TRT
    if hasattr(action_head, "model"):
        del action_head.model
    if hasattr(action_head, "state_encoder"):
        del action_head.state_encoder
    if hasattr(action_head, "action_encoder"):
        del action_head.action_encoder
    if hasattr(action_head, "action_decoder"):
        del action_head.action_decoder
    torch.cuda.empty_cache()

    # Verify action_dim consistency (export uses config.max_action_dim)
    assert action_head.action_dim == action_head.config.max_action_dim, (
        f"action_dim mismatch: action_head.action_dim={action_head.action_dim} "
        f"!= config.max_action_dim={action_head.config.max_action_dim}"
    )

    # Load action head TRT engines
    print(f"Loading action head engines from: {trt_engine_path}")
    action_head.state_encoder_engine = Engine(os.path.join(trt_engine_path, "state_encoder.engine"))
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    action_head.dit_engine = Engine(os.path.join(trt_engine_path, "dit_bf16.engine"))
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )

    # Monkey-patch forward methods
    backbone.forward = partial(eagle3_tensorrt_forward, backbone)
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    print("Full-pipeline TRT engines loaded and forward methods patched.")


def _setup_dit_only(policy, trt_engine_path):
    """Set up TRT engine for DiT-only acceleration (backward compatible).

    Only replaces the DiT model in the action head. The backbone and other
    action head components remain in PyTorch.
    """
    action_head = policy.model.action_head

    # Delete the PyTorch DiT model
    if hasattr(action_head, "model"):
        del action_head.model
    torch.cuda.empty_cache()

    # Load DiT TRT engine
    # Support both naming conventions
    dit_path = os.path.join(trt_engine_path, "dit_bf16.engine")
    if not os.path.exists(dit_path):
        dit_path = os.path.join(trt_engine_path, "dit_model_bf16.engine")
    if not os.path.exists(dit_path):
        # Try the old naming convention
        dit_path = os.path.join(trt_engine_path, "dit_model_bf16.trt")

    print(f"Loading DiT engine: {dit_path}")
    action_head.dit_engine = Engine(dit_path)

    # Monkey-patch only the get_action method
    # We need a simpler forward that only replaces the DiT call
    @torch.no_grad()
    def dit_only_get_action_with_features(
        backbone_features, state_features, embodiment_id, backbone_output
    ):
        """get_action_with_features with DiT replaced by TRT."""
        vl_embs = backbone_features
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        engine_dtype = torch.bfloat16

        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        dt = 1.0 / action_head.num_inference_timesteps

        for t in range(action_head.num_inference_timesteps):
            t_cont = t / float(action_head.num_inference_timesteps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1).to(engine_dtype)

            # Use TRT for DiT
            vl_embs_trt = vl_embs.to(engine_dtype)
            timesteps_trt = timesteps_tensor.to(torch.int64)

            action_head.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            action_head.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs_trt.shape)
            action_head.dit_engine.set_runtime_tensor_shape("timestep", timesteps_trt.shape)

            dit_kwargs = {}
            if hasattr(backbone_output, "image_mask") and backbone_output.image_mask is not None:
                image_mask = backbone_output.image_mask
                action_head.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
                dit_kwargs["image_mask"] = image_mask

            if (
                hasattr(backbone_output, "backbone_attention_mask")
                and backbone_output.backbone_attention_mask is not None
            ):
                bb_mask = backbone_output.backbone_attention_mask
                action_head.dit_engine.set_runtime_tensor_shape(
                    "backbone_attention_mask", bb_mask.shape
                )
                dit_kwargs["backbone_attention_mask"] = bb_mask

            model_output = action_head.dit_engine(
                sa_embs, vl_embs_trt, timesteps_trt, **dit_kwargs
            )["output"]

            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embs,
                "state_features": state_features,
            }
        )

    action_head.get_action_with_features = dit_only_get_action_with_features
    print("DiT-only TRT engine loaded and forward method patched.")
