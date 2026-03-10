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

"""TensorRT forward functions for GR00T N1.7 inference.

This module provides TRT-accelerated forward functions that replace the
PyTorch backbone and action head during inference.

Architecture (n17_full_pipeline mode):
  Backbone: ViT (TRT) → embed_tokens + masked_scatter + get_rope_index (PyTorch)
            → LLM (TRT, with deepstack injection)
  Action Head: VLLN (PyTorch) → State Encoder (TRT) → denoising loop:
               [ Action Encoder (TRT) → DiT (TRT) → Action Decoder (TRT) ]

Architecture (action_head mode):
  Backbone: stays in PyTorch (Qwen3-VL)
  Action Head: same as above
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
# N1.7 Backbone TRT Forward (ViT TRT + LLM TRT)
# ============================================================


def _qwen3_vit_and_scatter(self, vl_input):
    """Shared logic: ViT TRT + embed_tokens + scatter + get_rope_index.

    Returns all inputs needed by either PyTorch LLM or LLM TRT engine.
    These ops stay in PyTorch because they involve dynamic Python logic
    (get_rope_index, masked_scatter, get_placeholder_mask).
    """
    qwen_model = self.model  # Qwen3VLForConditionalGeneration
    inner_model = qwen_model.model  # Qwen3VLModel

    pixel_values = vl_input["pixel_values"]
    grid_thw = vl_input["image_grid_thw"]
    engine_dtype = torch.bfloat16

    # --- ViT TRT Engine ---
    if isinstance(pixel_values, (list, tuple)):
        pv = torch.cat(pixel_values, dim=0)
    else:
        pv = pixel_values
    if pv.dtype != engine_dtype:
        pv = pv.to(engine_dtype)

    self.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
    vit_result = self.vit_engine(pv)
    image_embeds = vit_result["image_embeds"]
    deepstack_features = vit_result.get("deepstack_features")

    # Unpack deepstack: [num_layers, N, D] → list of [N, D]
    deepstack_list = []
    if deepstack_features is not None and deepstack_features.numel() > 1:
        deepstack_list = list(deepstack_features.unbind(0))

    # --- PyTorch: embed_tokens + scatter ---
    input_ids = vl_input["input_ids"]
    inputs_embeds = self._embedding_layer(input_ids)

    if inputs_embeds.dtype != engine_dtype:
        inputs_embeds = inputs_embeds.to(engine_dtype)
    if image_embeds.dtype != engine_dtype:
        image_embeds = image_embeds.to(engine_dtype)

    image_embeds_cat = torch.cat([image_embeds], dim=0)
    image_mask, _ = inner_model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

    visual_pos_masks = image_mask[..., 0] if image_mask is not None else None

    # Compute 3D position IDs (stays in PyTorch — complex Python logic)
    attention_mask = vl_input["attention_mask"]
    position_ids, rope_deltas = inner_model.get_rope_index(
        input_ids, grid_thw, video_grid_thw=None, attention_mask=attention_mask
    )
    inner_model.rope_deltas = rope_deltas

    image_mask_out = input_ids == self._image_token_id
    backbone_attention_mask = attention_mask == 1

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_list": deepstack_list,
        "image_mask_out": image_mask_out,
        "backbone_attention_mask": backbone_attention_mask,
    }


def qwen3_backbone_tensorrt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + PyTorch LLM.

    ViT is replaced with a TRT engine. The LLM stays in PyTorch.
    Used when LLM TRT engine is not available.

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    qwen_model = self.model
    inner_model = qwen_model.model

    # LLM forward (PyTorch)
    outputs = inner_model.language_model(
        input_ids=None,
        position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"],
        inputs_embeds=prepared["inputs_embeds"],
        visual_pos_masks=prepared["visual_pos_masks"],
        deepstack_visual_embeds=prepared["deepstack_list"] or None,
        output_hidden_states=True,
    )

    return BatchFeature(
        data={
            "backbone_features": outputs.last_hidden_state,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


def qwen3_backbone_full_trt_forward(self, vl_input):
    """Replace Qwen3Backbone.forward() with ViT TRT + LLM TRT.

    Both ViT and LLM are replaced with TRT engines.
    PyTorch ops kept: embed_tokens, masked_scatter, get_rope_index (lightweight).

    Args:
        self: Qwen3Backbone instance (monkey-patched)
        vl_input: BatchFeature with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    self.set_frozen_modules_to_eval_mode()

    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    vl_input = {k: vl_input[k] for k in keys_to_use}

    prepared = _qwen3_vit_and_scatter(self, vl_input)

    engine_dtype = torch.bfloat16
    inputs_embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    position_ids = prepared["position_ids"]

    if inputs_embeds.dtype != engine_dtype:
        inputs_embeds = inputs_embeds.to(engine_dtype)
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)
    if position_ids.dtype != torch.int64:
        position_ids = position_ids.to(torch.int64)

    # Set LLM engine input shapes
    self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
    self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)

    llm_kwargs = {}

    # Visual pos masks and deepstack features
    visual_pos_masks = prepared["visual_pos_masks"]
    deepstack_list = prepared["deepstack_list"]

    if visual_pos_masks is not None and deepstack_list:
        self.llm_engine.set_runtime_tensor_shape("visual_pos_masks", visual_pos_masks.shape)
        llm_kwargs["visual_pos_masks"] = visual_pos_masks

        for i, ds in enumerate(deepstack_list):
            name = f"deepstack_{i}"
            if ds.dtype != engine_dtype:
                ds = ds.to(engine_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, ds.shape)
            llm_kwargs[name] = ds

    backbone_features = self.llm_engine(inputs_embeds, attention_mask, position_ids, **llm_kwargs)[
        "embeddings"
    ]

    return BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": prepared["backbone_attention_mask"],
            "image_mask": prepared["image_mask_out"],
        }
    )


# ============================================================
# Action Head TRT Forward
# ============================================================


def action_head_tensorrt_forward(self, backbone_output, action_input, options=None):
    """Replace ActionHead.get_action() with TRT-accelerated inference.
    VLLN (LayerNorm) stays in PyTorch. State Encoder, Action Encoder,
    DiT, and Action Decoder are replaced with TRT engines.

    N1.7 change: state is reshaped from [B, state_history_length, max_state_dim]
    to [B, 1, state_history_length * max_state_dim] before the state encoder.

    Args:
        self: ActionHead instance (monkey-patched)
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

    # --- State history reshape (N1.7) ---
    # N1.7: state comes as [B, state_history_length, max_state_dim]
    # Flatten to [B, 1, state_history_length * max_state_dim] for the encoder
    state = action_input.state
    if state.ndim == 3 and state.shape[1] > 1:
        state = state.view(state.shape[0], 1, -1)
    elif state.ndim == 3 and state.shape[1] == 1:
        # Already [B, 1, dim] — state_history_length=1
        pass
    else:
        # Unexpected shape, pass through
        logger.warning(f"Unexpected state shape: {state.shape}")

    # --- State Encoder TRT ---
    self.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(state, embodiment_id)["output"]

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


def setup_tensorrt_engines(policy, trt_engine_path, mode="n17_full_pipeline"):
    """Load TRT engines, delete PyTorch modules, and monkey-patch forward methods.

    Args:
        policy: Gr00tPolicy instance
        trt_engine_path: Path to directory containing TRT engine files
        mode: 'n17_full_pipeline' (ViT TRT + LLM TRT + Action Head TRT),
              'action_head' (Action Head TRT only), or 'dit_only'
    """
    if mode == "n17_full_pipeline":
        _setup_n17_full_pipeline(policy, trt_engine_path)
    elif mode == "action_head":
        _setup_action_head(policy, trt_engine_path)
    elif mode == "dit_only":
        _setup_dit_only(policy, trt_engine_path)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Expected 'n17_full_pipeline', 'action_head', or 'dit_only'."
        )


def _setup_n17_full_pipeline(policy, trt_engine_path):
    """Set up TRT engines for N1.7: ViT TRT + LLM TRT + Action Head TRT.

    The Qwen3-VL backbone's vision encoder and text model are both replaced
    with TRT engines. PyTorch ops kept: embed_tokens, masked_scatter,
    get_rope_index (lightweight, <1ms).

    Falls back to PyTorch LLM if llm_bf16.engine is not found.
    """
    backbone = policy.model.backbone
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration
    action_head = policy.model.action_head

    # --- Backbone setup ---
    # Save references needed by the TRT forward
    backbone._embedding_layer = qwen_model.model.language_model.get_input_embeddings()
    backbone._image_token_id = qwen_model.config.image_token_id

    # Load ViT TRT engine
    vit_engine_path = os.path.join(trt_engine_path, "vit_bf16.engine")
    if os.path.exists(vit_engine_path):
        print(f"Loading ViT engine: {vit_engine_path}")
        backbone.vit_engine = Engine(vit_engine_path)

        del qwen_model.model.visual
        torch.cuda.empty_cache()
        print("  Deleted PyTorch ViT (replaced by TRT engine)")
    else:
        backbone.vit_engine = None

    # Load LLM TRT engine (if available)
    llm_engine_path = os.path.join(trt_engine_path, "llm_bf16.engine")
    use_llm_trt = os.path.exists(llm_engine_path)

    if use_llm_trt and backbone.vit_engine is None:
        # LLM TRT requires ViT TRT — the TRT backbone forward functions
        # assume ViT output is produced by the ViT engine. Without it,
        # loading the LLM engine would delete PyTorch LLM layers while
        # leaving the backbone forward un-patched, causing AttributeError.
        raise RuntimeError(
            f"LLM TRT engine found at {llm_engine_path} but ViT TRT engine is "
            f"missing at {vit_engine_path}.\n"
            f"The n17_full_pipeline mode requires both ViT and LLM engines.\n\n"
            f"To fix this, either:\n"
            f"  1. Rebuild the ViT engine (recommended for full pipeline performance):\n"
            f"       python scripts/deployment/export_onnx_n1d7.py \\\n"
            f"         --model_path <MODEL> --dataset_path <DATA> \\\n"
            f"         --output_dir ./gr00t_n1d7_onnx --export_mode full_pipeline\n"
            f"       python scripts/deployment/build_tensorrt_engine.py \\\n"
            f"         --mode full_pipeline --onnx_dir ./gr00t_n1d7_onnx \\\n"
            f"         --engine_dir {trt_engine_path} --precision bf16\n\n"
            f"  2. Use action_head mode instead (backbone stays in PyTorch):\n"
            f"       setup_tensorrt_engines(policy, '{trt_engine_path}', mode='action_head')"
        )

    if use_llm_trt:
        print(f"Loading LLM engine: {llm_engine_path}")
        backbone.llm_engine = Engine(llm_engine_path)

        # Delete PyTorch LLM layers to free GPU memory
        # Keep embed_tokens (needed for token embedding before TRT)
        # Keep get_rope_index via inner_model (needed for position IDs)
        del qwen_model.model.language_model.layers
        del qwen_model.model.language_model.norm
        torch.cuda.empty_cache()
        print("  Deleted PyTorch LLM layers (replaced by TRT engine)")
    else:
        backbone.llm_engine = None
        print(f"  LLM engine not found at {llm_engine_path}, using PyTorch LLM")

    # Monkey-patch backbone forward
    if backbone.vit_engine is not None:
        if use_llm_trt:
            backbone.forward = partial(qwen3_backbone_full_trt_forward, backbone)
        else:
            backbone.forward = partial(qwen3_backbone_tensorrt_forward, backbone)
    else:
        print(f"  ViT engine not found at {vit_engine_path}, backbone remains in PyTorch")

    # --- Action head setup ---
    if hasattr(action_head, "model"):
        del action_head.model
    if hasattr(action_head, "state_encoder"):
        del action_head.state_encoder
    if hasattr(action_head, "action_encoder"):
        del action_head.action_encoder
    if hasattr(action_head, "action_decoder"):
        del action_head.action_decoder
    torch.cuda.empty_cache()

    assert action_head.action_dim == action_head.config.max_action_dim

    print(f"Loading action head engines from: {trt_engine_path}")
    action_head.state_encoder_engine = Engine(os.path.join(trt_engine_path, "state_encoder.engine"))
    action_head.action_encoder_engine = Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    action_head.dit_engine = Engine(os.path.join(trt_engine_path, "dit_bf16.engine"))
    action_head.action_decoder_engine = Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )

    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    llm_status = "TRT" if use_llm_trt else "PyTorch"
    vit_status = "TRT" if backbone.vit_engine else "PyTorch"
    print("N1.7 full-pipeline TRT engines loaded.")
    print(f"  ViT: {vit_status} | LLM: {llm_status} | Action Head: TRT")


def _setup_action_head(policy, trt_engine_path):
    """Set up TRT engines for action head only (N1.7 mode).

    Backbone (Qwen3-VL) stays in PyTorch. Only the 4 action head components
    (State Encoder, Action Encoder, DiT, Action Decoder) are replaced with
    TRT engines.
    """
    action_head = policy.model.action_head

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

    # Verify action_dim consistency
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

    # Monkey-patch: backbone.forward stays original, only action head is replaced
    action_head.get_action = partial(action_head_tensorrt_forward, action_head)

    print("Action head TRT engines loaded and forward method patched.")
    print("  Backbone remains in PyTorch (Qwen3-VL).")


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
