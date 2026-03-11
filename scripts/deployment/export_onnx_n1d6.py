#!/usr/bin/env python3
"""
Export GR00T N1.6 model components to ONNX for TensorRT optimization.

Supports two export modes:
- dit_only: Export only the DiT model (backward compatible)
- full_pipeline: Export LLM + State Encoder + Action Encoder + DiT + Action Decoder

Usage:
    # DiT only (default):
    python scripts/deployment/export_onnx_n1d6.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path demo_data/gr1.PickNPlace \
        --output_dir ./groot_n1d6_onnx

    # Full pipeline:
    python scripts/deployment/export_onnx_n1d6.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path demo_data/gr1.PickNPlace \
        --output_dir ./groot_n1d6_onnx \
        --export_mode full_pipeline
"""

import argparse
import copy
import json
import logging
import os
import sys
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


# Ensure scripts/deployment/ is on sys.path for sibling module imports
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from accuracy_thresholds import VIT_WRAPPER_COSINE_MIN  # noqa: E402  # isort: skip


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Input Capture Helpers
# ============================================================


class DiTInputCapture:
    """Capture DiT forward pass inputs during inference."""

    def __init__(self):
        self.captured = False
        self.sa_embs = None
        self.vl_embs = None
        self.timestep = None
        self.image_mask = None
        self.backbone_attention_mask = None

    def hook_fn(self, module, args, kwargs):
        """Pre-forward hook to capture inputs."""
        if not self.captured:
            self.sa_embs = kwargs["hidden_states"].detach().cpu().clone()
            self.vl_embs = kwargs["encoder_hidden_states"].detach().cpu().clone()
            self.timestep = kwargs["timestep"].detach().cpu().clone()
            i_mask = kwargs.get("image_mask")
            if i_mask is not None:
                self.image_mask = i_mask.detach().cpu().clone()
            bb_mask = kwargs.get("backbone_attention_mask")
            if bb_mask is not None:
                self.backbone_attention_mask = bb_mask.detach().cpu().clone()

            self.captured = True
            logger.info("  Captured DiT inputs:")
            logger.info(f"    sa_embs shape: {self.sa_embs.shape}")
            logger.info(f"    vl_embs shape: {self.vl_embs.shape}")
            logger.info(f"    timestep shape: {self.timestep.shape}")
            if self.image_mask is not None:
                logger.info(f"    image_mask shape: {self.image_mask.shape}")
            if self.backbone_attention_mask is not None:
                logger.info(
                    f"    backbone_attention_mask shape: {self.backbone_attention_mask.shape}"
                )


# ============================================================
# Observation Helpers
# ============================================================


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def prepare_observation(policy, dataset, traj_idx=0):
    """Prepare a single observation for inference."""
    logger.info(f"\nPreparing observation from trajectory {traj_idx}...")

    traj = dataset[traj_idx]
    modality_configs = policy.get_modality_config()

    data_point = extract_step_data(
        traj,
        0,
        modality_configs=modality_configs,
        embodiment_tag=policy.embodiment_tag,
    )

    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in modality_configs["language"].modality_keys:
        observation[key] = data_point.text

    parsed_obs = parse_observation_gr00t(observation, modality_configs)
    logger.info("  Observation prepared")
    return parsed_obs


# ============================================================
# ViT (Siglip2) ONNX-Exportable Wrappers
# ============================================================


class Siglip2AttentionOpt(nn.Module):
    """ONNX-exportable attention for Siglip2 (no RoPE, no windowed attention).

    Uses F.scaled_dot_product_attention (SDPA) for normal inference — SDPA
    selects the most efficient kernel (flash, memory-efficient, or math) and
    produces numerically closer results to the flash_attention_2 used by the
    original model, reducing wrapper-vs-original drift.

    During ONNX export, falls back to manual matmul+softmax since the legacy
    ONNX exporter cannot trace SDPA (ComplexDouble type error).
    """

    def __init__(self, original_attn):
        super().__init__()
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = original_attn.scale

        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.out_proj = original_attn.out_proj

    def forward(self, hidden_states):
        B, N, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if torch.onnx.is_in_onnx_export():
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class Siglip2EncoderLayerOpt(nn.Module):
    """ONNX-exportable encoder layer for Siglip2."""

    def __init__(self, original_layer):
        super().__init__()
        self.layer_norm1 = original_layer.layer_norm1
        self.self_attn = Siglip2AttentionOpt(original_layer.self_attn)
        self.layer_norm2 = original_layer.layer_norm2
        self.mlp = original_layer.mlp

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2EncoderOpt(nn.Module):
    """ONNX-exportable encoder for Siglip2 (no RoPE, no win_meta_list)."""

    def __init__(self, original_encoder):
        super().__init__()
        self.layers = nn.ModuleList(
            [Siglip2EncoderLayerOpt(layer) for layer in original_encoder.layers]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Siglip2VisionTransformerOpt(nn.Module):
    """ONNX-exportable wrapper for Siglip2 ViT (fixed resolution).

    Pre-computes all metadata-dependent operations (window reordering,
    positional embeddings, reverse mapping) as static buffers. This avoids
    Python-level logic (lists, dicts) that ONNX cannot export.

    The image_size is determined from actual inference (e.g. 252 for GR1)
    since Eagle's smart_resize() may adjust the resolution at runtime.

    Requires: use_rope=False and use_windows_attn=False in the model config.
    """

    def __init__(self, original_vit, image_size):
        super().__init__()
        config = original_vit.config
        patch_size = config.patch_size
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        num_patches_per_side = image_size // patch_size

        # Validate assumptions
        assert not config.use_rope, "ViT ONNX export requires use_rope=False"
        assert not config.use_windows_attn, "ViT ONNX export requires use_windows_attn=False"

        # Copy core submodules (weights are shared, not cloned)
        self.patch_embedding = original_vit.embeddings.patch_embedding
        self.encoder = Siglip2EncoderOpt(original_vit.encoder)
        self.post_layernorm = original_vit.post_layernorm

        # Pre-compute static positional embeddings for the target resolution
        embeddings_module = original_vit.embeddings
        pos_emb_size = embeddings_module.position_embedding_size  # sqrt(num_patches)
        pos_weight = embeddings_module.position_embedding.weight.reshape(
            pos_emb_size, pos_emb_size, -1
        )

        spatial_shapes = torch.tensor([[num_patches_per_side, num_patches_per_side]])
        with torch.no_grad():
            # resize_positional_embeddings: (pos_size, pos_size, C) -> (1, H*W, C)
            # Access the static method from the embeddings module's class
            static_pos = embeddings_module.resize_positional_embeddings(pos_weight, spatial_shapes)
        self.register_buffer("static_position_embeddings", static_pos)

        # Pre-compute window reordering and reverse mapping
        # Run original embeddings once with a dummy image to capture the indices
        device = next(original_vit.parameters()).device
        dtype = next(original_vit.parameters()).dtype
        dummy_images = [torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)]

        with torch.no_grad():
            windows_tensor, win_meta_list, _, reverse_mapping = embeddings_module(dummy_images)

        # Build forward mapping: original patch order → windowed patch order.
        #
        # reverse_mapping[orig_idx] = windowed_idx  (from embeddings code)
        # Original ViT uses: output[:, orig_idx] = hidden[:, reverse_mapping[orig_idx]]
        #   = hidden[:, windowed_idx]  →  restores original order from windowed order.
        #
        # We need forward_mapping such that:
        #   output[:, windowed_idx] = input[:, forward_mapping[windowed_idx]]
        #   = input[:, orig_idx]  →  reorders original to windowed order.
        #
        # So forward_mapping[windowed_idx] = orig_idx (the inverse of reverse_mapping).
        total_patches = num_patches_per_side * num_patches_per_side
        forward_mapping = torch.zeros(total_patches, dtype=torch.long, device=device)
        for orig_idx in range(total_patches):
            windowed_idx = reverse_mapping[orig_idx].item()
            forward_mapping[windowed_idx] = orig_idx

        self.register_buffer("forward_mapping", forward_mapping)
        self.register_buffer("reverse_mapping", reverse_mapping)

        # Save constants
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = num_patches_per_side
        self.total_patches = total_patches

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, H, H) single tensor, NOT a list.
                          H is the image_size passed to __init__ (e.g. 252).

        Returns:
            vit_embeds: (B, num_patches, 1152)
        """
        B = pixel_values.shape[0]

        # 1. Patch embedding: (B, 3, H, H) -> (B*N, ps*ps*3) -> (B*N, 1152)
        # convert_images_to_patches equivalent using reshape + permute
        ps = self.patch_size
        nph = self.num_patches_per_side
        npw = self.num_patches_per_side
        # (B, 3, nph, ps, npw, ps)
        x = pixel_values.reshape(B, 3, nph, ps, npw, ps)
        # (B, nph, npw, ps, ps, 3) -> (B*nph*npw, ps*ps*3)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nph * npw, -1)

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(x.to(dtype=target_dtype))
        patch_embeds = patch_embeds.reshape(B, self.total_patches, -1)

        # 2. Add pre-computed position embeddings: (1, N, 1152)
        embeddings = patch_embeds + self.static_position_embeddings

        # 3. Apply window reordering (static index_select)
        embeddings = torch.index_select(embeddings, 1, self.forward_mapping)

        # 4. Encoder (27 layers, eager attention, no RoPE)
        hidden_states = self.encoder(embeddings)

        # 5. Post LayerNorm
        hidden_states = self.post_layernorm(hidden_states)

        # 6. Reverse mapping (restore original patch order)
        hidden_states = torch.index_select(hidden_states, 1, self.reverse_mapping)

        return hidden_states


# ============================================================
# Export Functions: ViT (Siglip2)
# ============================================================


def export_vit_to_onnx(policy, output_dir, use_bf16=True, captured_image_size=None):
    """Export Siglip2 ViT to ONNX.

    Args:
        policy: Loaded Gr00tPolicy
        output_dir: Output directory for ONNX file
        use_bf16: Whether to export in BF16 precision
        captured_image_size: Actual image size from inference (e.g. 252).
                             If None, runs one inference to detect it.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting ViT (Siglip2) to ONNX")
    logger.info("=" * 80)

    eagle_model = policy.model.backbone.model
    original_vit = eagle_model.vision_model.vision_model

    logger.info(
        f"  ViT config: hidden_size={original_vit.config.hidden_size}, "
        f"num_layers={original_vit.config.num_hidden_layers}, "
        f"use_rope={original_vit.config.use_rope}, "
        f"use_windows_attn={original_vit.config.use_windows_attn}"
    )

    # Determine actual image size if not provided
    if captured_image_size is None:
        logger.warning(
            "  No captured_image_size provided, defaulting to 224. "
            "Pass captured_image_size for correct resolution."
        )
        captured_image_size = 224

    image_size = captured_image_size
    patch_size = original_vit.config.patch_size
    nps = image_size // patch_size
    logger.info(
        f"  Image size: {image_size}x{image_size} "
        f"({nps}x{nps} = {nps * nps} patches, patch_size={patch_size})"
    )

    # Create optimized wrapper
    opt_model = Siglip2VisionTransformerOpt(original_vit, image_size=image_size).eval().cuda()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    opt_model = opt_model.to(dtype)

    # Verify numerical equivalence before export
    logger.info("  Verifying wrapper vs original...")
    dummy_input = torch.randn(1, 3, image_size, image_size, device="cuda", dtype=dtype)
    with torch.inference_mode():
        # Original: expects a list of tensors
        orig_out = original_vit([dummy_input])
        orig_embeds = orig_out.last_hidden_state

        # Wrapper: expects a single tensor
        opt_embeds = opt_model(dummy_input)

    cos_sim = F.cosine_similarity(
        orig_embeds.float().flatten().unsqueeze(0),
        opt_embeds.float().flatten().unsqueeze(0),
    ).item()
    l1_diff = (orig_embeds.float() - opt_embeds.float()).abs().mean().item()
    logger.info(f"  Wrapper vs Original: cosine_sim={cos_sim:.6f}, L1_mean={l1_diff:.8f}")

    if cos_sim < VIT_WRAPPER_COSINE_MIN:
        logger.warning(
            f"  WARNING: Wrapper output diverges from original! "
            f"cosine_sim < {VIT_WRAPPER_COSINE_MIN}"
        )

    # Export to ONNX
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=dtype, device="cuda")
    output_path = os.path.join(output_dir, "vit_bf16.onnx")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"  pixel_values: {pixel_values.shape} ({pixel_values.dtype})")
    logger.info(f"  Exporting to {output_path}...")

    with torch.inference_mode():
        torch.onnx.export(
            opt_model,
            (pixel_values,),
            output_path,
            input_names=["pixel_values"],
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )

    logger.info("  ViT exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# ONNX Verification Helper
# ============================================================


def verify_onnx_export(output_path):
    """Verify an ONNX export and log file size."""
    import onnx

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  Model size on disk: {file_size_mb:.2f} MB")

    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        external_size_mb = os.path.getsize(external_data_path) / (1024 * 1024)
        logger.info(f"  External data size: {external_size_mb:.2f} MB")
        logger.info(f"  Total model size: {file_size_mb + external_size_mb:.2f} MB")

    try:
        onnx.checker.check_model(output_path)
        logger.info("  ONNX model is valid!")
    except ValueError as e:
        if "too large" in str(e):
            logger.info("  Model is very large, skipping full validation...")
            try:
                tmp_path = output_path + ".tmp"
                onnx.shape_inference.infer_shapes_path(output_path, tmp_path)
                os.remove(tmp_path)
                logger.info("  ONNX model structure verified!")
            except Exception as e2:
                logger.warning(f"  Could not fully validate (this is OK): {e2}")
        else:
            raise


# ============================================================
# Export Functions: LLM
# ============================================================


def export_llm_to_onnx(policy, seq_len, output_dir, use_bf16=True):
    """Export the truncated Qwen3 LLM to ONNX with eager attention.

    Args:
        policy: Loaded Gr00tPolicy
        seq_len: Sequence length from captured inference (for dummy inputs)
        output_dir: Output directory for ONNX file
        use_bf16: Whether to export in BF16 precision
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting LLM to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    eagle_model = backbone.model  # Eagle3_VLForConditionalGeneration
    language_model = eagle_model.language_model  # Qwen3ForCausalLM (truncated)
    select_layer = backbone.select_layer

    # Create a Qwen3Model with eager attention (no flash)
    llm_config = copy.deepcopy(language_model.config)
    llm_config._attn_implementation = "eager"
    llm_config.num_hidden_layers = select_layer

    logger.info(
        f"  LLM config: hidden_size={llm_config.hidden_size}, "
        f"num_layers={llm_config.num_hidden_layers}, "
        f"attn_implementation={llm_config._attn_implementation}"
    )

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    class LLMForExport(torch.nn.Module):
        """Wrapper that returns last hidden state instead of logits.

        Replaces the internal _update_causal_mask to avoid COMPLEX128 type casts
        that TensorRT's ONNX parser cannot handle.
        """

        def __init__(self, config):
            super().__init__()
            self.model = Qwen3Model(config)
            # Monkey-patch the causal mask to avoid complex type operations
            self.model._update_causal_mask = self._simple_causal_mask

        def _simple_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position=None,
            past_key_values=None,
            output_attentions=False,
        ):
            """ONNX-compatible causal mask without complex type casts."""
            dtype = input_tensor.dtype
            device = input_tensor.device
            batch_size, seq_len = input_tensor.shape[:2]

            # Upper-triangular causal mask (future tokens masked with large negative).
            # Use dtype-aware min value (not hardcoded FP16 min) so this works
            # correctly for BF16 and FP32 as well.
            mask_value = torch.finfo(dtype).min * 0.5
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

            # Apply padding mask from attention_mask [B, seq_len]
            if attention_mask is not None and attention_mask.dim() == 2:
                padding_mask = attention_mask[:, None, None, :].to(dtype)
                padding_mask = (1.0 - padding_mask) * mask_value
                causal_mask = causal_mask + padding_mask

            return causal_mask

        def forward(self, inputs_embeds, attention_mask):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            return outputs.last_hidden_state

    wrapper = LLMForExport(llm_config)

    # Load weights from the existing truncated Qwen3Model
    original_qwen3_model = language_model.model  # Qwen3Model inside Qwen3ForCausalLM
    wrapper.model.load_state_dict(original_qwen3_model.state_dict())

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    wrapper = wrapper.to(dtype).eval().cuda()

    # Create dummy inputs
    hidden_size = llm_config.hidden_size
    inputs_embeds = torch.randn(1, seq_len, hidden_size, dtype=dtype, device="cuda")
    attention_mask = torch.ones(1, seq_len, dtype=torch.int64, device="cuda")

    logger.info(f"  inputs_embeds: {inputs_embeds.shape} ({inputs_embeds.dtype})")
    logger.info(f"  attention_mask: {attention_mask.shape} ({attention_mask.dtype})")

    output_path = os.path.join(output_dir, "llm_bf16.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs_embeds, attention_mask),
            output_path,
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

    # Consolidate external data into a single file for TRT compatibility.
    # torch.onnx.export scatters weights as many small files; TRT parser
    # needs them consolidated next to the .onnx file.
    logger.info("  Consolidating external data...")
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model = onnx.load(output_path)
    external_data_file = os.path.basename(output_path) + ".data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=external_data_file,
        size_threshold=0,
    )
    onnx.save(model, output_path)
    logger.info(f"  External data consolidated to {external_data_file}")

    # Clean up old scattered external data files
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if (
            os.path.isfile(fpath)
            and f.startswith("onnx__")
            and f not in (os.path.basename(output_path), external_data_file)
        ):
            os.remove(fpath)

    logger.info("  LLM exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: State Encoder
# ============================================================


def export_state_encoder_to_onnx(policy, output_dir, use_bf16=True):
    """Export the state encoder (CategorySpecificMLP) to ONNX.

    Input: state [B, 1, max_state_dim], embodiment_id [B]
    Output: [B, 1, input_embedding_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting State Encoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    state_encoder = policy.model.action_head.state_encoder

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = state_encoder.to(dtype).eval().cuda()

    state = torch.randn(1, 1, config.max_state_dim, dtype=dtype, device="cuda")
    embodiment_id = torch.zeros(1, dtype=torch.int64, device="cuda")

    logger.info(f"  state: {state.shape} ({state.dtype})")
    logger.info(f"  embodiment_id: {embodiment_id.shape} ({embodiment_id.dtype})")

    output_path = os.path.join(output_dir, "state_encoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (state, embodiment_id),
            output_path,
            input_names=["state", "embodiment_id"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "state": {0: "batch_size"},
                "embodiment_id": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    logger.info("  State Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: Action Encoder
# ============================================================


def export_action_encoder_to_onnx(policy, output_dir, use_bf16=True):
    """Export the action encoder (MultiEmbodimentActionEncoder) to ONNX.

    Input: actions [B, action_horizon, max_action_dim], timesteps [B], embodiment_id [B]
    Output: [B, action_horizon, input_embedding_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Action Encoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    action_encoder = policy.model.action_head.action_encoder

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = action_encoder.to(dtype).eval().cuda()

    actions = torch.randn(
        1, config.action_horizon, config.max_action_dim, dtype=dtype, device="cuda"
    )
    timesteps = torch.zeros(1, dtype=torch.int64, device="cuda")
    embodiment_id = torch.zeros(1, dtype=torch.int64, device="cuda")

    logger.info(f"  actions: {actions.shape} ({actions.dtype})")
    logger.info(f"  timesteps: {timesteps.shape} ({timesteps.dtype})")
    logger.info(f"  embodiment_id: {embodiment_id.shape} ({embodiment_id.dtype})")

    output_path = os.path.join(output_dir, "action_encoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (actions, timesteps, embodiment_id),
            output_path,
            input_names=["actions", "timesteps", "embodiment_id"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "actions": {0: "batch_size"},
                "timesteps": {0: "batch_size"},
                "embodiment_id": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    logger.info("  Action Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: DiT
# ============================================================


def export_dit_to_onnx(
    policy: Gr00tPolicy, captured_inputs: DiTInputCapture, output_path: str, use_bf16: bool = True
):
    """Export the DiT (AlternateVLDiT) to ONNX.

    Input: sa_embs [B, sa_seq_len, input_embedding_dim],
           vl_embs [B, vl_seq_len, backbone_embedding_dim],
           timestep [B], image_mask [B, vl_seq_len],
           backbone_attention_mask [B, vl_seq_len]
    Output: [B, sa_seq_len, hidden_size]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting DiT to ONNX")
    logger.info("=" * 80)

    dit_model = policy.model.action_head.model
    dit_model.eval()

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    dit_model = dit_model.to(dtype).cuda()

    sa_embs = torch.randn(captured_inputs.sa_embs.shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(captured_inputs.vl_embs.shape, dtype=dtype, device="cuda")
    timestep = torch.ones(captured_inputs.timestep.shape, dtype=torch.int64, device="cuda")

    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]
    dynamic_axes = {
        "sa_embs": {0: "batch_size", 1: "sa_seq_len"},
        "vl_embs": {0: "batch_size", 1: "vl_seq_len"},
        "timestep": {0: "batch_size"},
        "output": {0: "batch_size", 1: "sa_seq_len"},
    }

    if captured_inputs.image_mask is not None:
        image_mask = torch.ones(captured_inputs.image_mask.shape, dtype=torch.bool, device="cuda")
        export_inputs.append(image_mask)
        input_names.append("image_mask")
        dynamic_axes["image_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    if captured_inputs.backbone_attention_mask is not None:
        backbone_attention_mask = torch.ones(
            captured_inputs.backbone_attention_mask.shape, dtype=torch.bool, device="cuda"
        )
        export_inputs.append(backbone_attention_mask)
        input_names.append("backbone_attention_mask")
        dynamic_axes["backbone_attention_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    logger.info("  Export input shapes:")
    for name, tensor in zip(input_names, export_inputs):
        logger.info(f"    {name}: {tensor.shape} ({tensor.dtype})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Wrapper to convert positional args → keyword args for DiT
    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit_model, has_image_mask, has_backbone_mask):
            super().__init__()
            self.dit_model = dit_model
            self.has_image_mask = has_image_mask
            self.has_backbone_mask = has_backbone_mask

        def forward(
            self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None
        ):
            kwargs = {}
            if self.has_image_mask and image_mask is not None:
                kwargs["image_mask"] = image_mask
            if self.has_backbone_mask and backbone_attention_mask is not None:
                kwargs["backbone_attention_mask"] = backbone_attention_mask
            return self.dit_model(sa_embs, vl_embs, timestep, **kwargs)

    wrapped_model = DiTWrapper(
        dit_model,
        has_image_mask=captured_inputs.image_mask is not None,
        has_backbone_mask=captured_inputs.backbone_attention_mask is not None,
    )
    wrapped_model.eval()

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapped_model,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    logger.info("  DiT exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: Action Decoder
# ============================================================


def export_action_decoder_to_onnx(policy, output_dir, use_bf16=True):
    """Export the action decoder (CategorySpecificMLP) to ONNX.

    Input: model_output [B, sa_seq_len, hidden_size], embodiment_id [B]
    Output: [B, sa_seq_len, max_action_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Action Decoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    action_decoder = policy.model.action_head.action_decoder

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = action_decoder.to(dtype).eval().cuda()

    # sa_seq_len = 1 (state) + action_horizon
    sa_seq_len = 1 + config.action_horizon
    model_output = torch.randn(1, sa_seq_len, config.hidden_size, dtype=dtype, device="cuda")
    embodiment_id = torch.zeros(1, dtype=torch.int64, device="cuda")

    logger.info(f"  model_output: {model_output.shape} ({model_output.dtype})")
    logger.info(f"  embodiment_id: {embodiment_id.shape} ({embodiment_id.dtype})")

    output_path = os.path.join(output_dir, "action_decoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (model_output, embodiment_id),
            output_path,
            input_names=["model_output", "embodiment_id"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "model_output": {0: "batch_size"},
                "embodiment_id": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    logger.info("  Action Decoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# FP8 Quantization: Calibration Datasets & Quantize Functions
# ============================================================


def _load_calibration_observations(policy, dataset_path, video_backend, calib_size):
    """Load a list of observations for calibration.

    Returns observations WITHOUT batch dimension, matching the format expected
    by VLAStepData (same as what _to_vla_step_data receives after unbatching).

    Returns:
        list of observation dicts with structure:
            {"video": {key: ndarray(T, H, W, C)},
             "state": {key: ndarray(T, D)},
             "language": {key: [str]}}
    """
    modality_configs = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend=video_backend,
    )

    observations = []
    for traj_idx in range(min(len(dataset), calib_size)):
        try:
            traj = dataset[traj_idx]
            data_point = extract_step_data(
                traj,
                0,
                modality_configs=modality_configs,
                embodiment_tag=policy.embodiment_tag,
            )
            obs = {}
            for key, value in data_point.states.items():
                obs[f"state.{key}"] = value
            for key, value in data_point.images.items():
                obs[f"video.{key}"] = np.array(value)
            for key in modality_configs["language"].modality_keys:
                obs[key] = data_point.text
            # parse_observation_gr00t adds a batch dim (arr[None,:]) needed for
            # batched inference, but calibration datasets process single samples.
            # Unbatch immediately so the data matches what VLAStepData expects
            # (same format as _to_vla_step_data after _unbatch_observation).
            parsed = parse_observation_gr00t(obs, modality_configs)
            unbatched = {}
            for modality in parsed:
                unbatched[modality] = {}
                for key, val in parsed[modality].items():
                    if isinstance(val, list):
                        unbatched[modality][key] = val[0]  # [[str]] -> [str]
                    else:
                        unbatched[modality][key] = val[0]  # (1,...) -> (...)
            observations.append(unbatched)
        except Exception as e:
            logger.warning(f"  Skipping traj {traj_idx}: {e}")
            continue
        if len(observations) >= calib_size:
            break

    logger.info(f"  Loaded {len(observations)} calibration observations")
    return observations


class ViTCalibrationDataset(torch.utils.data.Dataset):
    """Provides pixel_values tensors for ViT FP8 calibration."""

    def __init__(self, policy, observations):
        self.pixel_values_list = []
        for obs in observations:
            try:
                with torch.inference_mode():
                    from gr00t.data.types import MessageType, VLAStepData

                    vla_step_data = VLAStepData(
                        images=obs["video"],
                        states=obs["state"],
                        actions={},
                        text=obs["language"][policy.language_key][0],
                        embodiment=policy.embodiment_tag,
                    )
                    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
                    processed = policy.processor(messages)
                    collated = policy.collate_fn([processed])
                    pv = collated["inputs"]["pixel_values"]
                    # pv is a list of tensors; for single-view, take first element
                    if isinstance(pv, (list, tuple)):
                        pv = torch.cat(pv, dim=0)
                    self.pixel_values_list.append(pv.to(torch.bfloat16).cuda())
            except Exception as e:
                logger.warning(f"  Skipping calib sample: {e}")

    def __len__(self):
        return len(self.pixel_values_list)

    def __getitem__(self, idx):
        return self.pixel_values_list[idx]


class LLMCalibrationDataset(torch.utils.data.Dataset):
    """Provides (inputs_embeds, attention_mask) for LLM FP8 calibration.

    Pre-computes ViT + pixel_shuffle + MLP1 + embedding scatter for each observation.
    """

    def __init__(self, policy, observations):
        from gr00t.data.types import MessageType, VLAStepData

        self.inputs_embeds_list = []
        self.attention_mask_list = []

        backbone = policy.model.backbone
        eagle_model = backbone.model

        for obs in observations:
            try:
                vla_step_data = VLAStepData(
                    images=obs["video"],
                    states=obs["state"],
                    actions={},
                    text=obs["language"][policy.language_key][0],
                    embodiment=policy.embodiment_tag,
                )
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
                processed = policy.processor(messages)
                collated = policy.collate_fn([processed])
                inputs = collated["inputs"]
                inputs = {
                    k: v.to(torch.bfloat16).cuda()
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                    else (v.cuda() if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                }

                with torch.inference_mode():
                    backbone_inputs, _ = policy.model.prepare_input(inputs)
                    pixel_values = backbone_inputs["pixel_values"]
                    vit_embeds = eagle_model.extract_feature(pixel_values)

                    input_ids = backbone_inputs["input_ids"]
                    input_embeds = backbone.embedding_layer(input_ids)
                    input_embeds = input_embeds.to(torch.bfloat16)
                    vit_embeds = vit_embeds.to(torch.bfloat16)

                    B, N, C = input_embeds.shape
                    input_embeds = input_embeds.reshape(B * N, C)
                    input_ids_flat = input_ids.reshape(B * N)
                    selected = input_ids_flat == eagle_model.image_token_index
                    input_embeds[selected] = vit_embeds.reshape(-1, C)[: selected.sum()]
                    input_embeds = input_embeds.reshape(B, N, C)

                    attention_mask = backbone_inputs["attention_mask"]
                    if attention_mask.dtype != torch.int64:
                        attention_mask = attention_mask.to(torch.int64)

                self.inputs_embeds_list.append(input_embeds.cpu())
                self.attention_mask_list.append(attention_mask.cpu())
            except Exception as e:
                logger.warning(f"  Skipping LLM calib sample: {e}")

    def __len__(self):
        return len(self.inputs_embeds_list)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.inputs_embeds_list[idx].cuda(),
            "attention_mask": self.attention_mask_list[idx].cuda(),
        }


def quantize_and_export_vit_fp8(policy, observations, output_dir, image_size=252):
    """FP8 quantize ViT and export to ONNX.

    Args:
        policy: Loaded Gr00tPolicy
        observations: List of calibration observations
        output_dir: Output directory for ONNX
        image_size: Actual image resolution from inference capture
    """
    import modelopt.torch.quantization as mtq

    logger.info("\n" + "=" * 80)
    logger.info("FP8 Quantizing ViT (Siglip2)")
    logger.info("=" * 80)

    eagle_model = policy.model.backbone.model
    original_vit = eagle_model.vision_model.vision_model

    opt_model = (
        Siglip2VisionTransformerOpt(original_vit, image_size=image_size)
        .eval()
        .cuda()
        .to(torch.bfloat16)
    )

    # Create calibration dataset
    calib_ds = ViTCalibrationDataset(policy, observations)
    logger.info(f"  Calibration samples: {len(calib_ds)}")

    # FP8 config: disable Conv2d quantization to preserve accuracy
    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    def calibrate_loop(model):
        with torch.inference_mode():
            for i in range(min(len(calib_ds), 64)):
                model(calib_ds[i])

    mtq.quantize(opt_model, quant_cfg, forward_loop=calibrate_loop)

    # Export quantized ONNX
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.bfloat16, device="cuda")
    output_path = os.path.join(output_dir, "vit_bf16.onnx")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"  Exporting FP8-quantized ViT to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            opt_model,
            (pixel_values,),
            output_path,
            input_names=["pixel_values"],
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )

    logger.info("  FP8-quantized ViT exported!")
    verify_onnx_export(output_path)
    return output_path


def quantize_and_export_llm_fp8(policy, observations, seq_len, output_dir):
    """FP8 quantize LLM and export to ONNX.

    Args:
        policy: Loaded Gr00tPolicy
        observations: List of calibration observations
        seq_len: LLM sequence length
        output_dir: Output directory for ONNX
    """
    import modelopt.torch.quantization as mtq

    logger.info("\n" + "=" * 80)
    logger.info("FP8 Quantizing LLM (Qwen3)")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    language_model = backbone.model.language_model
    select_layer = backbone.select_layer

    # Save embedding layer for calibration dataset
    if not hasattr(backbone, "embedding_layer"):
        backbone.embedding_layer = language_model.get_input_embeddings()
    if not hasattr(backbone, "image_token_index"):
        backbone.image_token_index = backbone.model.image_token_index

    llm_config = copy.deepcopy(language_model.config)
    llm_config._attn_implementation = "eager"
    llm_config.num_hidden_layers = select_layer

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    # Create LLM wrapper (same as BF16 export)
    class LLMForExport(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = Qwen3Model(config)
            self.model._update_causal_mask = self._simple_causal_mask

        def _simple_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position=None,
            past_key_values=None,
            output_attentions=False,
        ):
            dtype = input_tensor.dtype
            device = input_tensor.device
            batch_size, seq_len = input_tensor.shape[:2]
            mask_value = torch.finfo(dtype).min * 0.5
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            if attention_mask is not None and attention_mask.dim() == 2:
                padding_mask = attention_mask[:, None, None, :].to(dtype)
                padding_mask = (1.0 - padding_mask) * mask_value
                causal_mask = causal_mask + padding_mask
            return causal_mask

        def forward(self, inputs_embeds, attention_mask):
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return outputs.last_hidden_state

    wrapper = LLMForExport(llm_config)
    original_qwen3_model = language_model.model
    wrapper.model.load_state_dict(original_qwen3_model.state_dict())
    wrapper = wrapper.to(torch.bfloat16).eval().cuda()

    # Create calibration dataset
    calib_ds = LLMCalibrationDataset(policy, observations)
    logger.info(f"  LLM calibration samples: {len(calib_ds)}")

    def calibrate_loop(model):
        with torch.inference_mode():
            for i in range(min(len(calib_ds), 64)):
                sample = calib_ds[i]
                model(sample["inputs_embeds"], sample["attention_mask"])

    mtq.quantize(wrapper, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate_loop)

    # Export quantized ONNX
    hidden_size = llm_config.hidden_size
    inputs_embeds = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")
    attention_mask = torch.ones(1, seq_len, dtype=torch.int64, device="cuda")
    output_path = os.path.join(output_dir, "llm_bf16.onnx")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"  Exporting FP8-quantized LLM to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs_embeds, attention_mask),
            output_path,
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

    # Consolidate external data
    logger.info("  Consolidating external data...")
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model_onnx = onnx.load(output_path)
    external_data_file = os.path.basename(output_path) + ".data"
    convert_model_to_external_data(
        model_onnx,
        all_tensors_to_one_file=True,
        location=external_data_file,
        size_threshold=0,
    )
    onnx.save(model_onnx, output_path)

    # Clean up scattered files
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if (
            os.path.isfile(fpath)
            and f.startswith("onnx__")
            and f not in (os.path.basename(output_path), external_data_file)
        ):
            os.remove(fpath)

    logger.info("  FP8-quantized LLM exported!")
    verify_onnx_export(output_path)
    return output_path


def quantize_and_export_dit_fp8(policy, captured_inputs, observations, output_dir):
    """FP8 quantize DiT and export to ONNX.

    Uses a multi-step denoising calibration loop to properly calibrate
    the DiT's dynamic range across all denoising timesteps.

    Args:
        policy: Loaded Gr00tPolicy
        captured_inputs: DiTInputCapture with captured tensor shapes
        observations: List of calibration observations
        output_dir: Output directory for ONNX
    """
    import modelopt.torch.quantization as mtq

    logger.info("\n" + "=" * 80)
    logger.info("FP8 Quantizing DiT")
    logger.info("=" * 80)

    dit_model = policy.model.action_head.model
    dit_model.eval().to(torch.bfloat16).cuda()

    # FP8 config with attention quantizer tuning
    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quant_cfg["quant_cfg"]["*[qkv]_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None}
    quant_cfg["quant_cfg"]["*softmax_quantizer"] = {"num_bits": (4, 3), "axis": None}

    def dit_calibrate_loop(model):
        """Run full denoising loop for DiT calibration.

        policy.get_action() expects batched observations (B, T, ...).
        Calibration observations are unbatched, so re-add batch dim here.
        """
        with torch.inference_mode():
            for obs in observations[:32]:
                try:
                    batched_obs = {
                        "video": {k: v[None, ...] for k, v in obs["video"].items()},
                        "state": {k: v[None, ...] for k, v in obs["state"].items()},
                        "language": {k: [v] for k, v in obs["language"].items()},
                    }
                    action, _ = policy.get_action(batched_obs)
                except Exception:
                    continue

    mtq.quantize(dit_model, quant_cfg, forward_loop=dit_calibrate_loop)

    # Export (same structure as BF16 export)
    dtype = torch.bfloat16
    sa_embs = torch.randn(captured_inputs.sa_embs.shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(captured_inputs.vl_embs.shape, dtype=dtype, device="cuda")
    timestep = torch.ones(captured_inputs.timestep.shape, dtype=torch.int64, device="cuda")

    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]
    dynamic_axes = {
        "sa_embs": {0: "batch_size", 1: "sa_seq_len"},
        "vl_embs": {0: "batch_size", 1: "vl_seq_len"},
        "timestep": {0: "batch_size"},
        "output": {0: "batch_size", 1: "sa_seq_len"},
    }

    if captured_inputs.image_mask is not None:
        image_mask = torch.ones(captured_inputs.image_mask.shape, dtype=torch.bool, device="cuda")
        export_inputs.append(image_mask)
        input_names.append("image_mask")
        dynamic_axes["image_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    if captured_inputs.backbone_attention_mask is not None:
        bb_mask = torch.ones(
            captured_inputs.backbone_attention_mask.shape, dtype=torch.bool, device="cuda"
        )
        export_inputs.append(bb_mask)
        input_names.append("backbone_attention_mask")
        dynamic_axes["backbone_attention_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit_model, has_image_mask, has_backbone_mask):
            super().__init__()
            self.dit_model = dit_model
            self.has_image_mask = has_image_mask
            self.has_backbone_mask = has_backbone_mask

        def forward(
            self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None
        ):
            kwargs = {}
            if self.has_image_mask and image_mask is not None:
                kwargs["image_mask"] = image_mask
            if self.has_backbone_mask and backbone_attention_mask is not None:
                kwargs["backbone_attention_mask"] = backbone_attention_mask
            return self.dit_model(sa_embs, vl_embs, timestep, **kwargs)

    wrapped = DiTWrapper(
        dit_model,
        has_image_mask=captured_inputs.image_mask is not None,
        has_backbone_mask=captured_inputs.backbone_attention_mask is not None,
    ).eval()

    output_path = os.path.join(output_dir, "dit_bf16.onnx")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"  Exporting FP8-quantized DiT to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    logger.info("  FP8-quantized DiT exported!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Main
# ============================================================


def main(args):
    logger.info("=" * 80)
    logger.info("GR00T N1.6 ONNX Export Script")
    logger.info("=" * 80)
    logger.info(f"Model path:    {args.model_path}")
    logger.info(f"Dataset path:  {args.dataset_path}")
    logger.info(f"Embodiment:    {args.embodiment_tag}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Export mode:   {args.export_mode}")
    logger.info(f"Precision:     {args.precision}")
    logger.info("=" * 80)

    use_fp8 = args.precision == "fp8"
    if use_fp8:
        try:
            import modelopt.torch.quantization as mtq  # noqa: F401

            logger.info("  nvidia-modelopt loaded successfully")
        except ImportError:
            logger.error("  FP8 requires nvidia-modelopt: uv pip install 'nvidia-modelopt[torch]'")
            return

    # Step 1: Load the policy
    logger.info("\n[Step 1] Loading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    logger.info("  Policy loaded")

    # Step 2: Load dataset
    logger.info("\n[Step 2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    logger.info(f"  Dataset loaded ({len(dataset)} trajectories)")

    # Step 3: Capture component inputs from actual inference
    logger.info("\n[Step 3] Capturing component inputs from actual inference...")

    dit_capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(
        dit_capture.hook_fn, with_kwargs=True
    )

    # Also capture pixel_values shape to determine actual image resolution
    # (Eagle's smart_resize may change 224→252 depending on input images)
    captured_pixel_values_shape = [None]

    def capture_pixel_values(module, args, kwargs):
        if captured_pixel_values_shape[0] is None:
            # pixel_values may be positional (args[0]) or keyword
            pv = args[0] if len(args) > 0 else kwargs.get("pixel_values")
            if pv is not None:
                if isinstance(pv, (list, tuple)) and len(pv) > 0:
                    captured_pixel_values_shape[0] = pv[0].shape
                elif isinstance(pv, torch.Tensor):
                    captured_pixel_values_shape[0] = pv.shape

    eagle_model = policy.model.backbone.model
    vit_hook = eagle_model.vision_model.vision_model.register_forward_pre_hook(
        capture_pixel_values, with_kwargs=True
    )

    observation = prepare_observation(policy, dataset, traj_idx=0)
    logger.info("  Running inference to capture shapes...")
    with torch.inference_mode():
        _ = policy.get_action(observation)

    hook.remove()
    vit_hook.remove()

    if not dit_capture.captured:
        logger.error("  Failed to capture DiT inputs!")
        return

    # Derive LLM seq_len from captured DiT inputs
    # vl_embs shape is [B, vl_seq_len, backbone_embedding_dim]
    # vl_seq_len == LLM output seq_len
    llm_seq_len = dit_capture.vl_embs.shape[1]
    logger.info(f"  Derived LLM sequence length: {llm_seq_len}")

    # Derive actual image size from captured pixel_values
    # pixel_values shape: (B, C, H, W) — H should equal W
    if captured_pixel_values_shape[0] is not None:
        captured_image_size = captured_pixel_values_shape[0][-1]  # W dimension
        logger.info(f"  Captured image size: {captured_image_size}x{captured_image_size}")
    else:
        captured_image_size = 224
        logger.warning("  Could not capture pixel_values shape, defaulting to 224")

    # Save export metadata for downstream tools (builder, inference)
    action_head_config = policy.model.action_head.config
    sa_seq_len = 1 + action_head_config.action_horizon  # 1 state + action_horizon
    vit_config = policy.model.backbone.model.vision_model.vision_model.config
    export_metadata = {
        "image_size": int(captured_image_size),
        "patch_size": int(vit_config.patch_size),
        "llm_seq_len": int(llm_seq_len),
        "sa_seq_len": int(sa_seq_len),
        "vl_seq_len": int(llm_seq_len),
        "action_horizon": int(action_head_config.action_horizon),
        "max_action_dim": int(action_head_config.max_action_dim),
        "hidden_size": int(action_head_config.hidden_size),
        "embodiment_tag": str(args.embodiment_tag),
        "export_mode": args.export_mode,
        "precision": args.precision,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "export_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(export_metadata, f, indent=2)
    logger.info(f"  Saved export metadata to {metadata_path}")

    # Step 4: Export components

    if args.export_mode == "dit_only":
        # Backward-compatible: export only DiT
        logger.info("\n[Step 4] Exporting DiT to ONNX (dit_only mode)...")
        dit_output_path = os.path.join(args.output_dir, "dit_model.onnx")
        export_dit_to_onnx(
            policy=policy,
            captured_inputs=dit_capture,
            output_path=dit_output_path,
            use_bf16=True,
        )

    elif args.export_mode == "full_pipeline":
        logger.info("\n[Step 4] Exporting full pipeline to ONNX...")

        if use_fp8:
            # FP8 mode: quantize + export components that benefit from FP8
            calib_path = args.calib_dataset_path or args.dataset_path
            logger.info(f"\n[Step 4 FP8] Loading calibration data from {calib_path}...")
            calib_observations = _load_calibration_observations(
                policy, calib_path, args.video_backend, args.calib_size
            )

            # 4a. FP8 ViT
            logger.info("\n--- [4a] ViT (Siglip2) FP8 ---")
            quantize_and_export_vit_fp8(
                policy,
                calib_observations,
                args.output_dir,
                image_size=captured_image_size,
            )

            # 4b. FP8 LLM
            logger.info("\n--- [4b] LLM FP8 ---")
            quantize_and_export_llm_fp8(policy, calib_observations, llm_seq_len, args.output_dir)

            # 4c. State Encoder (small model, BF16 is fine)
            logger.info("\n--- [4c] State Encoder (BF16) ---")
            export_state_encoder_to_onnx(policy, args.output_dir, use_bf16=True)

            # 4d. Action Encoder (small model, BF16 is fine)
            logger.info("\n--- [4d] Action Encoder (BF16) ---")
            export_action_encoder_to_onnx(policy, args.output_dir, use_bf16=True)

            # 4e. FP8 DiT
            logger.info("\n--- [4e] DiT FP8 ---")
            quantize_and_export_dit_fp8(policy, dit_capture, calib_observations, args.output_dir)

            # 4f. Action Decoder (small model, BF16 is fine)
            logger.info("\n--- [4f] Action Decoder (BF16) ---")
            export_action_decoder_to_onnx(policy, args.output_dir, use_bf16=True)

        else:
            # BF16 mode: standard export
            # 4a. Export ViT (Siglip2)
            logger.info("\n--- [4a] ViT (Siglip2) ---")
            export_vit_to_onnx(
                policy,
                args.output_dir,
                use_bf16=True,
                captured_image_size=captured_image_size,
            )

            # 4b. Export LLM
            logger.info("\n--- [4b] LLM ---")
            export_llm_to_onnx(policy, llm_seq_len, args.output_dir, use_bf16=True)

            # 4c. Export State Encoder
            logger.info("\n--- [4c] State Encoder ---")
            export_state_encoder_to_onnx(policy, args.output_dir, use_bf16=True)

            # 4d. Export Action Encoder
            logger.info("\n--- [4d] Action Encoder ---")
            export_action_encoder_to_onnx(policy, args.output_dir, use_bf16=True)

            # 4e. Export DiT
            logger.info("\n--- [4e] DiT ---")
            dit_output_path = os.path.join(args.output_dir, "dit_bf16.onnx")
            export_dit_to_onnx(
                policy=policy,
                captured_inputs=dit_capture,
                output_path=dit_output_path,
                use_bf16=True,
            )

            # 4f. Export Action Decoder
            logger.info("\n--- [4f] Action Decoder ---")
            export_action_decoder_to_onnx(policy, args.output_dir, use_bf16=True)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nExported files in: {args.output_dir}")

    # List exported files
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info(f"  {f}: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GR00T N1.6 model to ONNX")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset (used to capture input shapes)",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=EmbodimentTag,
        default=EmbodimentTag.GR1,
        help="Embodiment tag (default: GR1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./groot_n1d6_onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--export_mode",
        type=str,
        default="dit_only",
        choices=["dit_only", "full_pipeline"],
        help="Export mode: 'dit_only' (default) or 'full_pipeline'",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Options: ['decord', 'torchvision_av', 'torchcodec']",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp8"],
        help="Export precision: 'bf16' (default) or 'fp8' (requires nvidia-modelopt)",
    )
    parser.add_argument(
        "--calib_dataset_path",
        type=str,
        default=None,
        help="Path to calibration dataset for FP8 quantization. "
        "If not provided, uses --dataset_path.",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=100,
        help="Number of calibration samples for FP8 quantization (default: 100)",
    )

    args = parser.parse_args()
    main(args)
