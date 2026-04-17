#!/usr/bin/env python3

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

"""
Export GR00T N1.7 model components to ONNX for TensorRT optimization.

Supports three export modes:
  - dit_only:      Export only the DiT (backward compatible with N1.6).
  - action_head:   Export 4 action head components (ViT + LLM stay in PyTorch).
  - full_pipeline: Export ViT + LLM + 4 action head components. Lightweight
                   glue ops (embed_tokens, masked_scatter, get_rope_index,
                   VLLN) remain in PyTorch. Referred to as 'n17_full_pipeline'
                   by the engine-loading code (trt_model_forward.py,
                   build_trt_pipeline.py) and as 'trt_full_pipeline' by the
                   standalone_inference_script.py --inference-mode flag;
                   both names describe the same engine set.

Usage:
    # Download finetuned model first (HF doesn't support nested repo paths)
    uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO

    # DiT only (default)
    python export_onnx_n1d7.py \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo \\
        --output-dir ./gr00t_n1d7_onnx

    # Action head (4 components)
    python export_onnx_n1d7.py \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo \\
        --output-dir ./gr00t_n1d7_onnx \\
        --export-mode action_head
"""

import copy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import tyro


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _is_spark_sm121() -> bool:
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)


def _should_use_dynamo_exporter() -> bool:
    override = os.environ.get("GR00T_ONNX_EXPORTER_MODE")
    if override == "legacy":
        return False
    if override == "default":
        return True

    return not _is_spark_sm121()


def _consolidate_external_data(onnx_path: str) -> None:
    """Merge scattered external-data files into a single .data file next to the ONNX model."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    onnx_dir = os.path.dirname(onnx_path)
    onnx_name = os.path.basename(onnx_path)
    data_file = onnx_name + ".data"

    # Check if there are scattered files (files without .onnx/.json extension)
    scattered = [
        f
        for f in os.listdir(onnx_dir)
        if os.path.isfile(os.path.join(onnx_dir, f))
        and not f.endswith((".onnx", ".json", ".data"))
        and f != data_file
    ]
    if not scattered:
        return

    logger.info(f"  Consolidating {len(scattered)} external data files into {data_file}...")
    model = onnx.load(onnx_path, load_external_data=True)
    convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location=data_file, size_threshold=0
    )
    onnx.save(model, onnx_path)

    # Clean up scattered files
    for f in scattered:
        os.remove(os.path.join(onnx_dir, f))
    logger.info(f"  Consolidated and cleaned up {len(scattered)} files.")


def verify_onnx_export(onnx_path: str) -> None:
    """Load and check the exported ONNX model for validity."""
    import onnx

    logger.info(f"  Verifying {onnx_path} ...")
    onnx.checker.check_model(onnx_path)
    logger.info("  ONNX model verified successfully.")


def verify_onnx_with_ort(
    onnx_path: str,
    pytorch_module: torch.nn.Module,
    sample_inputs: dict[str, torch.Tensor],
    output_names: list[str],
    label: str = "model",
) -> dict[str, float]:
    """Run ONNX Runtime inference and compare against PyTorch output.

    Returns dict of {output_name: cosine_similarity}.
    Requires onnxruntime-gpu; skips gracefully if not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("  onnxruntime not installed — skipping ORT verification")
        return {}

    logger.info(f"  ORT verification for {label}...")

    # Run PyTorch
    with torch.inference_mode():
        pt_inputs = tuple(sample_inputs[name] for name in sample_inputs)
        pt_outputs = pytorch_module(*pt_inputs)
        if not isinstance(pt_outputs, (tuple, list)):
            pt_outputs = (pt_outputs,)

    # Run ONNX Runtime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {name: t.cpu().numpy() for name, t in sample_inputs.items()}
    ort_outputs = sess.run(output_names, ort_inputs)

    # Compare
    results = {}
    for i, name in enumerate(output_names):
        pt_flat = pt_outputs[i].float().flatten().cpu()
        ort_flat = torch.tensor(ort_outputs[i]).float().flatten()
        cosine = torch.nn.functional.cosine_similarity(
            pt_flat.unsqueeze(0), ort_flat.unsqueeze(0)
        ).item()
        results[name] = cosine
        logger.info(f"    {name}: ORT vs PyTorch cosine = {cosine:.6f}")

    return results


# ============================================================
# Input Capture
# ============================================================


class DiTInputCapture:
    """Capture DiT forward pass inputs during inference via a pre-forward hook."""

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
            logger.info(f"    sa_embs: {self.sa_embs.shape}")
            logger.info(f"    vl_embs: {self.vl_embs.shape}")
            logger.info(f"    timestep: {self.timestep.shape}")
            if self.image_mask is not None:
                logger.info(f"    image_mask: {self.image_mask.shape}")
            if self.backbone_attention_mask is not None:
                logger.info(f"    backbone_attention_mask: {self.backbone_attention_mask.shape}")


class ViTInputCapture:
    """Capture ViT (VisionModel) forward inputs/outputs during inference."""

    def __init__(self):
        self.captured = False
        self.pixel_values_shape = None
        self.grid_thw = None
        self.output_shape = None
        self.deepstack_shapes = []

    def hook_fn(self, module, args, kwargs, output):
        if not self.captured:
            self.pixel_values_shape = args[0].shape
            grid = args[1] if len(args) > 1 else kwargs.get("grid_thw")
            self.grid_thw = grid.detach().cpu().clone()
            if isinstance(output, tuple):
                self.output_shape = output[0].shape
                self.deepstack_shapes = [f.shape for f in output[1]]
            else:
                self.output_shape = output.shape
            self.captured = True
            logger.info("  Captured ViT inputs:")
            logger.info(f"    pixel_values: {self.pixel_values_shape}")
            logger.info(f"    grid_thw: {self.grid_thw.tolist()}")
            logger.info(f"    output: {self.output_shape}")
            logger.info(f"    deepstack: {len(self.deepstack_shapes)} features")


class LLMInputCapture:
    """Capture LLM (Qwen3VLTextModel) inputs during inference via a pre-forward hook.

    Captures inputs_embeds, position_ids, attention_mask, visual_pos_masks,
    and deepstack_visual_embeds — everything needed to reproduce the LLM forward.
    """

    def __init__(self):
        self.captured = False
        self.inputs_embeds = None
        self.position_ids = None
        self.attention_mask = None
        self.visual_pos_masks = None
        self.deepstack_visual_embeds = None  # list of tensors

    def hook_fn(self, module, args, kwargs):
        if not self.captured:
            ie = kwargs.get("inputs_embeds")
            if ie is None and len(args) > 0:
                ie = args[0]
            if ie is not None:
                self.inputs_embeds = ie.detach().cpu().clone()

            pid = kwargs.get("position_ids")
            if pid is not None:
                self.position_ids = pid.detach().cpu().clone()

            am = kwargs.get("attention_mask")
            if am is not None:
                self.attention_mask = am.detach().cpu().clone()

            vpm = kwargs.get("visual_pos_masks")
            if vpm is not None:
                self.visual_pos_masks = vpm.detach().cpu().clone()

            dve = kwargs.get("deepstack_visual_embeds")
            if dve is not None:
                self.deepstack_visual_embeds = [d.detach().cpu().clone() for d in dve]

            self.captured = True
            logger.info("  Captured LLM inputs:")
            if self.inputs_embeds is not None:
                logger.info(f"    inputs_embeds: {self.inputs_embeds.shape}")
            if self.position_ids is not None:
                logger.info(f"    position_ids: {self.position_ids.shape}")
            if self.attention_mask is not None:
                logger.info(f"    attention_mask: {self.attention_mask.shape}")
            if self.visual_pos_masks is not None:
                logger.info(f"    visual_pos_masks: {self.visual_pos_masks.shape}")
            if self.deepstack_visual_embeds is not None:
                logger.info(
                    f"    deepstack: {len(self.deepstack_visual_embeds)} tensors, "
                    f"shapes: {[d.shape for d in self.deepstack_visual_embeds]}"
                )


# ============================================================
# ViT Export: ONNX-friendly attention + wrapper
# ============================================================


def _apply_rotary_real(x, cos, sin):
    """Apply rotary position embeddings using only real-valued ops (no complex).

    Uses float32 internally to match transformers' apply_rotary_pos_emb_vision
    precision, then casts back to the original dtype.

    Args:
        x: [seq, heads, head_dim]
        cos, sin: [seq, head_dim]
    Returns:
        [seq, heads, head_dim]
    """
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)  # [seq, 1, head_dim]
    sin = sin.float().unsqueeze(1)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos + rotated * sin).to(orig_dtype)


def _make_onnx_vision_attention_forward(attn_module, chunk_sizes=None):
    """Create an ONNX-exportable attention forward for a single VisionAttention.

    Three key changes from the original:
    1. Replaces cu_seqlens-based splitting with static chunk splitting
       (each image's patches attend only within their own chunk)
    2. Replaces apply_rotary_pos_emb_vision (uses complex numbers) with
       real-valued rotate_half implementation
    3. Casts to float32 before softmax for TRT accuracy

    Args:
        attn_module: The VisionAttention module to wrap
        chunk_sizes: List of ints, number of patches per image.
            e.g. [256, 256] for 2 images of 256 patches each.
            If None or single chunk, does full-sequence attention.
    """

    def forward(
        hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs
    ):
        seq_length = hidden_states.shape[0]
        qkv = attn_module.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, attn_module.num_heads, -1)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        # q, k, v: [seq_length, num_heads, head_dim]

        cos, sin = position_embeddings
        q = _apply_rotary_real(q, cos, sin)
        k = _apply_rotary_real(k, cos, sin)

        # Split by image chunks (mirrors cu_seqlens-based splitting in original)
        # Each image's patches attend only within their own chunk.
        if chunk_sizes is not None and len(chunk_sizes) > 1:
            q_chunks = torch.split(q, chunk_sizes, dim=0)
            k_chunks = torch.split(k, chunk_sizes, dim=0)
            v_chunks = torch.split(v, chunk_sizes, dim=0)

            attn_outputs = []
            for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
                # q_c, k_c, v_c: [chunk_seq, num_heads, head_dim]
                q_c = q_c.transpose(0, 1)  # [num_heads, chunk_seq, head_dim]
                k_c = k_c.transpose(0, 1)
                v_c = v_c.transpose(0, 1)

                w = torch.matmul(q_c, k_c.transpose(-2, -1)) * attn_module.scaling
                w = w.to(torch.float32)
                w = F.softmax(w, dim=-1)
                w = w.to(v_c.dtype)
                out = torch.matmul(w, v_c)  # [num_heads, chunk_seq, head_dim]
                attn_outputs.append(out.transpose(0, 1))  # [chunk_seq, num_heads, head_dim]

            attn_output = torch.cat(attn_outputs, dim=0)  # [seq, num_heads, head_dim]
        else:
            # Single image: full-sequence attention
            q = q.transpose(0, 1)  # [num_heads, seq, head_dim]
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * attn_module.scaling
            attn_weights = attn_weights.to(torch.float32)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(v.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)  # [seq, num_heads, head_dim]

        # [seq, num_heads, head_dim] → [seq, num_heads * head_dim]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = attn_module.proj(attn_output)
        return attn_output

    return forward


def _patch_vision_attention_for_export(vision_model, chunk_sizes=None):
    """Monkey-patch all VisionAttention.forward with ONNX-friendly versions.

    Args:
        vision_model: The vision model whose attention blocks to patch
        chunk_sizes: List of ints, patches per image (from grid_thw).
            Enables per-image attention splitting. If None, full-sequence attention.

    Returns list of original forwards for restoration after export.
    """
    originals = []
    for block in vision_model.blocks:
        attn = block.attn
        originals.append(attn.forward)
        attn.forward = _make_onnx_vision_attention_forward(attn, chunk_sizes=chunk_sizes)
    info = f"  Patched {len(originals)} vision attention blocks for ONNX export"
    if chunk_sizes and len(chunk_sizes) > 1:
        info += f" (chunk_sizes={chunk_sizes})"
    logger.info(info)
    return originals


def _restore_vision_attention(vision_model, originals):
    """Restore original attention forwards after export."""
    for block, orig in zip(vision_model.blocks, originals):
        block.attn.forward = orig


class Qwen3VisionForExport(torch.nn.Module):
    """ONNX-exportable wrapper for Qwen3-VL Vision Model.

    Pre-computes position embeddings and rotary embeddings for a fixed grid_thw
    to avoid ComplexDouble operations that ONNX cannot handle. Replaces the
    dynamic VisionModel.forward with a traceable version.

    Architecture: patch_embed → add pos_embed → blocks(attn+ffn) → deepstack → merger
    """

    def __init__(self, vision_model, grid_thw: torch.Tensor):
        super().__init__()
        self.patch_embed = vision_model.patch_embed
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list

        # Pre-compute position embeddings (avoids grid_thw-dependent Python loops
        # and ComplexDouble operations in rotary embedding computation)
        with torch.no_grad():
            pos_embeds = vision_model.fast_pos_embed_interpolate(grid_thw)
            rotary = vision_model.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)

        self.register_buffer("_pos_embeds", pos_embeds.clone().detach().contiguous())
        self.register_buffer("_rot_cos", emb.cos().clone().detach().contiguous())
        self.register_buffer("_rot_sin", emb.sin().clone().detach().contiguous())

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self._pos_embeds

        position_embeddings = (self._rot_cos, self._rot_sin)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=None,  # not used by patched attention
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))

        image_embeds = self.merger(hidden_states)

        if deepstack_features:
            deepstack = torch.stack(deepstack_features)  # [num_layers, N, D]
        else:
            deepstack = image_embeds.new_zeros(1, 1, 1)

        return image_embeds, deepstack


# ============================================================
# Export Functions: ViT
# ============================================================


def export_vit_to_onnx(policy, output_dir, captured_vit, use_bf16=True, batch_size=1):
    """Export Qwen3-VL Vision Model to ONNX.

    Pre-computes position/rotary embeddings for the captured grid_thw to avoid
    ComplexDouble ops. Monkey-patches attention to use standard SDPA (valid for
    single-image inference where all patches attend to all patches).

    Input: pixel_values [num_patches * batch_size, C*T*pH*pW]
    Output: image_embeds [num_merged_patches * batch_size, hidden_dim],
            deepstack_features [num_layers, num_merged_patches * batch_size, hidden_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting ViT (Qwen3-VL Vision) to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    qwen_model = backbone.model
    vision = qwen_model.model.visual

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    vision = vision.to(dtype).eval().cuda()

    # Compute chunk sizes from grid_thw for per-image attention splitting
    # grid_thw: [num_images, 3] where each row is (temporal, height, width)
    # cu_seqlens derived as: patches_per_image = h * w, repeated t times
    grid_thw = captured_vit.grid_thw.to(device="cuda")
    # For batch_size > 1, repeat grid_thw to tile position embeddings for all batch elements
    if batch_size > 1:
        grid_thw = grid_thw.repeat(batch_size, 1)
    chunk_sizes = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
    logger.info(f"  grid_thw: {grid_thw.tolist()}, chunk_sizes: {chunk_sizes}")

    # Patch attention for ONNX export with per-image chunk splitting
    originals = _patch_vision_attention_for_export(vision, chunk_sizes=chunk_sizes)

    # Build wrapper with pre-computed position embeddings
    wrapper = Qwen3VisionForExport(vision, grid_thw)
    wrapper = wrapper.to(dtype).eval().cuda()

    # Input: only pixel_values (grid_thw is baked into pre-computed buffers)
    # For batch_size > 1, scale num_patches by batch_size
    pv_shape = captured_vit.pixel_values_shape
    if batch_size > 1:
        pv_shape = (pv_shape[0] * batch_size, pv_shape[1])
    pixel_values = torch.randn(pv_shape, dtype=dtype, device="cuda")

    logger.info(f"  pixel_values: {pixel_values.shape} ({pixel_values.dtype})")

    precision_tag = "bf16" if use_bf16 else "fp32"
    output_path = os.path.join(output_dir, f"vit_{precision_tag}.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_names = ["image_embeds", "deepstack_features"]

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            output_path,
            input_names=["pixel_values"],
            output_names=output_names,
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
        )

    logger.info("  ViT exported successfully!")
    _consolidate_external_data(output_path)
    verify_onnx_export(output_path)

    # ORT verification: compare ONNX output against PyTorch wrapper
    # Must run BEFORE restoring attention, since wrapper uses patched attention
    verify_onnx_with_ort(
        onnx_path=output_path,
        pytorch_module=wrapper,
        sample_inputs={"pixel_values": pixel_values},
        output_names=output_names,
        label="ViT",
    )

    # Restore original attention
    _restore_vision_attention(vision, originals)

    return output_path


# ============================================================
# LLM Export: Qwen3-VL Text Model with Deepstack
# ============================================================


def export_llm_to_onnx(policy, captured_llm, output_dir, use_bf16=True, batch_size=1):
    """Export the Qwen3-VL text model (LLM) to ONNX.

    The LLM receives inputs_embeds (with vision tokens already scattered in),
    pre-computed 3D position_ids, and deepstack visual embeddings. Position ID
    computation (get_rope_index) stays in PyTorch at runtime — only the
    transformer layers are exported to ONNX/TRT.

    Deepstack injection is handled inside the wrapper: visual features are added
    to hidden states at the first N layers (N = number of deepstack features,
    typically 3) at positions indicated by visual_pos_masks.

    Input:
        inputs_embeds:  [B, seq_len, hidden_size]
        attention_mask: [B, seq_len]  (int64, 1=attend, 0=pad)
        position_ids:   [3, B, seq_len]  (temporal, height, width for 3D RoPE)
        visual_pos_masks: [B, seq_len]  (bool, True at visual token positions)
        deepstack_0:    [num_vis_tokens, hidden_size]  (deepstack feature for layer 0)
        deepstack_1:    [num_vis_tokens, hidden_size]  (deepstack feature for layer 1)
        deepstack_2:    [num_vis_tokens, hidden_size]  (deepstack feature for layer 2)
    Output:
        embeddings: [B, seq_len, hidden_size]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting LLM (Qwen3-VL Text Model) to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    qwen_model = backbone.model  # Qwen3VLForConditionalGeneration
    inner_model = qwen_model.model  # Qwen3VLModel
    text_model = inner_model.language_model  # Qwen3VLTextModel
    select_layer = backbone.select_layer

    # Get text config and create eager-attention copy
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    text_config = copy.deepcopy(text_model.config)
    text_config._attn_implementation = "eager"
    text_config.num_hidden_layers = select_layer

    logger.info(
        f"  LLM config: hidden_size={text_config.hidden_size}, "
        f"num_layers={text_config.num_hidden_layers}, "
        f"attn_implementation={text_config._attn_implementation}"
    )

    # Determine deepstack count
    num_deepstack = 0
    if captured_llm.deepstack_visual_embeds is not None:
        num_deepstack = len(captured_llm.deepstack_visual_embeds)
    logger.info(f"  Deepstack layers: {num_deepstack}")

    class LLMForExport(torch.nn.Module):
        """ONNX-exportable wrapper for Qwen3-VL text model with deepstack.

        Key adaptations from the original Qwen3VLTextModel:
        1. Eager attention (no flash) — avoids ONNX-incompatible flash attention
        2. Simple causal mask — avoids COMPLEX128 ops from HuggingFace's mask
        3. Deepstack injection via torch.where — avoids boolean indexing
        4. Position IDs as explicit input — get_rope_index() stays in PyTorch
        5. Deepstack features as separate tensor inputs (not a Python list)
        """

        def __init__(self, config, n_deepstack):
            super().__init__()
            # Build fresh text model with eager attention
            self.layers = nn.ModuleList(
                [
                    # Import the decoder layer class
                    __import__(
                        "transformers.models.qwen3_vl.modeling_qwen3_vl",
                        fromlist=["Qwen3VLTextDecoderLayer"],
                    ).Qwen3VLTextDecoderLayer(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            # NOTE: No final norm! Qwen3Backbone.forward returns
            # hidden_states[-1] (pre-norm), not last_hidden_state (post-norm).
            # The action head's VLLN handles normalization downstream.
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
            self.n_deepstack = n_deepstack

        def _simple_causal_mask(self, dtype, device, batch_size, seq_len, attention_mask):
            """ONNX-compatible causal mask without complex type casts."""
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

        def _deepstack_add(self, hidden_states, visual_pos_masks, visual_embeds):
            """ONNX-friendly deepstack injection using torch.where.

            Original uses boolean indexing:
                hidden_states[visual_pos_masks, :] += visual_embeds
            This is not ONNX-friendly. Instead we:
            1. Build a full-size delta tensor (zeros except at visual positions)
            2. Add it to hidden_states
            """
            # visual_pos_masks: [B, seq_len] bool
            # visual_embeds: [num_vis_tokens, hidden_size]
            # hidden_states: [B, seq_len, hidden_size]
            B, S, H = hidden_states.shape

            # Scatter visual_embeds into a full [B, S, H] tensor at masked positions
            mask_expanded = visual_pos_masks.unsqueeze(-1)  # [B, S, 1]

            # Build cumulative index for visual tokens per batch
            # For single batch (B=1), this is straightforward
            delta = torch.zeros_like(hidden_states)
            # Use masked_scatter to place visual_embeds at the right positions
            delta = delta.masked_scatter(mask_expanded.expand_as(delta), visual_embeds)

            hidden_states = hidden_states + delta
            return hidden_states

        def forward(
            self,
            inputs_embeds,
            attention_mask,
            position_ids,
            visual_pos_masks=None,
            deepstack_0=None,
            deepstack_1=None,
            deepstack_2=None,
        ):
            batch_size, seq_len = inputs_embeds.shape[:2]
            dtype = inputs_embeds.dtype
            device = inputs_embeds.device

            # Build causal attention mask
            attn_mask = self._simple_causal_mask(dtype, device, batch_size, seq_len, attention_mask)

            # Position IDs: [3, B, seq_len] → extract text_position_ids
            text_position_ids = position_ids[0]  # [B, seq_len]

            # Cache position for rotary embeddings
            cache_position = torch.arange(seq_len, device=device)

            hidden_states = inputs_embeds

            # Compute rotary position embeddings (shared across layers)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # Collect deepstack features into indexable structure
            deepstack_list = []
            if deepstack_0 is not None:
                deepstack_list.append(deepstack_0)
            if deepstack_1 is not None:
                deepstack_list.append(deepstack_1)
            if deepstack_2 is not None:
                deepstack_list.append(deepstack_2)

            # Decoder layers
            for layer_idx, decoder_layer in enumerate(self.layers):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs

                # Deepstack injection at first N layers
                if visual_pos_masks is not None and layer_idx < len(deepstack_list):
                    hidden_states = self._deepstack_add(
                        hidden_states, visual_pos_masks, deepstack_list[layer_idx]
                    )

            # Return pre-norm hidden states (matching Qwen3Backbone.forward
            # which uses hidden_states[-1], not last_hidden_state)
            return hidden_states

    # Build wrapper and load weights
    wrapper = LLMForExport(text_config, num_deepstack)

    # Copy weights from the existing truncated text model
    # The text model has: embed_tokens, layers, norm, rotary_emb
    # We only need layers, norm, rotary_emb (embed_tokens not used — we pass inputs_embeds)
    src_state = text_model.state_dict()
    dst_state = wrapper.state_dict()
    loaded, skipped = 0, 0
    for key in dst_state:
        if key in src_state:
            dst_state[key] = src_state[key]
            loaded += 1
        else:
            skipped += 1
            logger.warning(f"  Key not found in source: {key}")
    wrapper.load_state_dict(dst_state)
    logger.info(f"  Loaded {loaded} weight tensors, skipped {skipped}")

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    wrapper = wrapper.to(dtype).eval().cuda()

    # Create dummy inputs from captured shapes
    seq_len = captured_llm.inputs_embeds.shape[1]
    hidden_size = text_config.hidden_size

    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device="cuda")
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.int64, device="cuda")

    export_inputs = [inputs_embeds, attention_mask, position_ids]
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    # seq_len varies with tokenized input length — must be dynamic for TRT profile
    llm_dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "embeddings": {1: "seq_len"},
    }

    if num_deepstack > 0 and captured_llm.visual_pos_masks is not None:
        # Use actual captured mask so masked_scatter sizes match deepstack
        vis_mask = captured_llm.visual_pos_masks.to(device="cuda")
        if vis_mask.shape[0] == 1 and batch_size > 1:
            vis_mask = vis_mask.expand(batch_size, -1)
        export_inputs.append(vis_mask)
        input_names.append("visual_pos_masks")
        llm_dynamic_axes["visual_pos_masks"] = {1: "seq_len"}

        for i in range(num_deepstack):
            ds = captured_llm.deepstack_visual_embeds[i]
            # deepstack is [num_vis_tokens, hidden_size] — no batch dim.
            # masked_scatter fills all B*num_vis_tokens positions, so repeat for batch_size > 1.
            if batch_size > 1:
                ds_dummy = torch.randn(
                    batch_size * ds.shape[0], ds.shape[1], dtype=dtype, device="cuda"
                )
            else:
                ds_dummy = torch.randn_like(ds, dtype=dtype, device="cuda")
            export_inputs.append(ds_dummy)
            name = f"deepstack_{i}"
            input_names.append(name)

    logger.info("  Export input shapes:")
    for name, tensor in zip(input_names, export_inputs):
        logger.info(f"    {name}: {tensor.shape} ({tensor.dtype})")

    precision_tag = "bf16" if use_bf16 else "fp32"
    output_path = os.path.join(output_dir, f"llm_{precision_tag}.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=llm_dynamic_axes,
            export_params=True,
        )

    logger.info("  LLM exported successfully!")
    _consolidate_external_data(output_path)
    verify_onnx_export(output_path)
    return output_path


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
        traj, 0, modality_configs=modality_configs, embodiment_tag=policy.embodiment_tag
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
# Export Functions: VL Self-Attention
# ============================================================


def export_vl_self_attention_to_onnx(policy, output_dir, vl_seq_len, use_bf16=True, batch_size=1):
    """Export the vl_self_attention (SelfAttentionTransformer) to ONNX.

    This module sits between VLLN and the DiT, transforming backbone
    embeddings. If the model has no vl_self_attention (nn.Identity), skip.

    Input: hidden_states [B, T, backbone_embedding_dim]  (T is dynamic)
    Output: hidden_states [B, T, backbone_embedding_dim]
    """
    vl_sa = policy.model.action_head.vl_self_attention
    if isinstance(vl_sa, nn.Identity):
        logger.info("  vl_self_attention is Identity — skipping export")
        return None

    logger.info("\n" + "=" * 80)
    logger.info("Exporting VL Self-Attention to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = vl_sa.to(dtype).eval().cuda()

    hidden_states = torch.randn(
        batch_size, vl_seq_len, config.backbone_embedding_dim, dtype=dtype, device="cuda"
    )
    logger.info(f"  hidden_states: {hidden_states.shape} ({hidden_states.dtype})")

    output_path = os.path.join(output_dir, "vl_self_attention.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  Exporting to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (hidden_states,),
            output_path,
            input_names=["hidden_states"],
            output_names=["output"],
            dynamic_axes={
                "hidden_states": {1: "seq_len"},
                "output": {1: "seq_len"},
            },
            opset_version=19,
            do_constant_folding=True,
        )

    logger.info("  VL Self-Attention exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: State Encoder
# ============================================================


def export_state_encoder_to_onnx(policy, output_dir, use_bf16=True, batch_size=1):
    """Export the state encoder (CategorySpecificMLP) to ONNX.

    N1.7 change: input_dim = max_state_dim * state_history_length.
    The state is reshaped from [B, state_history_length, max_state_dim]
    to [B, 1, state_history_length * max_state_dim] before encoding.

    Input: state [B, 1, max_state_dim * state_history_length], embodiment_id [B]
    Output: [B, 1, input_embedding_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting State Encoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    state_encoder = policy.model.action_head.state_encoder

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = state_encoder.to(dtype).eval().cuda()

    # N1.7: state is flattened to [B, 1, max_state_dim * state_history_length]
    state_input_dim = config.max_state_dim * config.state_history_length
    state = torch.randn(batch_size, 1, state_input_dim, dtype=dtype, device="cuda")
    embodiment_id = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

    logger.info(f"  state: {state.shape} ({state.dtype})")
    logger.info(f"  embodiment_id: {embodiment_id.shape} ({embodiment_id.dtype})")
    logger.info(
        f"  (max_state_dim={config.max_state_dim}, "
        f"state_history_length={config.state_history_length})"
    )

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
        )

    logger.info("  State Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: Action Encoder
# ============================================================


def export_action_encoder_to_onnx(policy, output_dir, use_bf16=True, batch_size=1):
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
        batch_size, config.action_horizon, config.max_action_dim, dtype=dtype, device="cuda"
    )
    timesteps = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    embodiment_id = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

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
        )

    logger.info("  Action Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: DiT
# ============================================================


def export_dit_to_onnx(policy, captured_inputs, output_path, use_bf16=True, batch_size=1):
    """Export the DiT (AlternateVLDiT) to ONNX.

    N1.7: image_mask and backbone_attention_mask are always present
    (from Qwen3Backbone output).

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

    # Use captured shapes but replace batch dim (index 0) with batch_size
    sa_shape = (batch_size,) + captured_inputs.sa_embs.shape[1:]
    vl_shape = (batch_size,) + captured_inputs.vl_embs.shape[1:]
    ts_shape = (batch_size,)

    sa_embs = torch.randn(sa_shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(vl_shape, dtype=dtype, device="cuda")
    timestep = torch.ones(ts_shape, dtype=torch.int64, device="cuda")

    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]
    has_image_mask = captured_inputs.image_mask is not None
    has_backbone_mask = captured_inputs.backbone_attention_mask is not None

    if has_image_mask:
        im_shape = (batch_size,) + captured_inputs.image_mask.shape[1:]
        image_mask = torch.ones(im_shape, dtype=torch.bool, device="cuda")
        export_inputs.append(image_mask)
        input_names.append("image_mask")

    if has_backbone_mask:
        bm_shape = (batch_size,) + captured_inputs.backbone_attention_mask.shape[1:]
        backbone_attention_mask = torch.ones(bm_shape, dtype=torch.bool, device="cuda")
        export_inputs.append(backbone_attention_mask)
        input_names.append("backbone_attention_mask")

    logger.info("  Export input shapes:")
    for name, tensor in zip(input_names, export_inputs):
        logger.info(f"    {name}: {tensor.shape} ({tensor.dtype})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    logger.info(f"Exporting to {output_path}...")
    use_dynamo_exporter = _should_use_dynamo_exporter()
    logger.info(
        "Using %s ONNX exporter",
        "dynamo" if use_dynamo_exporter else "legacy",
    )

    # Create a wrapper to handle keyword arguments
    # torch.onnx.export uses positional args: `dit.forward(arg1, arg2...)`
    # DiT module uses keyword args: `dit.forward(hidden_states=....)`
    # The DiTWrapper handles this translation
    # Wrapper to convert positional args -> keyword args for DiT
    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit, use_image_mask, use_backbone_mask):
            super().__init__()
            self.dit = dit
            self.use_image_mask = use_image_mask
            self.use_backbone_mask = use_backbone_mask

        def forward(
            self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None
        ):
            kwargs = {}
            if self.use_image_mask and image_mask is not None:
                kwargs["image_mask"] = image_mask
            if self.use_backbone_mask and backbone_attention_mask is not None:
                kwargs["backbone_attention_mask"] = backbone_attention_mask
            return self.dit(sa_embs, vl_embs, timestep, **kwargs)

    wrapped_model = DiTWrapper(dit_model, has_image_mask, has_backbone_mask)
    wrapped_model.eval()

    # vl_seq_len varies with input text length — mark it dynamic so the TRT engine
    # can handle any sequence length seen at runtime, not just the export-time value.
    dit_dynamic_axes = {
        "vl_embs": {1: "vl_seq_len"},
    }
    if has_image_mask:
        dit_dynamic_axes["image_mask"] = {1: "vl_seq_len"}
    if has_backbone_mask:
        dit_dynamic_axes["backbone_attention_mask"] = {1: "vl_seq_len"}

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
            export_params=True,
            dynamic_axes=dit_dynamic_axes,
            dynamo=False,  # DiT specializes vl_seq_len under dynamo; legacy exporter needed
        )

    logger.info("  DiT exported successfully!")

    # Consolidate scattered external data files into a single .data file.
    # torch.onnx.export scatters large tensors into many small files (one per tensor).
    # TensorRT's parser expects external data in a single file adjacent to the .onnx.
    _consolidate_external_data(output_path)

    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: Action Decoder
# ============================================================


def export_action_decoder_to_onnx(policy, output_dir, use_bf16=True, batch_size=1):
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
    model_output = torch.randn(
        batch_size, sa_seq_len, config.hidden_size, dtype=dtype, device="cuda"
    )
    embodiment_id = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

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
        )

    logger.info("  Action Decoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Main
# ============================================================


def main(args):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    logger.info("=" * 80)
    logger.info("GR00T N1.7 ONNX Export Script")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Export mode: {args.export_mode}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

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

    # Step 3: Capture inputs via hooks
    logger.info("\n[Step 3] Capturing model inputs from actual inference...")

    dit_capture = DiTInputCapture()
    dit_hook = policy.model.action_head.model.register_forward_pre_hook(
        dit_capture.hook_fn, with_kwargs=True
    )

    # Also capture ViT and LLM inputs if doing full_pipeline
    vit_capture = None
    vit_hook = None
    llm_capture = None
    llm_hook = None
    if args.export_mode == "full_pipeline":
        vit_capture = ViTInputCapture()
        qwen_model = policy.model.backbone.model
        vit_hook = qwen_model.model.visual.register_forward_hook(
            vit_capture.hook_fn, with_kwargs=True
        )

        llm_capture = LLMInputCapture()
        llm_hook = qwen_model.model.language_model.register_forward_pre_hook(
            llm_capture.hook_fn, with_kwargs=True
        )

    observation = prepare_observation(policy, dataset, traj_idx=0)
    logger.info("  Running inference to capture shapes...")
    with torch.inference_mode():
        _ = policy.get_action(observation)

    dit_hook.remove()
    if vit_hook is not None:
        vit_hook.remove()
    if llm_hook is not None:
        llm_hook.remove()

    if not dit_capture.captured:
        logger.error("  Failed to capture DiT inputs!")
        return
    if args.export_mode == "full_pipeline" and not vit_capture.captured:
        logger.error("  Failed to capture ViT inputs!")
        return
    if args.export_mode == "full_pipeline" and not llm_capture.captured:
        logger.error("  Failed to capture LLM inputs!")
        return

    # Derive metadata
    action_head_config = policy.model.action_head.config
    sa_seq_len = 1 + action_head_config.action_horizon
    vl_seq_len = dit_capture.vl_embs.shape[1]

    # Save export metadata
    num_patches = vit_capture.pixel_values_shape[0] if vit_capture and vit_capture.captured else 256
    num_merged_patches = vit_capture.output_shape[0] if vit_capture and vit_capture.captured else 64
    # LLM metadata
    llm_seq_len = (
        llm_capture.inputs_embeds.shape[1] if llm_capture and llm_capture.captured else vl_seq_len
    )
    llm_hidden_size = (
        llm_capture.inputs_embeds.shape[2] if llm_capture and llm_capture.captured else 0
    )
    num_deepstack = (
        len(llm_capture.deepstack_visual_embeds)
        if (llm_capture and llm_capture.deepstack_visual_embeds)
        else 0
    )
    num_vis_tokens = llm_capture.deepstack_visual_embeds[0].shape[0] if num_deepstack > 0 else 0

    export_metadata = {
        "model_version": "n1d7",
        "sa_seq_len": int(sa_seq_len),
        "vl_seq_len": int(vl_seq_len),
        "llm_seq_len": int(llm_seq_len),
        "llm_hidden_size": int(llm_hidden_size),
        "num_deepstack": int(num_deepstack),
        "num_vis_tokens": int(num_vis_tokens),
        "num_patches": int(num_patches),  # ViT input seq length
        "num_merged_patches": int(num_merged_patches),  # ViT output after merger
        "action_horizon": int(action_head_config.action_horizon),
        "max_action_dim": int(action_head_config.max_action_dim),
        "max_state_dim": int(action_head_config.max_state_dim),
        "state_history_length": int(action_head_config.state_history_length),
        "hidden_size": int(action_head_config.hidden_size),
        "input_embedding_dim": int(action_head_config.input_embedding_dim),
        "backbone_embedding_dim": int(action_head_config.backbone_embedding_dim),
        "embodiment_tag": str(args.embodiment_tag),
        "export_mode": args.export_mode,
        "precision": args.precision,
        "batch_size": args.batch_size,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "export_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(export_metadata, f, indent=2)
    logger.info(f"  Saved export metadata to {metadata_path}")

    # Step 4: Export
    bs = args.batch_size
    if args.export_mode == "dit_only":
        logger.info("\n[Step 4] Exporting DiT to ONNX (dit_only mode)...")
        dit_output_path = os.path.join(args.output_dir, "dit_bf16.onnx")
        export_dit_to_onnx(
            policy=policy,
            captured_inputs=dit_capture,
            output_path=dit_output_path,
            use_bf16=True,
            batch_size=bs,
        )

    elif args.export_mode == "action_head":
        logger.info("\n[Step 4] Exporting action head components to ONNX...")

        # 4a. State Encoder
        logger.info("\n--- [4a] State Encoder ---")
        export_state_encoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

        # 4b. Action Encoder
        logger.info("\n--- [4b] Action Encoder ---")
        export_action_encoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

        # 4c. DiT
        logger.info("\n--- [4c] DiT ---")
        dit_output_path = os.path.join(args.output_dir, "dit_bf16.onnx")
        export_dit_to_onnx(
            policy=policy,
            captured_inputs=dit_capture,
            output_path=dit_output_path,
            use_bf16=True,
            batch_size=bs,
        )

        # 4d. Action Decoder
        logger.info("\n--- [4d] Action Decoder ---")
        export_action_decoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

    elif args.export_mode == "full_pipeline":
        logger.info("\n[Step 4] Exporting full pipeline to ONNX...")
        logger.info("  (ViT TRT + LLM TRT + Action Head TRT)")

        # 4a. ViT — exported in FP32 to avoid TRT BF16 kernel fusion accuracy issues
        # ViT is patch-level: for batch_size > 1, num_patches scales by batch_size
        logger.info("\n--- [4a] ViT (Qwen3-VL Vision, FP32 for TRT accuracy) ---")
        export_vit_to_onnx(policy, args.output_dir, vit_capture, use_bf16=False, batch_size=bs)

        # 4b. LLM
        logger.info("\n--- [4b] LLM (Qwen3-VL Text Model) ---")
        export_llm_to_onnx(policy, llm_capture, args.output_dir, use_bf16=True, batch_size=bs)

        # 4c. VL Self-Attention (if present)
        logger.info("\n--- [4c] VL Self-Attention ---")
        export_vl_self_attention_to_onnx(
            policy, args.output_dir, vl_seq_len=vl_seq_len, use_bf16=True, batch_size=bs
        )

        # 4d. State Encoder
        logger.info("\n--- [4d] State Encoder ---")
        export_state_encoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

        # 4e. Action Encoder
        logger.info("\n--- [4e] Action Encoder ---")
        export_action_encoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

        # 4f. DiT
        logger.info("\n--- [4f] DiT ---")
        dit_output_path = os.path.join(args.output_dir, "dit_bf16.onnx")
        export_dit_to_onnx(
            policy=policy,
            captured_inputs=dit_capture,
            output_path=dit_output_path,
            use_bf16=True,
            batch_size=bs,
        )

        # 4g. Action Decoder
        logger.info("\n--- [4g] Action Decoder ---")
        export_action_decoder_to_onnx(policy, args.output_dir, use_bf16=True, batch_size=bs)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nExported files in: {args.output_dir}")

    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info(f"  {f}: {size_mb:.2f} MB")


@dataclass
class ExportConfig:
    """Configuration for exporting GR00T N1.7 model to ONNX."""

    model_path: str
    """Path to the model checkpoint (required)."""

    dataset_path: str
    """Path to the dataset (required, used to capture input shapes)."""

    embodiment_tag: Optional[EmbodimentTag] = None
    """Embodiment tag. If not provided, auto-detected from model's processor_config.json."""

    output_dir: str = "./gr00t_n1d7_onnx"
    """Output directory for ONNX models."""

    export_mode: Literal["dit_only", "action_head", "full_pipeline"] = "dit_only"
    """Export mode: 'dit_only', 'action_head' (4 components), or 'full_pipeline' (ViT + action head)."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use."""

    precision: Literal["bf16", "fp8"] = "bf16"
    """Export precision: 'bf16' (default) or 'fp8' (TODO for N1.7)."""

    batch_size: int = 1
    """Batch size baked into the exported ONNX models (default: 1)."""


if __name__ == "__main__":
    args = tyro.cli(ExportConfig)
    if args.embodiment_tag is None:
        # Auto-detect from model's processor_config.json
        config_file = Path(args.model_path) / "processor_config.json"
        if not config_file.exists():
            raise ValueError(
                f"Cannot auto-detect embodiment_tag: {config_file} not found. "
                "Please provide --embodiment-tag explicitly."
            )
        with open(config_file, "r") as f:
            processor_config = json.load(f)
        modality_configs = processor_config.get("processor_kwargs", {}).get("modality_configs", {})
        if len(modality_configs) == 0:
            raise ValueError(
                "Cannot auto-detect embodiment_tag: no modality_configs found in processor_config.json. "
                "Please provide --embodiment-tag explicitly."
            )
        if len(modality_configs) == 1:
            embodiment_key = next(iter(modality_configs))
            args.embodiment_tag = EmbodimentTag.resolve(embodiment_key)
            logger.info(
                f"Auto-detected embodiment tag: {args.embodiment_tag} (from {embodiment_key})"
            )
        else:
            available = sorted(modality_configs.keys())
            raise ValueError(
                f"Multiple embodiments found in processor_config.json: {available}. "
                "Please provide --embodiment-tag explicitly."
            )
    main(args)
