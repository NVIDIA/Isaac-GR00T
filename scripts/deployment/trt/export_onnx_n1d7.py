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

    # DiT only (default). N1.7 TRT export uses the legacy ONNX exporter
    # explicitly (`dynamo=False`) so dynamic axes remain TensorRT-friendly.
    python -m scripts.deployment.trt.export_onnx_n1d7 \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo \\
        --output-dir ./gr00t_trt_deployment/onnx

    # Action head (4 components)
    python -m scripts.deployment.trt.export_onnx_n1d7 \\
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --dataset-path demo_data/libero_demo \\
        --output-dir ./gr00t_trt_deployment/onnx \\
        --export-mode action_head
"""

import copy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Callable, Literal, Optional

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.deployment.modes import ExportMode
from gr00t.model.modules.qwen3_backbone import _assign_inv_freq, recompute_vision_rotary_inv_freq
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import tyro

from ._trt_contract import EXPORT_METADATA_SCHEMA_VERSION
from .n1d7_optimization_config import (
    ACTION_HEAD_CONSTANTS_FILENAME,
    ACTION_HEAD_FORMAT_VERSION,
    ExecutionProfile,
    QuantizationPolicy,
    component_precisions,
    format_quantization_summary,
    resolve_dtype,
)


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _concatenate_linear_projections(linears: tuple[nn.Linear, ...]) -> nn.Linear:
    """Concatenate same-input Linear layers along their output dimension."""
    if not linears:
        raise ValueError("At least one Linear projection is required")
    if any(layer.in_features != linears[0].in_features for layer in linears):
        raise ValueError("Fused Linear projections must have identical input sizes")

    has_bias = [layer.bias is not None for layer in linears]
    if len(set(has_bias)) != 1:
        raise ValueError("Fused Linear projections must use the same bias setting")

    fused = nn.Linear(
        linears[0].in_features,
        sum(layer.out_features for layer in linears),
        bias=has_bias[0],
        device=linears[0].weight.device,
        dtype=linears[0].weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([layer.weight for layer in linears], dim=0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([layer.bias for layer in linears], dim=0))
    return fused


class FusedQwenMLP(nn.Module):
    """Export-only Qwen gated MLP with a single Gate/Up projection."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.config = source.config
        self.hidden_size = source.hidden_size
        self.intermediate_size = source.intermediate_size
        self.down_proj = source.down_proj
        self.act_fn = source.act_fn
        projections = (source.gate_proj, source.up_proj)
        self.gateup_out_features = tuple(layer.out_features for layer in projections)
        self.gateup_proj = _concatenate_linear_projections(projections)

    def forward(self, hidden_states):
        gate_states, up_states = self.gateup_proj(hidden_states).split(
            self.gateup_out_features, dim=-1
        )
        return self.down_proj(self.act_fn(gate_states) * up_states)


def fuse_llm_gateup(wrapper: nn.Module) -> int:
    """Fuse each decoder layer's Gate/Up projections into one export-only Linear."""
    for decoder_layer in wrapper.layers:
        decoder_layer.mlp = FusedQwenMLP(decoder_layer.mlp)
    logger.info("  Fused Gate/Up LLM projections in %d layers", len(wrapper.layers))
    return len(wrapper.layers)


class DiTCrossKV(nn.Module):
    """Project fixed VL embeddings to per-cross-block K/V once per frame."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    def forward(self, vl_embs):
        keys = []
        values = []
        for index, block in enumerate(self.dit.transformer_blocks):
            if index % 2 == 1:
                continue
            attention = block.attn1
            encoder_states = vl_embs
            if attention.norm_cross:
                encoder_states = attention.norm_encoder_hidden_states(encoder_states)
            key = attention.to_k(encoder_states)
            value = attention.to_v(encoder_states)
            batch_size, seq_len, inner_dim = key.shape
            head_dim = inner_dim // attention.heads
            key = key.view(batch_size, seq_len, attention.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, attention.heads, head_dim).transpose(1, 2)
            if attention.norm_k is not None:
                key = attention.norm_k(key)
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)


class CachedCrossKVDiT(nn.Module):
    """DiT consuming cached cross-K/V and precomputed timestep modulation."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    @staticmethod
    def _cross_attention(attention, hidden_states, key, value, attention_mask):
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        query = attention.to_q(hidden_states)
        head_dim = query.shape[-1] // attention.heads
        query = query.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        if attention.norm_q is not None:
            query = attention.norm_q(query)

        zero = torch.zeros((), dtype=query.dtype, device=query.device)
        negative_infinity = torch.full((), float("-inf"), dtype=query.dtype, device=query.device)
        additive_mask = torch.where(attention_mask[:, None, None, :], zero, negative_infinity)
        from gr00t.model.modules.dit import _sdpa_context

        # Match the upstream DiT's Spark math-SDPA workaround during export checks.
        with _sdpa_context():
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attention.heads * head_dim
        )
        hidden_states = attention.to_out[0](hidden_states.to(query.dtype))
        hidden_states = attention.to_out[1](hidden_states)
        if attention.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attention.rescale_output_factor

    def forward(
        self,
        sa_embs,
        cross_keys,
        cross_values,
        adaln_modulations,
        output_modulation,
        image_mask,
        backbone_attention_mask,
    ):
        hidden_states = sa_embs.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        cache_index = 0
        for index, block in enumerate(self.dit.transformer_blocks):
            scale, shift = adaln_modulations[index].chunk(2, dim=-1)
            normalized = block.norm1.norm(hidden_states)
            normalized = normalized * (1 + scale[:, None]) + shift[:, None]
            if block.pos_embed is not None:
                normalized = block.pos_embed(normalized)

            if index % 2 == 1:
                from gr00t.model.modules.dit import _sdpa_context

                with _sdpa_context():
                    attention_output = block.attn1(normalized)
            else:
                if index % (2 * self.dit.attend_text_every_n_blocks) == 0:
                    attention_mask = non_image_attention_mask
                else:
                    attention_mask = image_attention_mask
                attention_output = self._cross_attention(
                    block.attn1,
                    normalized,
                    cross_keys[cache_index],
                    cross_values[cache_index],
                    attention_mask,
                )
                cache_index += 1

            if block.final_dropout:
                attention_output = block.final_dropout(attention_output)
            hidden_states = attention_output + hidden_states
            hidden_states = block.ff(block.norm3(hidden_states)) + hidden_states

        shift, scale = output_modulation.chunk(2, dim=-1)
        hidden_states = self.dit.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        return self.dit.proj_out_2(hidden_states)


def save_action_head_constants(policy, dit_model, output_dir: str, batch_size: int):
    """Evaluate fixed timesteps once and persist action-head modulation constants."""
    action_head = policy.model.action_head
    num_steps = int(action_head.num_inference_timesteps)
    num_buckets = int(action_head.num_timestep_buckets)
    timestep_ids = torch.tensor(
        [int(step / float(num_steps) * num_buckets) for step in range(num_steps)],
        dtype=torch.int64,
        device=next(dit_model.parameters()).device,
    )

    adaln_modulations = []
    output_modulations = []
    action_time_embeddings = []
    action_encoder = action_head.action_encoder
    action_device = next(action_encoder.parameters()).device
    horizon = int(action_head.config.action_horizon)
    with torch.inference_mode():
        for timestep_id in timestep_ids:
            timestep = timestep_id.expand(batch_size)
            timestep_embedding = dit_model.timestep_encoder(timestep)
            block_modulations = [
                block.norm1.linear(block.norm1.silu(timestep_embedding))
                for block in dit_model.transformer_blocks
            ]
            adaln_modulations.append(torch.stack(block_modulations))
            output_modulations.append(dit_model.proj_out_1(F.silu(timestep_embedding)))
            action_timestep = timestep_id.to(action_device).expand(batch_size, horizon)
            action_time_embeddings.append(action_encoder.pos_encoding(action_timestep))

    constants = {
        "format_version": 1,
        "timesteps": timestep_ids.detach().cpu(),
        "action_time_embeddings": torch.stack(action_time_embeddings).detach().cpu(),
        "adaln_modulations": torch.stack(adaln_modulations).detach().cpu(),
        "output_modulations": torch.stack(output_modulations).detach().cpu(),
    }
    constants_path = os.path.join(output_dir, ACTION_HEAD_CONSTANTS_FILENAME)
    torch.save(constants, constants_path)
    logger.info("  Saved fixed timestep/AdaLN constants to %s", constants_path)
    return constants


def export_optimized_action_encoder(
    policy,
    output_dir: str,
    *,
    dtype: torch.dtype,
    batch_size: int,
):
    """Export Action Encoder with precomputed timestep embeddings as input."""

    action_encoder = policy.model.action_head.action_encoder
    config = policy.model.action_head.config

    class ActionEncoderWithTimeEmbedding(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.action_encoder = encoder

        def forward(self, actions, timestep_embedding, embodiment_id):
            action_embedding = self.action_encoder.W1(actions, embodiment_id)
            hidden_states = torch.cat([action_embedding, timestep_embedding], dim=-1)
            hidden_states = self.action_encoder.W2(hidden_states, embodiment_id)
            hidden_states = hidden_states * torch.sigmoid(hidden_states)
            return self.action_encoder.W3(hidden_states, embodiment_id)

    model = ActionEncoderWithTimeEmbedding(action_encoder.to(dtype)).eval().cuda()
    actions = torch.randn(
        batch_size,
        config.action_horizon,
        config.max_action_dim,
        dtype=dtype,
        device="cuda",
    )
    time_ids = torch.zeros(batch_size, config.action_horizon, dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        timestep_embedding = action_encoder.pos_encoding(time_ids).to(dtype)
    embodiment_id = torch.zeros(batch_size, dtype=torch.int64, device="cuda")

    output_path = os.path.join(output_dir, "action_encoder.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (actions, timestep_embedding, embodiment_id),
            output_path,
            input_names=["actions", "timestep_embedding", "embodiment_id"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
        )
    verify_onnx_export(output_path)
    return output_path


def export_optimized_dit(
    policy,
    captured_inputs,
    output_path: str,
    *,
    precision_tag: str,
    batch_size: int,
    pre_export_fn: Optional[Callable] = None,
):
    """Export split Cross-KV and cached/modulated DiT engines."""

    dtype = resolve_dtype(precision_tag)
    enable_nvfp4 = precision_tag == "nvfp4"
    dit_model = policy.model.action_head.model.to(dtype).eval().cuda()

    class CalibrationWrapper(nn.Module):
        def __init__(self, dit):
            super().__init__()
            self.dit = dit

        def forward(
            self,
            sa_embs,
            vl_embs,
            timestep,
            image_mask,
            backbone_attention_mask,
        ):
            return self.dit(
                sa_embs,
                vl_embs,
                timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )

    wrapped_model = CalibrationWrapper(dit_model).to(dtype).eval().cuda()
    sa_embs = (
        captured_inputs.sa_embs[:1]
        .expand(batch_size, *captured_inputs.sa_embs.shape[1:])
        .to(device="cuda", dtype=dtype)
    )
    vl_embs = (
        captured_inputs.vl_embs[:1]
        .expand(batch_size, *captured_inputs.vl_embs.shape[1:])
        .to(device="cuda", dtype=dtype)
    )
    image_mask = (
        captured_inputs.image_mask[:1]
        .expand(batch_size, *captured_inputs.image_mask.shape[1:])
        .to(device="cuda", dtype=torch.bool)
    )
    backbone_attention_mask = (
        captured_inputs.backbone_attention_mask[:1]
        .expand(batch_size, *captured_inputs.backbone_attention_mask.shape[1:])
        .to(device="cuda", dtype=torch.bool)
    )
    if pre_export_fn is not None:
        pre_export_fn(wrapped_model)

    constants = save_action_head_constants(
        policy, dit_model, os.path.dirname(output_path), batch_size
    )
    cross_kv_model = DiTCrossKV(dit_model).to(dtype).eval().cuda()
    optimized_dit = CachedCrossKVDiT(dit_model).to(dtype).eval().cuda()
    with torch.inference_mode():
        cross_keys, cross_values = cross_kv_model(vl_embs)

    adaln_modulations = constants["adaln_modulations"][0].to("cuda", dtype=dtype)
    output_modulation = constants["output_modulations"][0].to("cuda", dtype=dtype)
    reference_timestep = constants["timesteps"][0].to("cuda").expand(batch_size)
    with torch.inference_mode():
        reference_output = wrapped_model(
            sa_embs,
            vl_embs,
            reference_timestep,
            image_mask,
            backbone_attention_mask,
        )
        optimized_output = optimized_dit(
            sa_embs,
            cross_keys,
            cross_values,
            adaln_modulations,
            output_modulation,
            image_mask,
            backbone_attention_mask,
        )
    cosine = F.cosine_similarity(
        reference_output.float().flatten().unsqueeze(0),
        optimized_output.float().flatten().unsqueeze(0),
    ).item()
    max_abs = (reference_output.float() - optimized_output.float()).abs().max().item()
    logger.info(
        "  Optimized DiT PyTorch equivalence: cosine=%.8f, max_abs=%.6f",
        cosine,
        max_abs,
    )
    if not np.isfinite(cosine) or cosine < 0.999:
        raise RuntimeError(f"Optimized DiT equivalence check failed: cosine={cosine:.8f}")

    cross_kv_path = os.path.join(
        os.path.dirname(output_path),
        os.path.basename(output_path).replace("dit_", "dit_cross_kv_", 1),
    )
    with torch.inference_mode():
        torch.onnx.export(
            cross_kv_model,
            (vl_embs,),
            cross_kv_path,
            input_names=["vl_embs"],
            output_names=["cross_keys", "cross_values"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes={
                "vl_embs": {1: "vl_seq_len"},
                "cross_keys": {3: "vl_seq_len"},
                "cross_values": {3: "vl_seq_len"},
            },
            dynamo=False,
        )
    _consolidate_external_data(cross_kv_path)
    verify_onnx_export(cross_kv_path)

    optimized_inputs = (
        sa_embs,
        cross_keys,
        cross_values,
        adaln_modulations,
        output_modulation,
        image_mask,
        backbone_attention_mask,
    )
    input_names = [
        "sa_embs",
        "cross_keys",
        "cross_values",
        "adaln_modulations",
        "output_modulation",
        "image_mask",
        "backbone_attention_mask",
    ]
    attention_handles = _patch_dit_attn_for_export(dit_model)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                optimized_dit,
                optimized_inputs,
                output_path,
                input_names=input_names,
                output_names=["output"],
                opset_version=19,
                do_constant_folding=True,
                export_params=True,
                dynamic_axes={
                    "cross_keys": {3: "vl_seq_len"},
                    "cross_values": {3: "vl_seq_len"},
                    "image_mask": {1: "vl_seq_len"},
                    "backbone_attention_mask": {1: "vl_seq_len"},
                },
                dynamo=False,
            )
    finally:
        _unpatch_dit_attn(attention_handles)
    _consolidate_external_data(output_path, enable_nvfp4=enable_nvfp4)
    verify_onnx_export(output_path, skip_check=enable_nvfp4)
    return output_path


def _consolidate_external_data(onnx_path: str, enable_nvfp4: bool = False) -> None:
    """Merge scattered external-data files into a single .data file next to the ONNX model.

    When ``enable_nvfp4`` is True, also converts FP4 QDQ ops to 2DQ format
    for TensorRT compatibility.
    """
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
        and not f.endswith((".onnx", ".json", ".data", ".pt"))
        and f != data_file
    ]
    if not scattered and not enable_nvfp4:
        return

    if scattered:
        logger.info(f"  Consolidating {len(scattered)} external data files into {data_file}...")
    model = onnx.load(onnx_path, load_external_data=True)

    if enable_nvfp4:
        logger.info("  Converting NVFP4 QDQ ops to 2DQ format for TensorRT...")
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

        model = fp4qdq_to_2dq(model, verbose=True)

    convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location=data_file, size_threshold=0
    )
    onnx.save(model, onnx_path)

    # Clean up scattered files
    for f in scattered:
        os.remove(os.path.join(onnx_dir, f))
    if scattered:
        logger.info(f"  Consolidated and cleaned up {len(scattered)} files.")


def verify_onnx_export(onnx_path: str, skip_check: bool = False) -> None:
    """Load and check the exported ONNX model for validity.

    ``skip_check=True`` skips ``onnx.checker.check_model`` (used for NVFP4 LLM
    whose 2DQ-converted graph trips the stock checker but is valid for TRT).
    """
    import onnx

    logger.info(f"  Verifying {onnx_path} ...")
    if skip_check:
        model = onnx.load(onnx_path, load_external_data=False)
        logger.info(
            f"  ONNX model loaded (skip_check=True): {len(model.graph.node)} nodes, "
            f"{len(model.graph.initializer)} initializers."
        )
        return
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


def _restore_vision_rotary_inv_freq_fp32(vision, vision_config):
    """Re-derive the ViT RoPE ``inv_freq`` in fp32 right before baking it into the engine.

    ``Gr00tPolicy`` loads the whole model in bf16 (``gr00t_policy.py``), which rounds
    this non-persistent buffer even though the ViT itself is exported in fp32 for TRT
    accuracy (max|delta| ~1.8e-4 for head_dim=64 -- a pure bf16 round-trip error).
    Re-deriving the analytic fp32 value here keeps the baked rotary cos/sin at full
    fp32 precision and lets the analytic-oracle guard verify a clean buffer instead of
    the bf16-degraded one. Fails closed when the rotary submodule/layout is missing.
    """
    rotary = getattr(vision, "rotary_pos_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise RuntimeError(
            "ViT export: vision rotary_pos_emb/inv_freq not found; cannot rebuild the "
            "rotary buffers before baking them into the engine (transformers Qwen3-VL "
            "layout drift). Refusing to export unverified rotary."
        )
    head_dim = vision_config.hidden_size // vision_config.num_heads
    fp32_inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, rotary.inv_freq.device)
    _assign_inv_freq(rotary, "inv_freq", fp32_inv_freq, persistent=False)


def _assert_vision_rotary_matches_analytic(vision, vision_config, *, atol=1e-5, rtol=1e-4):
    """Abort export if the loaded ViT RoPE ``inv_freq`` drifts from the analytic oracle.

    The baked rotary cos/sin derive from this non-persistent buffer; the post-export
    cosine checks are backend-vs-backend and cannot catch a common-mode error in it.
    """
    rotary = getattr(vision, "rotary_pos_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise RuntimeError(
            "ViT export: vision rotary_pos_emb/inv_freq not found; cannot verify the "
            "rotary buffers before baking them into the engine (transformers Qwen3-VL "
            "layout drift). Refusing to export unverified rotary."
        )
    head_dim = vision_config.hidden_size // vision_config.num_heads
    baked = rotary.inv_freq.detach().float()
    analytic = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, baked.device)
    if not torch.allclose(baked, analytic, atol=atol, rtol=rtol):
        max_err = (baked - analytic).abs().max().item()
        raise RuntimeError(
            "ViT export: loaded vision RoPE inv_freq diverges from the analytic oracle "
            f"(max|delta|={max_err:.3e}); the baked rotary cos/sin would be silently wrong "
            "and the backend-vs-backend cosine checks cannot catch it. Ensure "
            "Qwen3Backbone._reset_rotary_inv_freq ran at load before exporting."
        )


def export_vit_to_onnx(
    policy,
    output_dir,
    captured_vit,
    use_bf16=True,
    batch_size=1,
    *,
    precision_tag=None,
    pre_export_fn=None,
):
    """Export Qwen3-VL Vision Model to ONNX.

    Pre-computes position/rotary embeddings for the captured grid_thw to avoid
    ComplexDouble ops. Monkey-patches attention to use standard SDPA (valid for
    single-image inference where all patches attend to all patches).

    Input: pixel_values [num_patches * batch_size, C*T*pH*pW]
    Output: image_embeds [num_merged_patches * batch_size, hidden_dim],
            deepstack_features [num_layers, num_merged_patches * batch_size, hidden_dim]

    Quantization hooks (optional, used by the FP8/NVFP4 policies in main()):
        precision_tag:  Precision tag driving the export dtype and the output
                        filename (e.g. "fp8" -> vit_fp8.onnx).
        pre_export_fn:  Called as ``pre_export_fn(wrapper)`` after the wrapper is
                        built but before ``torch.onnx.export``. Use for modelopt
                        calibration / quantization.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting ViT (Qwen3-VL Vision) to ONNX")
    logger.info("=" * 80)

    backbone = policy.model.backbone
    qwen_model = backbone.model
    vision = qwen_model.model.visual

    # Gr00tPolicy loads the model in bf16, which rounds the non-persistent RoPE
    # inv_freq. The ViT is exported in fp32 by default for TRT accuracy, so
    # re-derive the analytic fp32 inv_freq before baking cos/sin, then verify
    # the buffer against the independent analytic oracle.
    _restore_vision_rotary_inv_freq_fp32(vision, qwen_model.config.vision_config)
    _assert_vision_rotary_matches_analytic(vision, qwen_model.config.vision_config)

    dtype = (
        resolve_dtype(precision_tag)
        if precision_tag
        else (torch.bfloat16 if use_bf16 else torch.float32)
    )
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

    # --- Quantization hook: calibrate / quantize the wrapper in-place ---
    if pre_export_fn is not None:
        pre_export_fn(wrapper)

    ptag = precision_tag or ("bf16" if use_bf16 else "fp32")
    output_path = os.path.join(output_dir, f"vit_{ptag}.onnx")
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
            dynamo=False,
        )

    logger.info("  ViT exported successfully!")
    _consolidate_external_data(output_path)
    verify_onnx_export(output_path)

    # ORT verification: compare ONNX output against PyTorch wrapper.
    # Must run BEFORE restoring attention, since wrapper uses patched attention.
    # Skip for quantized models (TRT custom Q/DQ ops are not runnable in ORT).
    if pre_export_fn is None:
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


def export_llm_to_onnx(
    policy,
    captured_llm,
    output_dir,
    use_bf16=True,
    batch_size=1,
    *,
    precision_tag=None,
    pre_export_fn=None,
    fuse_gateup=False,
):
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

    Quantization hooks (optional, used by the FP8/NVFP4 policies in main()):
        precision_tag:  Precision tag driving the export dtype, the output
                        filename (e.g. "nvfp4" -> llm_nvfp4.onnx), and the
                        NVFP4 2DQ post-processing.
        pre_export_fn:  Called as ``pre_export_fn(wrapper)`` after the wrapper is
                        built but before ``torch.onnx.export``.
        fuse_gateup:    Fuse Gate/Up projections into one Linear per layer.
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

    if fuse_gateup:
        fuse_llm_gateup(wrapper)

    dtype = (
        resolve_dtype(precision_tag)
        if precision_tag
        else (torch.bfloat16 if use_bf16 else torch.float32)
    )
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

    # --- Quantization hook: calibrate / quantize the wrapper in-place ---
    if pre_export_fn is not None:
        pre_export_fn(wrapper)

    ptag = precision_tag or ("bf16" if use_bf16 else "fp32")
    output_path = os.path.join(output_dir, f"llm_{ptag}.onnx")
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
            dynamo=False,
        )

    logger.info("  LLM exported successfully!")
    is_nvfp4 = precision_tag == "nvfp4"
    _consolidate_external_data(output_path, enable_nvfp4=is_nvfp4)
    verify_onnx_export(output_path, skip_check=is_nvfp4)
    return output_path


# ============================================================
# Observation Helpers
# ============================================================


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


def export_vl_self_attention_to_onnx(
    policy,
    output_dir,
    vl_seq_len,
    use_bf16=True,
    batch_size=1,
    *,
    dtype_override=None,
    pre_export_fn=None,
):
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
    dtype = dtype_override or (torch.bfloat16 if use_bf16 else torch.float32)
    model = vl_sa.to(dtype).eval().cuda()

    hidden_states = torch.randn(
        batch_size, vl_seq_len, config.backbone_embedding_dim, dtype=dtype, device="cuda"
    )
    logger.info(f"  hidden_states: {hidden_states.shape} ({hidden_states.dtype})")

    output_path = os.path.join(output_dir, "vl_self_attention.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if pre_export_fn is not None:
        pre_export_fn(model)

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
            export_params=True,
            dynamo=False,
        )

    logger.info("  VL Self-Attention exported successfully!")
    _consolidate_external_data(output_path)
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: State Encoder
# ============================================================


def export_state_encoder_to_onnx(
    policy,
    output_dir,
    use_bf16=True,
    batch_size=1,
    *,
    dtype_override=None,
):
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

    dtype = dtype_override or (torch.bfloat16 if use_bf16 else torch.float32)
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
            # torch 2.9 flipped torch.onnx.export's dynamo default to True, but the
            # dynamo graph for the state/action encoder + action decoder fuses into a
            # Myelin ForeignNode (Unsqueeze...add) that TRT 10.15's NVRTC backend can't
            # compile at bs=2 on Blackwell. Pin the legacy exporter for these three
            # (ViT/LLM/VL-SA stay on dynamo — legacy can't export Qwen3VL memory_format).
            dynamo=False,
        )

    logger.info("  State Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: Action Encoder
# ============================================================


def export_action_encoder_to_onnx(
    policy,
    output_dir,
    use_bf16=True,
    batch_size=1,
    *,
    dtype_override=None,
):
    """Export the action encoder (MultiEmbodimentActionEncoder) to ONNX.

    Input: actions [B, action_horizon, max_action_dim], timesteps [B], embodiment_id [B]
    Output: [B, action_horizon, input_embedding_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Action Encoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    action_encoder = policy.model.action_head.action_encoder

    dtype = dtype_override or (torch.bfloat16 if use_bf16 else torch.float32)
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
            dynamo=False,
        )

    logger.info("  Action Encoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Export Functions: DiT
# ============================================================


def _dit_attn_bool_mask_to_additive_pre_hook(module, args, kwargs):
    """Convert bool attention_mask to additive (0/-inf) in the compute dtype."""
    am = kwargs.get("attention_mask")
    if am is None or am.dtype != torch.bool:
        return None
    q = args[0] if args else kwargs.get("hidden_states")
    if q is None:
        return None
    dtype = q.dtype
    zero = torch.zeros((), dtype=dtype, device=am.device)
    ninf = torch.full((), float("-inf"), dtype=dtype, device=am.device)
    kwargs["attention_mask"] = torch.where(am, zero, ninf)
    return args, kwargs


def _patch_dit_attn_for_export(dit_model):
    """Register pre-hooks on every BasicTransformerBlock.attn1 to enable TRT
    mha_v2 fusion on cross-attn. Returns handles for restoration."""
    handles = []
    blocks = getattr(dit_model, "transformer_blocks", None)
    if blocks is None:
        logger.warning("  DiT has no transformer_blocks; skipping mha_v2 fusion patch")
        return handles
    for blk in blocks:
        attn = getattr(blk, "attn1", None)
        if attn is None:
            continue
        h = attn.register_forward_pre_hook(
            _dit_attn_bool_mask_to_additive_pre_hook, with_kwargs=True
        )
        handles.append(h)
    logger.info(
        f"  Patched {len(handles)} DiT attn blocks: bool mask -> additive "
        f"(enables TRT mha_v2 fusion on cross-attn)"
    )
    return handles


def _unpatch_dit_attn(handles):
    for h in handles:
        h.remove()


def export_dit_to_onnx(
    policy,
    captured_inputs,
    output_path,
    use_bf16=True,
    batch_size=1,
    *,
    precision_tag=None,
    pre_export_fn=None,
):
    """Export the DiT (AlternateVLDiT) to ONNX.

    N1.7: image_mask and backbone_attention_mask are always present
    (from Qwen3Backbone output).

    Input: sa_embs [B, sa_seq_len, input_embedding_dim],
           vl_embs [B, vl_seq_len, backbone_embedding_dim],
           timestep [B], image_mask [B, vl_seq_len],
           backbone_attention_mask [B, vl_seq_len]
    Output: [B, sa_seq_len, hidden_size]

    Quantization hooks (optional, used by the FP8/NVFP4 policies in main()):
        precision_tag:  Precision tag driving the export dtype and the NVFP4
                        2DQ post-processing (the caller names the output file).
        pre_export_fn:  Called as ``pre_export_fn(wrapped_model)`` after the
                        wrapper is built but before ``torch.onnx.export``.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting DiT to ONNX")
    logger.info("=" * 80)

    dit_model = policy.model.action_head.model
    dit_model.eval()

    enable_nvfp4 = precision_tag == "nvfp4"
    dtype = (
        resolve_dtype(precision_tag)
        if precision_tag
        else (torch.bfloat16 if use_bf16 else torch.float32)
    )
    dit_model = dit_model.to(dtype).cuda()

    # Build the DiT wrapper up front so callers can run calibration through it.
    has_image_mask = captured_inputs.image_mask is not None
    has_backbone_mask = captured_inputs.backbone_attention_mask is not None

    # Wrapper to convert positional args -> keyword args for DiT.
    # torch.onnx.export uses positional args: `dit.forward(arg1, arg2...)`.
    # DiT module uses keyword args: `dit.forward(hidden_states=....)`.
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
    wrapped_model = wrapped_model.to(dtype).eval().cuda()

    # Use captured shapes but replace batch dim (index 0) with batch_size
    sa_shape = (batch_size,) + captured_inputs.sa_embs.shape[1:]
    vl_shape = (batch_size,) + captured_inputs.vl_embs.shape[1:]
    ts_shape = (batch_size,)

    sa_embs = torch.randn(sa_shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(vl_shape, dtype=dtype, device="cuda")
    timestep = torch.ones(ts_shape, dtype=torch.int64, device="cuda")

    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]

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

    # Export to ONNX. Keep the legacy exporter explicit: the dynamo exporter
    # specializes vl_seq_len here, which breaks the dynamic TensorRT profile.
    logger.info(f"Exporting to {output_path} with legacy ONNX exporter...")

    # --- Quantization hook: calibrate / quantize the wrapper in-place ---
    if pre_export_fn is not None:
        pre_export_fn(wrapped_model)

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
    mha_patch_handles = _patch_dit_attn_for_export(dit_model)
    try:
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
                dynamo=False,
            )
    finally:
        _unpatch_dit_attn(mha_patch_handles)

    logger.info("  DiT exported successfully!")

    # Consolidate scattered external data files into a single .data file.
    # torch.onnx.export scatters large tensors into many small files (one per tensor).
    # TensorRT's parser expects external data in a single file adjacent to the .onnx.
    _consolidate_external_data(output_path, enable_nvfp4=enable_nvfp4)

    verify_onnx_export(output_path, skip_check=enable_nvfp4)
    return output_path


# ============================================================
# Export Functions: Action Decoder
# ============================================================


def export_action_decoder_to_onnx(
    policy,
    output_dir,
    use_bf16=True,
    batch_size=1,
    *,
    dtype_override=None,
):
    """Export the action decoder (CategorySpecificMLP) to ONNX.

    Input: model_output [B, sa_seq_len, hidden_size], embodiment_id [B]
    Output: [B, sa_seq_len, max_action_dim]
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Action Decoder to ONNX")
    logger.info("=" * 80)

    config = policy.model.action_head.config
    action_decoder = policy.model.action_head.action_decoder

    dtype = dtype_override or (torch.bfloat16 if use_bf16 else torch.float32)
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
            dynamo=False,
        )

    logger.info("  Action Decoder exported successfully!")
    verify_onnx_export(output_path)
    return output_path


# ============================================================
# Main
# ============================================================


def main(args):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    optimized_profile = args.execution_profile == "optimized"
    if optimized_profile and args.export_mode == "dit_only":
        raise ValueError(
            "--execution-profile optimized requires --export-mode action_head or full_pipeline"
        )

    # Per-component precision tags come from the shared policy table so export,
    # engine build, and runtime can never disagree on artifact names.
    precisions = component_precisions(args.quantization)
    vit_prec = precisions["vit"]
    llm_prec = precisions["llm"]
    dit_prec = precisions["dit"]
    aux_prec = precisions["aux"]
    aux_dtype = resolve_dtype(aux_prec)
    quantized = args.quantization != "none"

    logger.info("=" * 80)
    logger.info("GR00T N1.7 ONNX Export Script")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Export mode: {args.export_mode}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(
        f"Component precisions: vit={vit_prec}, llm={llm_prec}, dit={dit_prec}, aux={aux_prec}"
    )
    logger.info(f"Execution profile: {args.execution_profile}")
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
        "schema_version": EXPORT_METADATA_SCHEMA_VERSION,
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
        "vit_precision": vit_prec,
        "llm_precision": llm_prec,
        "dit_precision": dit_prec,
        "aux_precision": aux_prec,
        "quantization": args.quantization,
        "execution_profile": args.execution_profile,
        "optimized_action_head": optimized_profile,
        "action_head_format_version": (ACTION_HEAD_FORMAT_VERSION if optimized_profile else None),
        "num_inference_timesteps": int(policy.model.action_head.num_inference_timesteps),
        "num_timestep_buckets": int(policy.model.action_head.num_timestep_buckets),
        "batch_size": args.batch_size,
    }
    if quantized:
        export_metadata["calibration"] = {
            "dataset": args.calib_dataset_path or args.dataset_path,
            "embodiment": str(args.embodiment_tag),
            "samples": args.calib_size,
        }
    # The ViT engine bakes pos/rotary buffers for this grid_thw and takes only
    # pixel_values as input; record the grid so the runtime can reject a
    # mismatched image config instead of silently using wrong embeddings.
    if vit_capture is not None and vit_capture.captured:
        export_metadata["vit_grid_thw"] = [
            [int(x) for x in row] for row in vit_capture.grid_thw.tolist()
        ]

    logger.info("\n" + format_quantization_summary(export_metadata))
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "export_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(export_metadata, f, indent=2)
    logger.info(f"  Saved export metadata to {metadata_path}")

    # Step 3.5: capture calibration inputs. Must run on the still-unquantized
    # policy: modelopt fake-quant mutates modules in place during Step 4, so a
    # later capture would observe quantized upstream activations.
    bs = args.batch_size
    calib_batches: dict = {}
    llm_input_names = None
    dit_input_names = ["sa_embs", "vl_embs", "timestep"]
    if dit_capture.image_mask is not None:
        dit_input_names.append("image_mask")
    if dit_capture.backbone_attention_mask is not None:
        dit_input_names.append("backbone_attention_mask")

    if quantized:
        from .n1d7_calibration import calibrate_component, capture_calibration_batches

        needed = {"dit"}
        if args.export_mode == "full_pipeline":
            needed |= {"vit", "llm"}
            if not isinstance(policy.model.action_head.vl_self_attention, nn.Identity):
                needed.add("vl_sa")
        logger.info("\n[Step 3.5] Capturing calibration data (before any quantization)...")
        calib_batches = capture_calibration_batches(
            policy,
            dataset_path=args.calib_dataset_path or args.dataset_path,
            calib_size=args.calib_size,
            components=needed,
        )
        if args.export_mode == "full_pipeline":
            llm_input_names = ["inputs_embeds", "attention_mask", "position_ids"]
            if num_deepstack > 0 and llm_capture.visual_pos_masks is not None:
                llm_input_names.append("visual_pos_masks")
                llm_input_names.extend(f"deepstack_{i}" for i in range(num_deepstack))

    def _pre_export(component, tag, input_names=None):
        """Calibration hook for one component; None for high-precision tags."""
        if tag not in ("fp8", "nvfp4"):
            return None

        def fn(wrapper):
            calibrate_component(
                wrapper, component, tag, calib_batches[component], input_names=input_names
            )

        return fn

    def _export_action_encoder_component():
        if optimized_profile:
            return export_optimized_action_encoder(
                policy, args.output_dir, dtype=aux_dtype, batch_size=bs
            )
        return export_action_encoder_to_onnx(
            policy, args.output_dir, batch_size=bs, dtype_override=aux_dtype
        )

    def _export_dit_component():
        output_path = os.path.join(args.output_dir, f"dit_{dit_prec}.onnx")
        pre_export_fn = _pre_export("dit", dit_prec, input_names=dit_input_names)
        if optimized_profile:
            if dit_capture.image_mask is None or dit_capture.backbone_attention_mask is None:
                raise ValueError("Optimized action head requires both DiT attention masks")
            return export_optimized_dit(
                policy,
                dit_capture,
                output_path,
                precision_tag=dit_prec,
                batch_size=bs,
                pre_export_fn=pre_export_fn,
            )
        return export_dit_to_onnx(
            policy=policy,
            captured_inputs=dit_capture,
            output_path=output_path,
            batch_size=bs,
            precision_tag=dit_prec,
            pre_export_fn=pre_export_fn,
        )

    # Step 4: Export
    if args.export_mode == "dit_only":
        logger.info(f"\n[Step 4] Exporting DiT to ONNX (dit_only, precision={dit_prec})...")
        _export_dit_component()

    elif args.export_mode == "action_head":
        logger.info("\n[Step 4] Exporting action head components to ONNX...")

        # 4a. State Encoder
        logger.info("\n--- [4a] State Encoder ---")
        export_state_encoder_to_onnx(
            policy, args.output_dir, batch_size=bs, dtype_override=aux_dtype
        )

        # 4b. Action Encoder
        logger.info("\n--- [4b] Action Encoder ---")
        _export_action_encoder_component()

        # 4c. DiT
        logger.info(f"\n--- [4c] DiT (precision={dit_prec}) ---")
        _export_dit_component()

        # 4d. Action Decoder
        logger.info("\n--- [4d] Action Decoder ---")
        export_action_decoder_to_onnx(
            policy, args.output_dir, batch_size=bs, dtype_override=aux_dtype
        )

    elif args.export_mode == "full_pipeline":
        logger.info("\n[Step 4] Exporting full pipeline to ONNX...")

        # 4a. ViT
        logger.info(f"\n--- [4a] ViT (precision={vit_prec}) ---")
        export_vit_to_onnx(
            policy,
            args.output_dir,
            vit_capture,
            batch_size=bs,
            precision_tag=vit_prec,
            pre_export_fn=_pre_export("vit", vit_prec),
        )

        # 4b. LLM
        logger.info(f"\n--- [4b] LLM (precision={llm_prec}) ---")
        export_llm_to_onnx(
            policy,
            llm_capture,
            args.output_dir,
            batch_size=bs,
            precision_tag=llm_prec,
            pre_export_fn=_pre_export("llm", llm_prec, input_names=llm_input_names),
            fuse_gateup=optimized_profile,
        )

        # 4c. VL Self-Attention (if present). Under a quantized policy only its
        # FFN linears are quantized to FP8; attention stays high precision.
        logger.info("\n--- [4c] VL Self-Attention ---")
        export_vl_self_attention_to_onnx(
            policy,
            args.output_dir,
            vl_seq_len=vl_seq_len,
            batch_size=bs,
            dtype_override=aux_dtype,
            pre_export_fn=_pre_export("vl_sa", "fp8" if quantized else aux_prec),
        )

        # 4d. State Encoder
        logger.info("\n--- [4d] State Encoder ---")
        export_state_encoder_to_onnx(
            policy, args.output_dir, batch_size=bs, dtype_override=aux_dtype
        )

        # 4e. Action Encoder
        logger.info("\n--- [4e] Action Encoder ---")
        _export_action_encoder_component()

        # 4f. DiT
        logger.info(f"\n--- [4f] DiT (precision={dit_prec}) ---")
        _export_dit_component()

        # 4g. Action Decoder
        logger.info("\n--- [4g] Action Decoder ---")
        export_action_decoder_to_onnx(
            policy, args.output_dir, batch_size=bs, dtype_override=aux_dtype
        )

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
    """Configuration for exporting GR00T N1.7 model to ONNX.

    Usage:
        # BF16 baseline (default)
        python -m scripts.deployment.trt.export_onnx_n1d7 --model-path ... --dataset-path ...

        # Fixed quantized precision recipes (see COMPONENT_PRECISIONS)
        python -m scripts.deployment.trt.export_onnx_n1d7 --model-path ... --dataset-path ... \\
            --quantization fp8 --export-mode full_pipeline
        python -m scripts.deployment.trt.export_onnx_n1d7 --model-path ... --dataset-path ... \\
            --quantization mixed_nvfp4 --export-mode full_pipeline

        # Cross-KV cache + folded timestep + CUDA-graph action head
        python -m scripts.deployment.trt.export_onnx_n1d7 --model-path ... --dataset-path ... \\
            --execution-profile optimized --export-mode full_pipeline
    """

    model_path: str
    """Path to the model checkpoint (required)."""

    dataset_path: str
    """Path to the dataset (required, used to capture input shapes)."""

    embodiment_tag: Optional[EmbodimentTag] = None
    """Embodiment tag. If not provided, auto-detected from model's processor_config.json."""

    output_dir: str = "./gr00t_trt_deployment/onnx"
    """Output directory for ONNX models."""

    export_mode: ExportMode = ExportMode.dit_only
    """Export mode: 'dit_only', 'action_head' (4 components), or 'full_pipeline'."""

    precision: Literal["bf16"] = "bf16"
    """Export precision of the default (non-quantized) graph.

    Fixed to 'bf16' (with the ViT promoted to FP32 for TRT accuracy).
    Quantized exports are selected via --quantization, which maps to a fixed
    per-component precision recipe in
    trt.n1d7_optimization_config.COMPONENT_PRECISIONS."""

    execution_profile: ExecutionProfile = "baseline"
    """'optimized' adds the cross-KV-cache action head (CUDA-graph runtime) and
    fuses the LLM Gate/Up projections. Requires action_head or full_pipeline
    export mode."""

    quantization: QuantizationPolicy = "none"
    """Quantization policy: none (BF16 baseline), fp8, or mixed_nvfp4."""

    calib_dataset_path: Optional[str] = None
    """LeRobot dataset for calibration. Defaults to --dataset-path."""

    calib_size: int = 10
    """Number of calibration samples (default: 10)."""

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
