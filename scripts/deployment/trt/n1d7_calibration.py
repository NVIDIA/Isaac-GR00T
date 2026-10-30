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

"""FP8 / NVFP4 post-training-quantization calibration for GR00T N1.7 export.

Two-phase design, consumed by ``export_onnx_n1d7.py``:

1. :func:`capture_calibration_batches` walks the calibration dataset ONCE
   through the still-unquantized policy and caches every component's inputs
   on CPU. It must run before any component is exported: modelopt fake-quant
   mutates the policy's modules in place, so later captures would observe
   quantized upstream activations.
2. :func:`calibrate_component` replays the cached batches through a
   component's export wrapper under the modelopt recipe selected by
   ``(component, precision_tag)``. The per-layer breakdown of the mixed
   recipes lives in :func:`_quant_config`:

   - llm/nvfp4: NVFP4 AWQ, with o_proj and down_proj kept in FP8.
   - dit/nvfp4: FP8 everywhere, with the FFN linears in NVFP4.
   - vl_sa/fp8: only the FFN linears in FP8, attention stays high precision.
"""

import copy
import logging
import os
import sys
import time
from typing import Any, Optional

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Compat shim: modelopt (<=0.39) + torch>=2.10
# ---------------------------------------------------------------------------


def _patch_modelopt_jitscalartype_import() -> None:
    """Inject ``JitScalarType`` into the torchscript exporter namespace so that
    older modelopt versions don't crash on newer PyTorch."""
    try:
        import torch.onnx._internal.torchscript_exporter as _tse
    except Exception:
        return
    if hasattr(_tse, "JitScalarType"):
        return
    try:
        from torch.onnx._internal.torchscript_exporter._type_utils import (
            JitScalarType as _JitScalarType,
        )
    except Exception:
        return
    _tse.JitScalarType = _JitScalarType


_patch_modelopt_jitscalartype_import()

# Make sibling modules importable when run directly from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Phase 1: capture calibration inputs from the clean policy
# ============================================================


def _prepare_step_observation(
    loader: LeRobotEpisodeLoader, ep_idx: int, step_idx: int, policy: Gr00tPolicy
) -> dict:
    """Prepare a single trajectory step as a batched observation dict."""
    traj = loader[ep_idx]
    modality_configs = policy.modality_configs

    data_point = extract_step_data(
        traj,
        step_idx,
        modality_configs={k: v for k, v in modality_configs.items() if k != "action"},
        embodiment_tag=policy.embodiment_tag,
    )

    observation: dict[str, Any] = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in modality_configs["language"].modality_keys:
        observation[key] = data_point.text

    new_obs: dict[str, Any] = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = observation[parsed_key]
            new_obs[modality][key] = [[arr]] if isinstance(arr, str) else arr[None, :]
    return new_obs


def _collate_inputs(policy: Gr00tPolicy, data: dict) -> dict:
    """Process one observation through the VLA processor and collate."""
    unbatched_observations = policy._unbatch_observation(data)
    processed_inputs = []
    for obs in unbatched_observations:
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated = policy.collate_fn(processed_inputs)
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return collated.get("inputs", collated)


def _to_cpu(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in batch.items()}


def _capture_llm_batch(policy: Gr00tPolicy, inputs: dict) -> dict[str, torch.Tensor]:
    """Mirror the runtime LLM path: embed tokens, run the ViT, scatter image
    embeds, compute 3D position_ids — everything the LLM export wrapper takes."""
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if input_ids is None or attention_mask is None or pixel_values is None:
        raise RuntimeError(
            f"LLM calibration: required inputs missing. Keys found: {list(inputs.keys())}"
        )

    device = next(policy.model.parameters()).device
    qwen = policy.model.backbone.model
    qwen_inner = qwen.model
    vision = qwen_inner.visual
    text_model = qwen_inner.language_model

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device=device)
    image_grid_thw = image_grid_thw.to(device=device)

    inputs_embeds = text_model.get_input_embeddings()(input_ids)

    vision_dtype = next(vision.parameters()).dtype
    vision_out = vision(pixel_values.to(vision_dtype), grid_thw=image_grid_thw)
    if isinstance(vision_out, tuple):
        image_embeds, deepstack_features = vision_out
    else:
        image_embeds = vision_out
        deepstack_features = None

    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_token_id = qwen.config.image_token_id
    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    position_ids, _ = qwen_inner.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )

    batch: dict[str, torch.Tensor] = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask.to(torch.int64),
        "position_ids": position_ids.to(torch.int64),
        "visual_pos_masks": input_ids == image_token_id,
    }
    if deepstack_features is not None:
        for i, feature in enumerate(deepstack_features):
            batch[f"deepstack_{i}"] = feature.to(inputs_embeds.device, inputs_embeds.dtype)
    return batch


def capture_calibration_batches(
    policy: Gr00tPolicy,
    dataset_path: str,
    calib_size: int,
    components: set[str],
) -> dict[str, list[dict[str, torch.Tensor]]]:
    """Capture calibration inputs for the requested components in one pass.

    Runs ``calib_size`` dataset steps through the (still unquantized) policy
    and returns, per component, a list of CPU input dicts matching the export
    wrapper's keyword arguments. Call this before exporting/quantizing any
    component.
    """
    from transformers import BatchFeature

    model = policy.model
    action_head = model.action_head
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=policy.modality_configs,
    )
    step_indices: list[tuple[int, int]] = []
    for ep_idx in range(len(loader)):
        for step_idx in range(len(loader[ep_idx])):
            step_indices.append((ep_idx, step_idx))
            if len(step_indices) >= calib_size:
                break
        if len(step_indices) >= calib_size:
            break
    if not step_indices:
        raise RuntimeError(f"Calibration dataset {dataset_path} yielded no steps")

    def _move(x):
        if isinstance(x, torch.Tensor):
            if x.dtype.is_floating_point:
                return x.to(device=device, dtype=dtype)
            return x.to(device=device)
        return x

    logger.info(
        "Capturing %d calibration batches for %s from %s ...",
        len(step_indices),
        sorted(components),
        dataset_path,
    )
    batches: dict[str, list[dict[str, torch.Tensor]]] = {name: [] for name in components}
    for count, (ep_idx, step_idx) in enumerate(step_indices):
        if count % 10 == 0:
            logger.info("  capturing calibration batch %d/%d ...", count, len(step_indices))
        observation = _prepare_step_observation(loader, ep_idx, step_idx, policy)
        inputs = _collate_inputs(policy, observation)

        pixel_values = inputs.get("pixel_values")
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        with torch.no_grad():
            if "vit" in components:
                if pixel_values is None:
                    raise RuntimeError(
                        "ViT calibration: 'pixel_values' missing from collated inputs. "
                        f"Keys found: {list(inputs.keys())}"
                    )
                batches["vit"].append(_to_cpu({"pixel_values": pixel_values}))

            if "llm" in components:
                llm_inputs = dict(inputs)
                llm_inputs["pixel_values"] = pixel_values
                batches["llm"].append(_to_cpu(_capture_llm_batch(policy, llm_inputs)))

            if "vl_sa" in components or "dit" in components:
                backbone_inputs = {
                    k: _move(inputs[k])
                    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
                    if k in inputs
                }
                backbone_output = model.backbone(BatchFeature(data=backbone_inputs))

                if "vl_sa" in components:
                    hidden_states = action_head.vlln(backbone_output["backbone_features"])
                    vl_inputs = {"hidden_states": hidden_states}
                    if action_head.config.strict_vl_padding_mask:
                        attention_mask = backbone_output.get("backbone_attention_mask")
                        if attention_mask is None:
                            attention_mask = torch.ones(
                                hidden_states.shape[:2],
                                dtype=torch.bool,
                                device=hidden_states.device,
                            )
                        vl_inputs["attention_mask"] = attention_mask.bool()
                    batches["vl_sa"].append(_to_cpu(vl_inputs))

                if "dit" in components:
                    processed = action_head.process_backbone_output(backbone_output)
                    vl_embs = processed["backbone_features"]

                    state = _move(inputs["state"])
                    embodiment_id = _move(inputs["embodiment_id"])
                    state_flat = state.view(state.shape[0], 1, -1)
                    state_features = action_head.state_encoder(state_flat, embodiment_id)

                    B = state_features.shape[0]
                    config = action_head.config
                    actions = torch.randn(
                        B,
                        config.action_horizon,
                        action_head.action_dim,
                        device=device,
                        dtype=state_features.dtype,
                    )
                    timestep = torch.randint(
                        0,
                        int(action_head.num_timestep_buckets),
                        (B,),
                        device=device,
                        dtype=torch.int64,
                    )
                    action_features = action_head.action_encoder(actions, timestep, embodiment_id)
                    if getattr(config, "add_pos_embed", False):
                        pos_ids = torch.arange(
                            action_features.shape[1], dtype=torch.long, device=device
                        )
                        pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                        action_features = action_features + pos_embs

                    batch = {
                        "sa_embs": torch.cat((state_features, action_features), dim=1),
                        "vl_embs": vl_embs,
                        "timestep": timestep,
                    }
                    if processed["image_mask"] is not None:
                        batch["image_mask"] = processed["image_mask"]
                    if processed["backbone_attention_mask"] is not None:
                        batch["backbone_attention_mask"] = processed["backbone_attention_mask"]
                    batches["dit"].append(_to_cpu(batch))

    logger.info("Captured %d calibration batches per component.", len(step_indices))
    return batches


# ============================================================
# Phase 2: modelopt PTQ over the captured batches
# ============================================================

# FFN projection layer names shared by the DiT and VL self-attention
# transformer blocks (diffusers BasicTransformerBlock).
_FFN_LAYERS = ("ff.net.0.proj", "ff.net.2")

# LLM layers kept in FP8 under the NVFP4 recipe to preserve accuracy.
_LLM_NVFP4_FP8_FALLBACK_LAYERS = ("mlp.down_proj", "self_attn.o_proj")

# Batch key and dimension whose export-time shape every calibration batch must
# match (the exported graphs are static except for vl_seq_len).
_SHAPE_KEYS = {
    "vit": ("pixel_values", 0),
    "llm": ("inputs_embeds", 1),
    "vl_sa": ("hidden_states", 1),
    "dit": ("vl_embs", 1),
}


def _quant_config(component: str, tag: str):
    """modelopt quantization config for one (component, precision-tag) recipe."""
    import modelopt.torch.quantization as mtq

    if component == "vit" and tag == "fp8":
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        cfg["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}
        cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}
        return cfg

    if component == "llm" and tag == "fp8":
        return copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    if component == "llm" and tag == "nvfp4":
        cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
        for layer in _LLM_NVFP4_FP8_FALLBACK_LAYERS:
            cfg["quant_cfg"][f"*layers.*.{layer}.input_quantizer"] = {
                "num_bits": (4, 3),
                "axis": None,
            }
            cfg["quant_cfg"][f"*layers.*.{layer}.weight_quantizer"] = {
                "num_bits": (4, 3),
                "axis": None,
            }
        return cfg

    if component == "vl_sa" and tag == "fp8":
        # Standard FP8 tensor format, but only the FFN projections of each
        # block are enabled; attention stays high precision.
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        cfg["quant_cfg"]["*weight_quantizer"] = {"enable": False}
        cfg["quant_cfg"]["*input_quantizer"] = {"enable": False}
        for layer in _FFN_LAYERS:
            for quantizer in ("input_quantizer", "weight_quantizer"):
                cfg["quant_cfg"][f"*transformer_blocks.*.{layer}.{quantizer}"] = copy.deepcopy(
                    mtq.FP8_DEFAULT_CFG["quant_cfg"][f"*{quantizer}"]
                )
        return cfg

    if component == "dit" and tag == "fp8":
        return copy.deepcopy(mtq.FP8_DEFAULT_CFG)

    if component == "dit" and tag == "nvfp4":
        # Attention in FP8, FFN linears in NVFP4.
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        for layer in _FFN_LAYERS:
            for quantizer in ("input_quantizer", "weight_quantizer"):
                cfg["quant_cfg"][f"*transformer_blocks.*.{layer}.{quantizer}"] = copy.deepcopy(
                    mtq.NVFP4_DEFAULT_CFG["quant_cfg"][f"*{quantizer}"]
                )
        return cfg

    raise ValueError(f"No calibration recipe for component={component!r}, tag={tag!r}")


def _count_ffn_linears(wrapper: nn.Module) -> int:
    return sum(
        1
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear) and any(f".{layer}" in name for layer in _FFN_LAYERS)
    )


def _patch_nvfp4_trt_attributes(
    wrapper: nn.Module, target_dtype: torch.dtype, *, only_block_quantized: bool
) -> list[str]:
    """Set the TRT export attributes modelopt's NVFP4 ONNX path expects.

    ``only_block_quantized`` restricts patching to linears whose weight
    quantizer carries NVFP4 block sizes (used for the mixed DiT, where the
    FP8 attention linears must stay untouched).
    """
    from modelopt.torch.quantization.utils import is_quantized_linear

    high_prec = "BFloat16" if target_dtype == torch.bfloat16 else "Half"
    patched = []
    for name, module in wrapper.named_modules():
        if not isinstance(module, nn.Linear) or not is_quantized_linear(module):
            continue
        if only_block_quantized and getattr(module.weight_quantizer, "block_sizes", None) is None:
            continue
        module.input_quantizer._trt_high_precision_dtype = high_prec
        module.input_quantizer._onnx_quantizer_type = "dynamic"
        module.weight_quantizer._onnx_quantizer_type = "static"
        patched.append(name)
    return patched


def _check_effective_samples(used: int, skipped: int, total: int, tag: str) -> None:
    """Log used/skipped counts; raise if effective samples are too few."""
    logger.info(
        f"  {tag} calibration used {used}/{total} samples (skipped {skipped} shape-mismatched)"
    )
    min_required = max(2, total // 2)
    if used < min_required:
        raise RuntimeError(
            f"{tag} calibration only used {used} samples (min {min_required}); "
            f"shape variance in dataset too high — increase --calib-size or filter the dataset."
        )


def _post_calibration_checks(
    wrapper: nn.Module, component: str, tag: str, target_dtype: torch.dtype
) -> None:
    """Recipe-specific TRT attribute patching and selective-quantization audits."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import is_quantized_linear

    if component == "llm" and tag == "nvfp4":
        patched = _patch_nvfp4_trt_attributes(wrapper, target_dtype, only_block_quantized=False)
        logger.info(f"  Patched {len(patched)} quantized LLM linears with TRT NVFP4 attributes")

    if component == "dit" and tag == "nvfp4":
        patched = _patch_nvfp4_trt_attributes(wrapper, target_dtype, only_block_quantized=True)
        expected = _count_ffn_linears(wrapper)
        if len(patched) != expected or any(".ff." not in name for name in patched):
            raise RuntimeError(
                f"DiT mixed quantization expected exactly {expected} NVFP4 FFN Linear "
                f"modules, got {len(patched)}: {patched}"
            )
        logger.info(f"  Patched {len(patched)} DiT NVFP4 FFN linears for TRT export")

    if component == "vl_sa":
        quantized = [
            name
            for name, module in wrapper.named_modules()
            if isinstance(module, nn.Linear)
            and is_quantized_linear(module)
            and module.input_quantizer.is_enabled
            and module.weight_quantizer.is_enabled
        ]
        expected = _count_ffn_linears(wrapper)
        if len(quantized) != expected or any(".ff." not in name for name in quantized):
            raise RuntimeError(
                f"VL-SA selective FP8 expected exactly {expected} FFN Linear modules, "
                f"got {len(quantized)}: {quantized}"
            )
        logger.info(f"  Selectively quantized {len(quantized)} VL-SA FFN Linear modules")

    mtq.print_quant_summary(wrapper)


def calibrate_component(
    wrapper: nn.Module,
    component: str,
    tag: str,
    batches: list[dict[str, torch.Tensor]],
    input_names: Optional[list[str]] = None,
) -> None:
    """Quantize an export wrapper in place with modelopt PTQ.

    ``batches`` come from :func:`capture_calibration_batches`. Batches whose
    shape (see ``_SHAPE_KEYS``) differs from the first batch are skipped: the
    exported graphs are static in those dimensions.
    """
    import modelopt.torch.quantization as mtq

    if not batches:
        raise RuntimeError(f"No calibration batches captured for component {component!r}")

    target_dtype = next(wrapper.parameters()).dtype
    target_device = next(wrapper.parameters()).device
    shape_key, shape_dim = _SHAPE_KEYS[component]
    expected_shape = None
    used = skipped = 0

    def calibrate_loop(model: nn.Module) -> None:
        nonlocal expected_shape, used, skipped
        for idx, batch in enumerate(batches):
            if idx % 10 == 0:
                logger.info(f"    {component} {tag} calibrating batch {idx}...")
            kwargs: dict[str, torch.Tensor] = {}
            for name, tensor in batch.items():
                if input_names is not None and name not in input_names:
                    continue
                if tensor.dtype.is_floating_point:
                    tensor = tensor.to(device=target_device, dtype=target_dtype)
                else:
                    tensor = tensor.to(device=target_device)
                kwargs[name] = tensor

            size = kwargs[shape_key].shape[shape_dim]
            if expected_shape is None:
                expected_shape = size
            elif size != expected_shape:
                logger.warning(
                    f"    skip batch {idx}: {shape_key}.shape[{shape_dim}]={size} "
                    f"!= expected {expected_shape}"
                )
                skipped += 1
                continue

            with torch.inference_mode():
                model(**kwargs)
            used += 1

    logger.info(f"  Running {component} {tag} calibration on {len(batches)} cached batches...")
    start = time.time()
    mtq.quantize(wrapper, _quant_config(component, tag), forward_loop=calibrate_loop)
    logger.info(f"  {component} {tag} calibration finished in {time.time() - start:.1f}s")
    _check_effective_samples(used, skipped, len(batches), f"{component} {tag}")
    _post_calibration_checks(wrapper, component, tag, target_dtype)
    torch.cuda.empty_cache()
