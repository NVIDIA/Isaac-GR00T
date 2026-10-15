# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Install an admitted N2 TensorRT bundle onto a normally loaded model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import types
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from groot.core.policy.env_utils import env_bool
from groot.core.training.models.n2.components.kv_cache import FrozenPrefixKVCache
from groot.core.training.models.n2.components.sequence_packing import (
    MODALITY_ACTION,
    MODALITY_STATE,
    MODALITY_VIDEO,
    MODALITY_VIDEO_COND,
    AttentionMaskPattern,
    PackedSequence,
)
from groot.deployment.n2_trt.conditioner import ActionContextConditioner
from groot.deployment.n2_trt.contract import N2TensorRTContract
from groot.deployment.n2_trt.golden import served_mode_name
from groot.deployment.n2_trt.manifest import RuntimeIdentity
from groot.deployment.n2_trt.runtime import N2TensorRTRuntime, _host_checks_allowed

logger = logging.getLogger(__name__)


class _N2TensorRTPacker:
    """Pack the fixed engine sequence without constructing a Torch attention mask."""

    def __init__(self, runtime: N2TensorRTRuntime):
        self.runtime = runtime

    def pack(
        self,
        video_tokens: torch.Tensor,
        video_timesteps: torch.Tensor,
        video_rope: tuple[torch.Tensor, torch.Tensor],
        video_extra_pos_emb: torch.Tensor | None = None,
        video_cond_token_mask: torch.Tensor | None = None,
        action_tokens: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        state_tokens: torch.Tensor | None = None,
        state_timesteps: torch.Tensor | None = None,
        state_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask_pattern: list[tuple[str, float]] | None = None,
        token_valid_mask: torch.Tensor | None = None,
    ) -> PackedSequence:
        del attention_mask_pattern
        if video_extra_pos_emb is not None:
            raise RuntimeError("N2 TensorRT does not support extra video position embeddings")
        if video_cond_token_mask is None:
            raise RuntimeError("N2 TensorRT requires conditioning-video token IDs")
        if action_tokens is None or action_timesteps is None or action_rope is None:
            raise RuntimeError("N2 TensorRT requires action tokens")
        if state_tokens is None or state_timesteps is None or state_rope is None:
            raise RuntimeError("N2 TensorRT requires state tokens")

        batch = video_tokens.shape[0]
        if batch != 1:
            raise RuntimeError(f"N2 TensorRT requires batch 1, got {batch}")
        segment_counts = (
            video_tokens.shape[1],
            state_tokens.shape[1],
            action_tokens.shape[1],
        )
        self.runtime.validate_segment_layout(segment_counts)
        if _host_checks_allowed():
            contract = self.runtime.contract
            valid_video_tokens = (
                segment_counts[0] // contract.total_camera_slot_count * contract.input_camera_count
            )
            expected_valid = torch.ones(
                (batch, sum(segment_counts)),
                dtype=torch.bool,
                device=video_tokens.device,
            )
            expected_valid[:, valid_video_tokens : segment_counts[0]] = False
            actual_valid = (
                token_valid_mask
                if token_valid_mask is not None
                else torch.ones_like(expected_valid)
            )
            if not torch.equal(actual_valid, expected_valid):
                raise RuntimeError(
                    "N2 TensorRT token padding does not match the exported camera layout"
                )

        device = video_tokens.device
        video_ids = torch.full(
            (batch, segment_counts[0]),
            MODALITY_VIDEO,
            dtype=torch.long,
            device=device,
        )
        video_ids[video_cond_token_mask] = MODALITY_VIDEO_COND
        state_ids = torch.full(
            (batch, segment_counts[1]),
            MODALITY_STATE,
            dtype=torch.long,
            device=device,
        )
        action_ids = torch.full(
            (batch, segment_counts[2]),
            MODALITY_ACTION,
            dtype=torch.long,
            device=device,
        )

        tokens = torch.cat((video_tokens, state_tokens, action_tokens), dim=1)
        timesteps = torch.cat((video_timesteps, state_timesteps, action_timesteps), dim=1)
        rope_freqs = (
            torch.cat((video_rope[0], state_rope[0], action_rope[0]), dim=0),
            torch.cat((video_rope[1], state_rope[1], action_rope[1]), dim=0),
        )
        modality_ids = torch.cat((video_ids, state_ids, action_ids), dim=1)
        video_end = segment_counts[0]
        state_end = video_end + segment_counts[1]
        action_end = state_end + segment_counts[2]
        return PackedSequence(
            tokens=tokens,
            timesteps=timesteps,
            rope_freqs=rope_freqs,
            modality_ids=modality_ids,
            attention_mask=None,
            extra_pos_emb=None,
            pad=0,
            video_token_indices=torch.arange(video_end, device=device),
            state_token_indices=torch.arange(video_end, state_end, device=device),
            action_token_indices=torch.arange(state_end, action_end, device=device),
            non_video_token_indices=torch.arange(video_end, action_end, device=device),
            has_video_cond_tokens=True,
            token_valid_mask=token_valid_mask,
        )


def _require_model_contract(model: Any, contract: N2TensorRTContract) -> None:
    """Reject a bundle resolved for a checkpoint other than the loaded model."""
    failures: dict[str, Any] = {}
    comparisons = {
        "model_type": (getattr(model.config, "model_type", None), contract.model_type),
        "model_size": (getattr(model.config, "model_size", None), contract.model_size),
        "total_camera_slot_count": (
            getattr(model, "max_n_cameras", None),
            contract.total_camera_slot_count,
        ),
        "vae_type": (getattr(model.config, "vae_type", None), contract.vae_type),
        "vae_bottleneck_kernel": (
            _optional_tuple(getattr(model.config, "vae_bottleneck_kernel", None)),
            contract.vae_bottleneck_kernel,
        ),
        "action_horizon": (
            (getattr(model.config, "action_cfg", None) or {}).get("action_horizon"),
            contract.action_horizon,
        ),
        "max_state_dim": (
            (getattr(model.config, "action_cfg", None) or {}).get("max_state_dim"),
            contract.max_state_dim,
        ),
        "max_action_dim": (
            (getattr(model.config, "action_cfg", None) or {}).get("max_action_dim"),
            contract.max_action_dim,
        ),
        "token_merge_kernel": (
            _optional_tuple(getattr(model.config, "token_merge_kernel", None)),
            contract.token_merge_kernel,
        ),
        "token_merge_layers": (
            (
                _optional_tuple(getattr(model.config, "token_merge_layers", None))
                if contract.token_merge_kernel is not None
                else None
            ),
            contract.token_merge_layers,
        ),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            failures[name] = {"loaded": actual, "contract": expected}
    if failures:
        raise RuntimeError(f"loaded N2 model does not match TensorRT contract: {failures}")


def _optional_tuple(value: Any) -> tuple[int, ...] | None:
    return tuple(value) if value is not None else None


def _call_transformer(
    self,
    packed: PackedSequence,
    prompt_embs: torch.Tensor,
    *,
    denoising_step: int = 0,
    is_cfg: bool = False,
    To: int = 1,
    total_seqlen=None,
    last_mini_batch_params_dict=None,
    state_collector=None,
    **transformer_kwargs,
) -> torch.Tensor:
    del denoising_step, is_cfg, To, total_seqlen
    del last_mini_batch_params_dict, state_collector
    runtime = self.n2_trt_runtime
    kv_cache = transformer_kwargs.get("kv_cache")
    frozen_prefix = isinstance(kv_cache, FrozenPrefixKVCache)
    if frozen_prefix and "joint_cond" not in getattr(runtime, "transformer_runners", {}):
        raise RuntimeError(
            "cond_kv_freeze is not served by this TensorRT bundle (no conditioning-cached "
            "joint engine role); disable GROOT_N2_COND_KV_FREEZE or install a bundle that "
            "provides it"
        )
    counts = (
        packed.video_token_indices.numel(),
        packed.state_token_indices.numel(),
        packed.action_token_indices.numel(),
    )
    if counts == (0, 0, runtime.action_horizon):
        mode = "action"
    else:
        runtime.validate_segment_layout(counts)
        mode = "joint_cond" if frozen_prefix else "joint"
    token_merge = transformer_kwargs.get("sequence_transform")
    expected_token_merge = runtime.token_merge_layers is not None
    if mode == "joint" and (token_merge is not None) != expected_token_merge:
        raise RuntimeError("token-merge dispatch does not match the admitted engine bundle")
    input_dtypes = runtime.transformer_input_dtypes(mode)
    engine_packed = replace(
        packed,
        tokens=packed.tokens.to(dtype=input_dtypes["latents"]),
        timesteps=packed.timesteps.to(dtype=input_dtypes["timesteps"]),
        rope_freqs=(
            packed.rope_freqs[0].to(dtype=input_dtypes["rope_cos"]),
            packed.rope_freqs[1].to(dtype=input_dtypes["rope_sin"]),
        ),
    )
    return runtime.run_transformer(
        mode,
        packed=engine_packed,
        prompt_embs=prompt_embs.to(dtype=runtime.prompt_input_dtype(mode)),
        kv_cache=kv_cache,
    )


def _build_action_context_conditioner(
    model: Any, runtime: N2TensorRTRuntime, device: torch.device
) -> ActionContextConditioner:
    """Move only the transformer's cross-attention projection back to the GPU as the conditioner.

    The rest of the eager transformer stays parked on the host; the projection is the
    exact module whose weights the projected-context action engine no longer contains.
    """
    transformer = model.transformer
    if not getattr(transformer.config, "use_crossattn_projection", False):
        raise RuntimeError("projected action context requires a transformer with crossattn_proj")
    projection = transformer.crossattn_proj.to(device=device, dtype=torch.bfloat16)
    cross_attentions = None
    if runtime.action_context_mode == "kv":
        if getattr(transformer, "per_view_text", False):
            raise RuntimeError(
                "kv action context requires a single text copy (per_view_text=False)"
            )
        cross_attentions = []
        for block in transformer.transformer_blocks:
            attention = block.attn2
            # Only the key/value side returns to the GPU; the query and output
            # projections stay inside the action engine (and parked on the host here).
            attention.to_k.to(device=device, dtype=torch.bfloat16)
            attention.to_v.to(device=device, dtype=torch.bfloat16)
            attention.norm_k.to(device=device, dtype=torch.bfloat16)
            cross_attentions.append(attention)
    return ActionContextConditioner(
        mode=runtime.action_context_mode,
        projection=projection,
        device=device,
        engine_inputs=runtime.transformer_runners["action"].inputs,
        cross_attentions=cross_attentions,
    )


def _encode_video(self, videos: torch.Tensor) -> torch.Tensor:
    """Encode the fixed multiview history with the admitted VAE engine."""
    contract = self.n2_trt_runtime.contract
    expected = (
        1,
        3,
        contract.total_camera_slot_count * contract.conditioning_frames,
        contract.image_height,
        contract.image_width,
    )
    if tuple(videos.shape) != expected:
        raise RuntimeError(f"N2 TensorRT video shape {tuple(videos.shape)} != {expected}")
    per_view = videos.reshape(
        1,
        3,
        contract.total_camera_slot_count,
        contract.conditioning_frames,
        contract.image_height,
        contract.image_width,
    )
    per_view = per_view.permute(0, 2, 1, 3, 4, 5).reshape(
        contract.total_camera_slot_count,
        3,
        contract.conditioning_frames,
        contract.image_height,
        contract.image_width,
    )
    # Production preprocessing emits FP32. The eager VAE performs this same
    # conversion to its parameter dtype inside GrootN2.encode_video; preserve
    # that seam when replacing the encoder with the fixed BF16 engine.
    per_view = per_view.to(dtype=torch.bfloat16)
    latents = self.n2_trt_runtime.encode_video(per_view, tiled=False)
    if latents.dim() != 5 or latents.shape[0] != contract.total_camera_slot_count:
        raise RuntimeError(
            f"N2 TensorRT VAE returned {tuple(latents.shape)}, expected "
            f"({contract.total_camera_slot_count}, C, T, H, W)"
        )
    channels, frames, height, width = latents.shape[1:]
    return latents.permute(1, 0, 2, 3, 4).reshape(
        1,
        channels,
        contract.total_camera_slot_count * frames,
        height,
        width,
    )


def _without_negative_text(inputs: dict | BatchFeature) -> dict | BatchFeature:
    outer = dict(inputs if isinstance(inputs, dict) else inputs.data)
    if "inputs" in outer:
        payload = dict(outer["inputs"])
        payload.pop("text_negative", None)
        outer["inputs"] = payload
    else:
        outer.pop("text_negative", None)
    return outer if isinstance(inputs, dict) else BatchFeature(data=outer)


def _text_encoder_offload_decision(
    device: torch.device, encoder: torch.nn.Module
) -> dict[str, Any]:
    name = "GROOT_N2_TRT_OFFLOAD_TEXT_ENCODER"
    override = env_bool(name, default=False) if name in os.environ else None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    weight_bytes = sum(t.numel() * t.element_size() for t in encoder.parameters())
    weight_bytes += sum(t.numel() * t.element_size() for t in encoder.buffers())
    # Called after TRT allocation. Reserve space for encoder activations, graph
    # pools, and the colocated client, not just the resident weights.
    reserve_bytes = 8 << 30
    offload = override if override is not None else free_bytes < weight_bytes + reserve_bytes
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "encoder_weight_bytes": weight_bytes,
        "reserve_bytes": reserve_bytes,
        "offload": offload,
        "mode": "module_cpu_offload" if offload else "gpu_resident",
        "selection": "environment" if override is not None else "auto",
    }


def install_n2_tensorrt(
    model: Any,
    engine_dir: str | Path,
    *,
    contract: N2TensorRTContract,
    runtime_identity: RuntimeIdentity | None = None,
    runtime_factory: Callable[[], N2TensorRTRuntime] | None = None,
    allow_quantized: bool = False,
    allow_amplitude_only: bool = False,
    allow_unadmitted_mode: bool = False,
    expected_codec: str | None = None,
    served_cond_kv_freeze: bool | None = None,
) -> N2TensorRTRuntime:
    """Replace only one normal model instance's fixed inference collaborators.

    ``runtime_factory`` constructs the runtime at the same point ``from_directory``
    would (after the eager collaborators are parked, so device memory peaks the
    same way) for engines that have no admitted manifest yet; every other
    installation step is unchanged. ``allow_quantized`` is the opt-in for bundles
    or runtimes whose transformer engines are quantized; ``allow_amplitude_only`` is the
    opt-in for a golden package certified at the amplitude level only; ``allow_unadmitted_mode`` is
    the opt-in for a served mode the platform row does not admit. ``served_cond_kv_freeze``
    selects the served conditioning-cache mode (``None`` = the bundle's default: on
    exactly when it carries the conditioning-cached joint engine).
    """
    if getattr(model, "n2_trt_runtime", None) is not None:
        raise RuntimeError("N2 TensorRT is already installed on this model")
    contract.validate()
    _require_model_contract(model, contract)
    if model.tiled:
        raise RuntimeError("N2 TensorRT does not support tiled VAE execution")
    expected_pattern = AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_FULL
    actual_pattern = model._resolve_attention_mask_pattern()  # noqa: SLF001
    if actual_pattern != expected_pattern:
        raise RuntimeError(
            f"N2 TensorRT attention pattern {actual_pattern.value} != {expected_pattern.value}"
        )
    runtime_device = model.device
    if runtime_device.type != "cuda":
        raise RuntimeError(f"N2 TensorRT requires a CUDA model, got {runtime_device}")

    model.transformer.to("cpu")
    model.vae.to("cpu")
    model.text_encoder.to("cpu")
    with torch.cuda.device(runtime_device):
        torch.cuda.empty_cache()
        if runtime_factory is None:
            runtime = N2TensorRTRuntime.from_directory(
                engine_dir,
                contract=contract,
                runtime_identity=runtime_identity,
                allow_quantized=allow_quantized,
                allow_amplitude_only=allow_amplitude_only,
                allow_unadmitted_mode=allow_unadmitted_mode,
                expected_codec=expected_codec,
                served_cond_kv_freeze=served_cond_kv_freeze,
            )
        else:
            runtime = runtime_factory()
            expected = (
                contract.with_transformer_precisions(
                    runtime.contract.transformer_precision,
                    runtime.contract.action_cache_precision,
                )
                .with_vae_precision(runtime.contract.vae_precision)
                .with_cond_kv_freeze(runtime.contract.cond_kv_freeze)
            )
            if runtime.contract.to_dict() != expected.to_dict():
                raise RuntimeError(
                    "injected N2 TensorRT runtime contract does not match the loaded model's checkpoint contract"
                )
            if served_cond_kv_freeze is not None and (
                getattr(runtime, "served_cond_kv_freeze", runtime.contract.cond_kv_freeze)
                != served_cond_kv_freeze
            ):
                raise RuntimeError(
                    "injected N2 TensorRT runtime does not serve the selected cond_kv_freeze mode "
                    f"(runtime={getattr(runtime, 'served_cond_kv_freeze', None)}, "
                    f"selected={served_cond_kv_freeze})"
                )
            if runtime.contract.is_quantized and not allow_quantized:
                raise RuntimeError(
                    "the injected runtime serves quantized engines; "
                    "quantized serving is opt-in (allow_quantized)"
                )
            if (
                getattr(getattr(runtime, "manifest", None), "certification_level", None)
                == "amplitude"
                and not allow_amplitude_only
            ):
                raise RuntimeError(
                    "the injected runtime serves a golden package certified at the amplitude "
                    "level only; serving it is opt-in (allow_amplitude_only)"
                )
            admitted = getattr(runtime, "admitted_served_modes", None)
            if admitted is not None and not allow_unadmitted_mode:
                served = bool(
                    getattr(runtime, "served_cond_kv_freeze", runtime.contract.cond_kv_freeze)
                )
                if served_mode_name(served) not in admitted:
                    raise RuntimeError(
                        f"the injected runtime serves {served_mode_name(served)!r}, a mode its "
                        f"platform row does not admit (admitted: {list(admitted)}); serving it is "
                        "opt-in (allow_unadmitted_mode)"
                    )
    if getattr(runtime, "action_context_mode", "raw") != "raw":
        runtime.attach_conditioner(
            _build_action_context_conditioner(model, runtime, runtime_device)
        )
        logger.info("N2 action context conditioner: %s", runtime.action_context_mode)

    from groot.deployment.n2_trt.prompt_encoder import CachedPromptEncoder

    decision = _text_encoder_offload_decision(runtime_device, model.text_encoder)
    model.n2_prompt_encoder = CachedPromptEncoder(
        model.text_encoder, runtime_device, offload=decision["offload"]
    )
    model.n2_text_encoder_diagnostics = decision
    model.encode_text = model.n2_prompt_encoder
    logger.info("N2 text encoding (GPU compute): %s", json.dumps(decision, sort_keys=True))

    model.n2_trt_runtime = runtime
    model.packer = _N2TensorRTPacker(runtime)
    model._call_transformer = types.MethodType(_call_transformer, model)  # noqa: SLF001
    model.encode_video = types.MethodType(_encode_video, model)

    original_predict = model.predict

    @torch.no_grad()
    def predict_with_contract(
        inputs: dict | BatchFeature,
        *,
        num_video_steps: int = 36,
        num_action_steps: int = 4,
        video_guidance_scale: float = 1.0,
        action_guidance_scale: float = 1.0,
        shift: float = 1.0,
        seed: int = 42,
        predict_video: bool = False,
        joint_denoise: bool = True,
        prompt_embs: torch.Tensor | None = None,
        kv_cache_cond: bool = False,
        cond_kv_freeze: bool = False,
        step_callback=None,
    ) -> dict[str, Any]:
        data = inputs if isinstance(inputs, dict) else inputs.data
        if "inputs" in data:
            data = data["inputs"]
        runtime.validate_runtime(
            batch_size=data["images"].shape[0],
            num_cameras=model.max_n_cameras,
            num_video_steps=num_video_steps,
            num_action_steps=num_action_steps,
            use_kv_cache=kv_cache_cond,
            cfg_on=video_guidance_scale != 1.0 or action_guidance_scale != 1.0,
            attention_pattern=model._resolve_attention_mask_pattern(),  # noqa: SLF001
            video_padded_mask=data.get("video_padded_mask"),
            cond_kv_freeze=cond_kv_freeze,
        )
        if shift != 5.0 or seed != 42 or not joint_denoise:
            raise RuntimeError("N2 TensorRT requires shift=5, seed=42, and joint denoising")
        if predict_video or step_callback is not None:
            raise RuntimeError("N2 TensorRT does not support decoded-video callbacks")
        return original_predict(
            _without_negative_text(inputs),
            num_video_steps=num_video_steps,
            num_action_steps=num_action_steps,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
            shift=shift,
            seed=seed,
            predict_video=False,
            joint_denoise=True,
            prompt_embs=prompt_embs,
            kv_cache_cond=kv_cache_cond,
            cond_kv_freeze=cond_kv_freeze,
            step_callback=None,
        )

    model.predict = predict_with_contract

    logger.info(
        "Installed %s from %s",
        runtime.runtime_contract,
        Path(engine_dir).expanduser().resolve(),
    )
    return runtime
