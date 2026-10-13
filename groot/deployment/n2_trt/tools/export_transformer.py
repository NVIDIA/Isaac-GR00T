#!/usr/bin/env python3
# ruff: noqa: SLF001, T201
"""Export the admitted N2 joint/action transformer graphs without quantization."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import transformers

from groot.core.training.models.n2.components.flash_attention import (
    scaled_dot_product_attention_bf16,
)
from groot.core.training.models.n2.utils.cosmos_model_deps import ensure_cosmos_model_deps
from groot.deployment.n2_trt.contract import N2TensorRTContract, TransformerPrecision
from groot.deployment.n2_trt.tools.conditioning import export_conditioning

DEFAULT_EMBODIMENT = "xdof_relative_eef_relative_joint"
DEFAULT_CODEC_ROUNDTRIP = "mp4v"


def _prepare_checkpoint(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    return Path(ensure_cosmos_model_deps(checkpoint_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--embodiment",
        default=DEFAULT_EMBODIMENT,
        help="Lowercase YAM XDOF embodiment tag present in the checkpoint processor.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC_ROUNDTRIP,
        choices=("h264", "mp4v", "none"),
        help="Observation codec roundtrip used only for the deterministic export probe.",
    )
    parser.add_argument("--num-video-steps", type=int, default=2)
    parser.add_argument("--num-action-steps", type=int, default=4)
    args = parser.parse_args()
    # Other export options retain their production defaults.
    args.mode = "production"
    args.embodiment_tag = args.embodiment
    args.policy_embodiment_tag = args.embodiment.upper()
    args.video_mode = "gradient"
    args.language = "pick up the object"
    args.opset = 20
    args.shift = 5.0
    args.seed = 42
    args.save_samples = True
    args.joint_mask_export = "segmented-sdpa"
    return args


def _resolve_routing_indices(
    modality_ids: torch.Tensor,
    video_token_indices: torch.Tensor | None = None,
    non_video_token_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve static routing indices before ONNX tracing starts."""
    if modality_ids.ndim != 2:
        raise ValueError(
            f"modality_ids must have shape (batch, sequence); got {tuple(modality_ids.shape)}"
        )
    if not torch.equal(modality_ids, modality_ids[:1].expand_as(modality_ids)):
        raise ValueError("ONNX export requires one shared modality layout per batch")

    ids = modality_ids[0]
    if video_token_indices is None:
        video_token_indices = torch.where((ids == 0) | (ids == 1))[0]
    if non_video_token_indices is None:
        non_video_token_indices = torch.where((ids != 0) & (ids != 1))[0]
    return (
        video_token_indices.to(device=modality_ids.device, dtype=torch.long),
        non_video_token_indices.to(device=modality_ids.device, dtype=torch.long),
    )


class _N2TransformerExportAdapterBase(torch.nn.Module):
    """Call the compile-friendly transformer entry point with static routing."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge: Any = None,
    ):
        super().__init__()
        self.transformer = transformer
        # Keep the original implementation while routing through this adapter.
        self._routing_forward = transformer._forward_with_routing_indices.__func__
        self.token_merge = token_merge
        self.register_buffer("video_token_indices", video_token_indices)
        self.register_buffer("non_video_token_indices", non_video_token_indices)

    def _forward_transformer(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        *,
        extra_pos_emb: torch.Tensor | None,
        attention_mask: Any,
        pad: int,
        kv_cache: Any,
        store_kv: bool,
    ) -> torch.Tensor:
        token_merge_kwargs = (
            {"sequence_transform": self.token_merge} if self.token_merge is not None else {}
        )
        return self._routing_forward(
            self.transformer,
            latents,
            timesteps,
            (rope_cos, rope_sin),
            context,
            modality_ids,
            extra_pos_emb=extra_pos_emb,
            attention_mask=attention_mask,
            pad=pad,
            kv_cache=kv_cache,
            store_kv=store_kv,
            video_token_indices=self.video_token_indices,
            non_video_token_indices=self.non_video_token_indices,
            **token_merge_kwargs,
        )


class N2TransformerJointCaptureExportAdapter(_N2TransformerExportAdapterBase):
    """Production joint transformer call that returns output tokens and captured KV cache."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        attention_mask: Any,
        pad: int,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge: Any = None,
    ):
        super().__init__(
            transformer,
            video_token_indices,
            non_video_token_indices,
            token_merge=token_merge,
        )
        self.attention_mask = attention_mask
        self.pad = pad

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        latents = latents.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        context = context.to(torch.bfloat16)
        cache = CondKVCache()
        output = self._forward_transformer(
            latents,
            timesteps,
            rope_cos,
            rope_sin,
            context,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=self.attention_mask,
            pad=self.pad,
            kv_cache=cache,
            store_kv=True,
        )
        if self.token_merge is None:
            return (
                output,
                torch.stack(cache.key_cache, dim=0),
                torch.stack(cache.value_cache, dim=0),
            )
        from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

        return (
            output,
            *_stack_token_merge_cache(
                cache,
                (self.token_merge.start_layer, self.token_merge.end_layer),
            ),
        )


class N2TransformerActionCachedExportAdapter(_N2TransformerExportAdapterBase):
    """Production action-only transformer call with sliced KV cache input."""

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key: torch.Tensor,
        kv_cache_value: torch.Tensor,
    ) -> torch.Tensor:
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        latents = latents.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        context = context.to(torch.bfloat16)
        kv_cache_key = kv_cache_key.to(torch.bfloat16)
        kv_cache_value = kv_cache_value.to(torch.bfloat16)
        cache = CondKVCache(
            key_cache=list(kv_cache_key.unbind(0)),
            value_cache=list(kv_cache_value.unbind(0)),
        )
        return self._forward_transformer(
            latents,
            timesteps,
            rope_cos,
            rope_sin,
            context,
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
        )


class N2TransformerTokenMergeActionCachedExportAdapter(_N2TransformerExportAdapterBase):
    """Action-only transformer call with heterogeneous fine/coarse KV inputs."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        video_token_indices: torch.Tensor,
        non_video_token_indices: torch.Tensor,
        token_merge_layers: tuple[int, int],
    ):
        super().__init__(transformer, video_token_indices, non_video_token_indices)
        self.token_merge_layers = token_merge_layers

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        context: torch.Tensor,
        modality_ids: torch.Tensor,
        kv_cache_key_fine: torch.Tensor,
        kv_cache_value_fine: torch.Tensor,
        kv_cache_key_coarse: torch.Tensor,
        kv_cache_value_coarse: torch.Tensor,
    ) -> torch.Tensor:
        from groot.deployment.n2_trt.runtime import _restore_token_merge_cache

        cache = _restore_token_merge_cache(
            kv_cache_key_fine.to(torch.bfloat16),
            kv_cache_value_fine.to(torch.bfloat16),
            kv_cache_key_coarse.to(torch.bfloat16),
            kv_cache_value_coarse.to(torch.bfloat16),
            self.token_merge_layers,
        )
        return self._forward_transformer(
            latents.to(torch.bfloat16),
            timesteps.to(torch.bfloat16),
            rope_cos,
            rope_sin,
            context.to(torch.bfloat16),
            modality_ids,
            extra_pos_emb=None,
            attention_mask=None,
            pad=0,
            kv_cache=cache,
            store_kv=False,
        )


def _dense_token_mask_from_block_mask(block_mask: Any) -> torch.Tensor:
    device = block_mask.kv_num_blocks.device
    q_len, kv_len = block_mask.seq_lengths
    batch = int(block_mask.kv_num_blocks.shape[0])
    b = torch.arange(batch, device=device)[:, None, None]
    h = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    q_idx = torch.arange(q_len, device=device)[None, :, None]
    kv_idx = torch.arange(kv_len, device=device)[None, None, :]
    return block_mask.mask_mod(b, h, q_idx, kv_idx)[:, None, :, :].to(torch.bool)


@dataclass(frozen=True)
class _BidirectionalVideoActionFullSegments:
    """Contiguous token ranges for the one segmented attention pattern we export."""

    video_tokens: int
    state_tokens: int
    action_tokens: int
    alignment_pad: int

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + self.state_tokens + self.action_tokens

    def to_manifest(self) -> dict[str, int | str]:
        return {
            "pattern": "bidirectional_video_action_full",
            "video_tokens": self.video_tokens,
            "state_tokens": self.state_tokens,
            "action_tokens": self.action_tokens,
            "total_tokens": self.total_tokens,
            "alignment_pad": self.alignment_pad,
        }


def _model_latent_shape(
    model: Any,
    *,
    conditioning_frames: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Per-camera VAE latent (C, T, H, W) for this checkpoint, measured not assumed.

    Base N2 (/8 into 16ch) and the Wan2.2 f16t4c48 VAE (turbo) disagree here, and the token
    count downstream follows it. Derived from the VAE directly rather than from the
    captured sequence, so the contract check stays a real check.
    """
    probe = torch.zeros(
        (1, 3, model.max_n_cameras * conditioning_frames, *image_size),
        device=model.device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        latents = model.encode_video(probe)
    channels, all_frames, height, width = latents.shape[1:]
    return (channels, all_frames // model.max_n_cameras, height, width)


def _require_segment_counts(
    segments: _BidirectionalVideoActionFullSegments,
    *,
    num_cameras: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int],
    total_latent_frames: int,
) -> None:
    from groot.deployment.n2_trt.runtime import n2_segment_token_counts

    actual = (segments.video_tokens, segments.state_tokens, segments.action_tokens)
    expected = n2_segment_token_counts(
        num_cameras,
        action_horizon,
        latent_shape,
        total_latent_frames,
    )
    if actual != expected:
        raise ValueError(
            f"segmented-sdpa N2 runtime contract requires exact token counts {expected}; got {actual}"
        )


def _validate_bidirectional_video_action_full_segments(
    modality_ids: torch.Tensor,
    dense_mask: torch.Tensor,
) -> _BidirectionalVideoActionFullSegments:
    """Prove a captured mask is exactly three unmasked, contiguous SDPA segments."""
    if modality_ids.ndim != 2:
        raise ValueError(
            f"segmented-sdpa requires modality_ids with shape (batch, sequence); got {tuple(modality_ids.shape)}"
        )
    if dense_mask.ndim != 4 or dense_mask.shape[1] != 1 or dense_mask.dtype != torch.bool:
        raise ValueError(
            "segmented-sdpa requires a boolean mask with shape (batch, 1, query, key); "
            f"got shape={tuple(dense_mask.shape)} dtype={dense_mask.dtype}"
        )

    batch, sequence = modality_ids.shape
    if (
        dense_mask.shape[0] != batch
        or dense_mask.shape[2] != dense_mask.shape[3]
        or dense_mask.shape[2] < sequence
    ):
        raise ValueError(
            "segmented-sdpa mask shape does not cover the captured modality layout: "
            f"mask={tuple(dense_mask.shape)} modality_ids={tuple(modality_ids.shape)}"
        )
    if not torch.equal(modality_ids, modality_ids[0:1].expand_as(modality_ids)):
        raise ValueError("segmented-sdpa requires the same modality layout for every batch item")

    ids = modality_ids[0]
    is_video_cond = ids == 1
    is_video_denoise = ids == 0
    is_video = is_video_cond | is_video_denoise
    is_state = ids == 2
    is_action = ids == 3
    if not torch.all(is_video | is_state | is_action).item():
        raise ValueError("segmented-sdpa encountered an unsupported modality ID")
    if not torch.any(is_video_cond).item() or not torch.any(is_video_denoise).item():
        raise ValueError("segmented-sdpa requires both conditioning and denoising video tokens")

    video_tokens = int(is_video.sum().item())
    state_tokens = int(is_state.sum().item())
    action_tokens = int(is_action.sum().item())
    if min(video_tokens, state_tokens, action_tokens) <= 0:
        raise ValueError("segmented-sdpa requires non-empty video, state, and action segments")

    state_end = video_tokens + state_tokens
    if not (
        torch.all(is_video[:video_tokens]).item()
        and torch.all(is_state[video_tokens:state_end]).item()
        and torch.all(is_action[state_end:]).item()
    ):
        raise ValueError(
            "segmented-sdpa requires contiguous [video | state | action] modality segments"
        )

    expected_mask = torch.zeros_like(dense_mask)
    expected_mask[:, :, :video_tokens, :video_tokens] = True
    expected_mask[:, :, video_tokens:state_end, video_tokens:state_end] = True
    expected_mask[:, :, state_end:sequence, :sequence] = True
    if not torch.equal(dense_mask, expected_mask):
        raise ValueError(
            "segmented-sdpa only supports bidirectional_video_action_full with all tokens valid"
        )

    return _BidirectionalVideoActionFullSegments(
        video_tokens=video_tokens,
        state_tokens=state_tokens,
        action_tokens=action_tokens,
        alignment_pad=int(dense_mask.shape[2] - sequence),
    )


def _segmented_bidirectional_video_action_full_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segments: _BidirectionalVideoActionFullSegments,
    flatten_heads: bool,
) -> torch.Tensor:
    """Exact unmasked-SDPA decomposition of ``bidirectional_video_action_full``."""
    batch, sequence, heads, head_dim = q.shape
    if sequence != segments.total_tokens or k.shape[1] != sequence or v.shape[1] != sequence:
        raise ValueError(
            "segmented-sdpa input sequence does not match the validated export layout: "
            f"q={sequence}, k={k.shape[1]}, v={v.shape[1]}, expected={segments.total_tokens}"
        )

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    video_end = segments.video_tokens
    state_end = video_end + segments.state_tokens

    video_out = scaled_dot_product_attention_bf16(
        q[:, :, :video_end], k[:, :, :video_end], v[:, :, :video_end]
    )
    state_out = scaled_dot_product_attention_bf16(
        q[:, :, video_end:state_end],
        k[:, :, video_end:state_end],
        v[:, :, video_end:state_end],
    )
    action_out = scaled_dot_product_attention_bf16(q[:, :, state_end:], k, v)
    out = torch.cat((video_out, state_out, action_out), dim=2)
    out = out.permute(0, 2, 1, 3).contiguous()
    if flatten_heads:
        return out.reshape(batch, sequence, heads * head_dim)
    return out


def _install_segmented_sdpa_flex_patch(
    modality_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    *,
    num_cameras: int,
    action_horizon: int,
    latent_shape: tuple[int, int, int, int],
    total_latent_frames: int,
    token_merge: Any = None,
) -> tuple[
    Any,
    _BidirectionalVideoActionFullSegments,
    _BidirectionalVideoActionFullSegments | None,
]:
    """Install the exact YAM segmented exporter after validating its camera contract."""
    from groot.core.training.models.n2.components import transformer_cosmos

    segments = _validate_bidirectional_video_action_full_segments(modality_ids, dense_mask)
    _require_segment_counts(
        segments,
        num_cameras=num_cameras,
        action_horizon=action_horizon,
        latent_shape=latent_shape,
        total_latent_frames=total_latent_frames,
    )
    coarse_segments = None
    if token_merge is not None:
        if token_merge.reduced_attention_mask is None:
            raise ValueError("segmented-sdpa token merge requires a coarse attention mask")
        coarse_modality_ids = modality_ids[:, token_merge.reduced_token_indices]
        coarse_dense_mask = _dense_token_mask_from_block_mask(token_merge.reduced_attention_mask)
        coarse_segments = _validate_bidirectional_video_action_full_segments(
            coarse_modality_ids, coarse_dense_mask
        )
        expected_coarse = (
            segments.video_tokens // (token_merge.kernel_height * token_merge.kernel_width),
            segments.state_tokens,
            segments.action_tokens,
        )
        actual_coarse = (
            coarse_segments.video_tokens,
            coarse_segments.state_tokens,
            coarse_segments.action_tokens,
        )
        if actual_coarse != expected_coarse:
            raise ValueError(
                "segmented-sdpa token-merge layout requires exact coarse token counts "
                f"{expected_coarse}; got {actual_coarse}"
            )
    original = transformer_cosmos.flex_attention_op

    def segmented_sdpa_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Any = None,
        pad: int = 0,
        flatten_heads: bool = True,
        compile_attention: bool = False,
    ) -> torch.Tensor:
        del compile_attention  # Export always uses the explicit segmented SDPA path.
        if attn_mask is None:
            raise ValueError("segmented-sdpa is only valid for a masked joint transformer call")
        sequence = q.shape[1]
        if sequence == segments.total_tokens:
            active_segments = segments
        elif coarse_segments is not None and sequence == coarse_segments.total_tokens:
            active_segments = coarse_segments
        else:
            supported = [segments.total_tokens]
            if coarse_segments is not None:
                supported.append(coarse_segments.total_tokens)
            raise ValueError(
                f"segmented-sdpa sequence {sequence} is not one of the exported layouts {supported}"
            )
        if pad != active_segments.alignment_pad:
            raise ValueError(
                "segmented-sdpa alignment pad differs from the validated export layout: "
                f"{pad} != {active_segments.alignment_pad}"
            )
        return _segmented_bidirectional_video_action_full_sdpa(
            q,
            k,
            v,
            segments=active_segments,
            flatten_heads=flatten_heads,
        )

    transformer_cosmos.flex_attention_op = segmented_sdpa_attention_op
    return original, segments, coarse_segments


def _restore_flex_patch(original: Any) -> None:
    if original is None:
        return
    from groot.core.training.models.n2.components import transformer_cosmos

    transformer_cosmos.flex_attention_op = original


def _tensor_record(tensor: Any) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _production_eager_oracle() -> str:
    from groot.deployment.n2_trt.manifest import TRANSFORMER_EAGER_ORACLE

    if os.environ.get("ENABLE_TENSORRT", "False").lower() == "true":
        raise RuntimeError(
            "refusing to record a transformer parity oracle through the TensorRT export surrogate"
        )
    return TRANSFORMER_EAGER_ORACLE


def _save_capture(path: Path, payload: dict[str, Any]) -> None:
    packed = payload["packed"]
    cache = payload.get("kv_cache")
    output_name = payload.get("output_name", "output")
    if output_name not in {"output", "output1"}:
        raise ValueError(f"unsupported production capture output name: {output_name!r}")
    sample = {
        "oracle": _production_eager_oracle(),
        "latents": packed.tokens.detach().cpu(),
        "timesteps": packed.timesteps.detach().cpu(),
        "rope_cos": packed.rope_freqs[0].detach().cpu(),
        "rope_sin": packed.rope_freqs[1].detach().cpu(),
        "context": payload["prompt_embs"].detach().cpu(),
        "modality_ids": packed.modality_ids.detach().cpu(),
        "video_token_indices": getattr(packed, "video_token_indices", None),
        "non_video_token_indices": getattr(packed, "non_video_token_indices", None),
        output_name: payload["output"].detach().cpu(),
        "pad": int(packed.pad),
    }
    for name in ("video_token_indices", "non_video_token_indices"):
        if isinstance(sample[name], torch.Tensor):
            sample[name] = sample[name].detach().cpu()
    if cache is not None and getattr(cache, "key_cache", None):
        token_merge_layers = payload.get("token_merge_layers")
        if token_merge_layers is None:
            sample["kv_cache_key"] = torch.stack(cache.key_cache).detach().cpu()
            sample["kv_cache_value"] = torch.stack(cache.value_cache).detach().cpu()
        else:
            from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

            cache_tensors = _stack_token_merge_cache(cache, token_merge_layers)
            for name, tensor in zip(
                (
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ),
                cache_tensors,
                strict=True,
            ):
                sample[name] = tensor.detach().cpu()
        token_merge = payload.get("token_merge")
        if token_merge is not None:
            sample["coarse_token_idx"] = token_merge.reduced_token_indices.detach().cpu()
    torch.save(sample, path)


def _install_call_probe(
    model: torch.nn.Module,
    sample_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Capture the first joint/action call for ONNX export."""
    captured: dict[str, dict[str, Any]] = {}
    sample_paths: dict[str, Path] = {}
    original = model._call_transformer
    model._conditioning_probe_calls = []
    token_merge_layers = (
        tuple(model.config.token_merge_layers)
        if model.config.token_merge_kernel is not None
        else None
    )

    def wrapped_call(
        packed: Any,
        prompt_embs: torch.Tensor,
        *,
        denoising_step: int = 0,
        is_cfg: bool = False,
        To: int = 1,
        total_seqlen: Any = None,
        last_mini_batch_params_dict: Any = None,
        state_collector: Any = None,
        **transformer_kwargs: Any,
    ) -> torch.Tensor:
        if packed.attention_mask is not None and transformer_kwargs.get("kv_cache") is None:
            from groot.core.training.models.n2.components.kv_cache import CondKVCache

            transformer_kwargs["kv_cache"] = CondKVCache()
        output = original(
            packed,
            prompt_embs,
            denoising_step=denoising_step,
            is_cfg=is_cfg,
            To=To,
            total_seqlen=total_seqlen,
            last_mini_batch_params_dict=last_mini_batch_params_dict,
            state_collector=state_collector,
            **transformer_kwargs,
        )
        cache = transformer_kwargs.get("kv_cache")
        store_kv = transformer_kwargs.get("store_kv", True)
        mode = None
        if packed.attention_mask is not None and cache is not None and store_kv:
            mode = "joint_full_capture"
        elif cache is not None and not store_kv:
            mode = "action_cached"
        if mode is not None:
            sigmas = packed.timesteps.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            if not torch.isfinite(sigmas).all():
                raise ValueError("production probe supplied nonfinite conditioning sigmas")
            model._conditioning_probe_calls.append(
                {
                    "mode": mode,
                    "denoising_step": denoising_step,
                    "shape": list(sigmas.shape),
                    "sigmas": torch.unique(sigmas),
                }
            )
        if mode is not None and mode not in captured:
            payload = {
                "packed": packed,
                "prompt_embs": prompt_embs,
                "output": output,
                "output_name": "output" if mode == "joint_full_capture" else "output1",
                "kv_cache": cache,
                # The merged model API threads the canonical TokenMergePlan as
                # ``sequence_transform``.  Retain the historical artifact key
                # consumed by the TensorRT exporter/runtime, but observe it at
                # the production call boundary rather than rebuilding a plan.
                "token_merge": transformer_kwargs.get("sequence_transform"),
                "token_merge_layers": token_merge_layers,
            }
            captured[mode] = payload
            path = sample_dir / f"{mode}.pt"
            _save_capture(path, payload)
            sample_paths[mode] = path
        return output

    model._call_transformer = wrapped_call
    return captured, sample_paths


# ── synthetic probe observation ─────────────────────────────────────────────
# Checkpoint-shaped input used only to capture the fixed export graph.

DEFAULT_EMBODIMENT = "xdof_relative_eef_relative_joint"
LANGUAGE = "pick up the object"


def _load_metadata(
    checkpoint: Path,
    embodiment: str = DEFAULT_EMBODIMENT,
) -> tuple[dict[str, Any], dict[str, Any], N2TensorRTContract]:
    """Read the checkpoint workload and check it is exportable."""
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    statistics_config = json.loads((checkpoint / "statistics.json").read_text())
    kwargs = processor["processor_kwargs"]
    configs = kwargs["modality_configs"]
    if embodiment not in configs or embodiment not in statistics_config:
        raise ValueError(f"Checkpoint does not contain the {embodiment!r} workload")

    config = configs[embodiment]
    contract = N2TensorRTContract.from_checkpoint(
        checkpoint,
        embodiment=embodiment,
        video_denoising_steps=2,
        action_denoising_steps=4,
        joint_denoise=True,
        conditional_kv_cache=True,
    )
    action_deltas = tuple(config["action"]["delta_indices"])
    checks = {
        "state deltas": (tuple(config["state"]["delta_indices"]), (0,)),
        "action horizon": (action_deltas, tuple(range(len(action_deltas)))),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                "unsupported N2 TensorRT contract (not supported yet): "
                f"{name} must be {expected}, got {actual}"
            )
    return config, statistics_config[embodiment], contract


def _gradient_video(shape: tuple[int, int, int, int]) -> np.ndarray:
    frames, height, width, _channels = shape
    x = np.linspace(0, 255, width, dtype=np.uint16)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint16)[:, None]
    video = np.empty(shape, dtype=np.uint8)
    for index in range(frames):
        video[index, :, :, 0] = ((x + index * 7) % 256).astype(np.uint8)
        video[index, :, :, 1] = ((y + index * 13) % 256).astype(np.uint8)
        video[index, :, :, 2] = ((x // 2 + y // 2 + index * 17) % 256).astype(np.uint8)
    return video


def make_synthetic_observation(
    checkpoint: Path,
    embodiment_tag: str = DEFAULT_EMBODIMENT,
    *,
    language: str = LANGUAGE,
    video_mode: str = "gradient",
    seed: int = 0,
) -> dict[str, Any]:
    """Build the checkpoint-shaped fixed input (extra args retain caller compatibility)."""
    if video_mode != "gradient":
        raise ValueError("Only the deterministic gradient video is supported")
    del seed  # The historical gradient is deterministic and seed-independent.

    config, stats, contract = _load_metadata(checkpoint, embodiment_tag.lower())
    video_shape = (
        contract.conditioning_frames,
        contract.image_height,
        contract.image_width,
        3,
    )
    observation: dict[str, Any] = {}
    for key in config["video"]["modality_keys"]:
        observation[f"video.{key}"] = _gradient_video(video_shape)
    for key in config["state"]["modality_keys"]:
        mean = np.asarray(stats["state"][key]["mean"], dtype=np.float32)
        observation[f"state.{key}"] = mean[None, :]
    for key in config["language"]["modality_keys"]:
        observation[key] = language
    return observation


def _run_policy_probe(
    args: argparse.Namespace, output_dir: Path
) -> tuple[torch.nn.Module, dict[str, Path], dict[str, dict[str, Any]]]:
    from groot.core.data import schema
    from groot.core.policy.gr00t_policy import Gr00tFlatPolicyWrapper
    from groot.core.policy.n2_policy import Gr00tN2Policy

    os.environ["GROOT_N2_JOINT_DENOISE"] = "1"
    os.environ["GROOT_N2_KV_CACHE_COND"] = "1"
    os.environ["GROOT_N2_NUM_VIDEO_STEPS"] = str(args.num_video_steps)
    os.environ["GROOT_N2_SHIFT"] = "5"
    os.environ["GROOT_N2_CODEC_ROUNDTRIP"] = args.codec
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    policy = Gr00tN2Policy(
        schema.EmbodimentTag[args.policy_embodiment_tag],
        args.checkpoint,
        num_inference_timesteps=args.num_action_steps,
        device="cuda",
        strict=True,
        use_cuda_graph=False,
    )
    original_call_transformer = policy.model._call_transformer
    captured, sample_paths = _install_call_probe(policy.model, sample_dir)
    observation = make_synthetic_observation(
        Path(args.checkpoint),
        args.embodiment_tag,
        language=args.language,
        video_mode="gradient",
        seed=0,
    )
    try:
        with torch.inference_mode():
            Gr00tFlatPolicyWrapper(policy, strict=True).get_action(observation)
            if "action_cached" not in captured:
                _probe_unused_action_engine(policy.model, captured["joint_full_capture"])
    finally:
        policy.model._call_transformer = original_call_transformer
    missing = {"joint_full_capture", "action_cached"} - set(captured)
    if missing:
        raise RuntimeError(
            f"policy probe did not capture required transformer calls: {sorted(missing)}"
        )
    return policy.model, sample_paths, captured


def _probe_unused_action_engine(model, joint_payload):
    """Capture the unused cached-action ABI for equal video/action schedules."""
    packed = joint_payload["packed"]
    action_indices = torch.where(packed.modality_ids[0] == 3)[0]
    positions = torch.arange(action_indices.numel(), device=packed.tokens.device)
    action_packed = replace(
        packed,
        tokens=packed.tokens.index_select(1, action_indices),
        timesteps=packed.timesteps.index_select(1, action_indices),
        rope_freqs=tuple(freq.index_select(0, action_indices) for freq in packed.rope_freqs),
        modality_ids=packed.modality_ids.index_select(1, action_indices),
        attention_mask=None,
        pad=0,
        extra_pos_emb=None,
        video_token_indices=positions[:0],
        state_token_indices=positions[:0],
        action_token_indices=positions,
        non_video_token_indices=positions,
        has_video_cond_tokens=False,
        token_valid_mask=None,
    )
    kwargs = {}
    if joint_payload.get("token_merge") is not None:
        kwargs["token_merge"] = joint_payload["token_merge"]
    cache = model._slice_cache_to_static_prefix(
        joint_payload["kv_cache"],
        packed.modality_ids,
        token_valid_mask=packed.token_valid_mask,
        **kwargs,
    )
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        model._call_transformer(
            action_packed, joint_payload["prompt_embs"], kv_cache=cache, store_kv=False
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _runtime_manifest(args: argparse.Namespace, *, num_cameras: int) -> dict[str, Any]:
    import onnx

    from groot.deployment.n2_trt.manifest import SourceRevision

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
    }
    return {
        "checkpoint": args.checkpoint,
        "source": asdict(
            SourceRevision.capture("exporter", __file__, revision=args.source_revision)
        ),
        "num_cameras": num_cameras,
        "mode": args.mode,
        "transformer_precision": TransformerPrecision.BF16.value,
        "joint_mask_export": args.joint_mask_export,
        "embodiment_tag": args.embodiment_tag,
        "policy_embodiment_tag": args.policy_embodiment_tag,
        "codec_roundtrip": args.codec,
        "capture_samples": 1,
        "opset": args.opset,
        "predict_kwargs": {
            "num_action_steps": args.num_action_steps,
            "num_video_steps": args.num_video_steps,
            "joint_denoise": True,
            "kv_cache_cond": True,
            "predict_video": False,
            "shift": args.shift,
            "seed": args.seed,
        },
        "versions": versions,
        "environment": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH"),
            "ENABLE_TENSORRT": os.environ.get("ENABLE_TENSORRT"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def _onnx_summary(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": opset.domain, "version": opset.version} for opset in model.opset_import
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stack_cache(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not key_cache or not value_cache:
        raise ValueError("Captured action_cached payload has no populated KV cache")
    return torch.stack(key_cache, dim=0), torch.stack(value_cache, dim=0)


def _action_inputs(
    payload: dict[str, Any], *, token_merge_layers: tuple[int, int] | None = None
) -> tuple[torch.Tensor, ...]:
    packed = payload["packed"]
    prompt_embs = payload["prompt_embs"]
    if token_merge_layers is None:
        cache_inputs = _stack_cache(payload["kv_cache"])
    else:
        from groot.deployment.n2_trt.runtime import _stack_token_merge_cache

        cache_inputs = _stack_token_merge_cache(payload["kv_cache"], token_merge_layers)
    return (
        packed.tokens,
        packed.timesteps,
        packed.rope_freqs[0],
        packed.rope_freqs[1],
        prompt_embs,
        packed.modality_ids,
        *cache_inputs,
    )


def _static_cache_outputs(
    joint_output: Any,
    joint_payload: dict[str, Any],
    *,
    token_merge_layers: tuple[int, int] | None,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(joint_output, tuple):
        raise TypeError("joint export adapter must return hidden state and KV tensors")
    packed = joint_payload["packed"]
    modality_ids = packed.modality_ids[0]
    fine_indices = torch.nonzero(modality_ids != 3).flatten()

    if token_merge_layers is None:
        if len(joint_output) != 3:
            raise ValueError(f"joint export returned {len(joint_output)} tensors; expected 3")
        return tuple(tensor.index_select(3, fine_indices) for tensor in joint_output[1:])

    if len(joint_output) != 5:
        raise ValueError(
            f"token-merge joint export returned {len(joint_output)} tensors; expected 5"
        )
    token_merge = joint_payload.get("token_merge")
    if token_merge is None:
        raise ValueError("token-merge joint export is missing its token layout")
    coarse_modality_ids = modality_ids.index_select(0, token_merge.reduced_token_indices.long())
    coarse_indices = torch.nonzero(coarse_modality_ids != 3).flatten()
    return (
        joint_output[1].index_select(3, fine_indices),
        joint_output[2].index_select(3, fine_indices),
        joint_output[3].index_select(3, coarse_indices),
        joint_output[4].index_select(3, coarse_indices),
    )


def _joint_inputs(payload: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    packed = payload["packed"]
    prompt_embs = payload["prompt_embs"]
    return (
        packed.tokens,
        packed.timesteps,
        packed.rope_freqs[0],
        packed.rope_freqs[1],
        prompt_embs,
        packed.modality_ids,
    )


def _captured_layout_record(payload: dict[str, Any]) -> dict[str, Any]:
    packed = payload["packed"]
    modality_ids = packed.modality_ids
    ids = modality_ids[0]
    return {
        "tokens": _tensor_record(packed.tokens),
        "timesteps": _tensor_record(packed.timesteps),
        "rope_cos": _tensor_record(packed.rope_freqs[0]),
        "rope_sin": _tensor_record(packed.rope_freqs[1]),
        "context": _tensor_record(payload["prompt_embs"]),
        "modality_ids": _tensor_record(modality_ids),
        "shared_batch_layout": bool(
            torch.equal(modality_ids, modality_ids[:1].expand_as(modality_ids))
        ),
        "segment_token_counts": {
            "video": int(((ids == 0) | (ids == 1)).sum().item()),
            "video_denoising": int((ids == 0).sum().item()),
            "video_conditioning": int((ids == 1).sum().item()),
            "state": int((ids == 2).sum().item()),
            "action": int((ids == 3).sum().item()),
        },
        "attention_mask_type": (
            type(packed.attention_mask).__name__ if packed.attention_mask is not None else None
        ),
        "attention_mask_pattern": "bidirectional_video_action_full",
        "all_tokens_valid": True,
        "pad": int(packed.pad),
    }


def _run_adapter(adapter: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> Any:
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        output = adapter(*inputs)
    _sync()
    return output


def _output_record(output: Any) -> Any:
    if isinstance(output, tuple):
        return [_tensor_record(item) for item in output]
    return _tensor_record(output)


def _save_export_reference(
    path: Path,
    *,
    oracle: str,
    input_names: list[str],
    inputs: tuple[torch.Tensor, ...],
    output_names: list[str],
    output: Any,
    extras: dict[str, torch.Tensor] | None = None,
) -> None:
    outputs = output if isinstance(output, tuple) else (output,)
    if len(input_names) != len(inputs) or len(output_names) != len(outputs):
        raise ValueError("export reference names do not match the captured tensors")
    sample: dict[str, Any] = {"oracle": oracle}
    sample.update(
        {name: tensor.detach().cpu() for name, tensor in zip(input_names, inputs, strict=True)}
    )
    sample.update(
        {name: tensor.detach().cpu() for name, tensor in zip(output_names, outputs, strict=True)}
    )
    if extras:
        sample.update({name: tensor.detach().cpu() for name, tensor in extras.items()})
    torch.save(sample, path)


def _export_onnx(
    *,
    adapter: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        torch.onnx.export(
            adapter,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            external_data=True,
            do_constant_folding=False,
        )
    _sync()
    elapsed_s = time.perf_counter() - start
    return {
        "onnx": str(output_path),
        "export_elapsed_s": elapsed_s,
        "onnx_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "onnx_summary": _onnx_summary(output_path),
    }


def main() -> int:
    args = _parse_args()
    from groot.deployment.n2_trt.manifest import (
        TRANSFORMER_EXPORT_ACTION_ORACLE,
        TRANSFORMER_EXPORT_JOINT_ORACLE,
        sha256_file,
    )

    args.checkpoint = str(_prepare_checkpoint(args.checkpoint))
    contract = N2TensorRTContract.from_checkpoint(
        args.checkpoint,
        embodiment=args.embodiment_tag,
        video_denoising_steps=args.num_video_steps,
        action_denoising_steps=args.num_action_steps,
        joint_denoise=True,
        conditional_kv_cache=True,
    )

    # Retain production-native samples for behavioral diagnosis. Engine
    # correctness is qualified separately against the exact SDPA export source.
    os.environ["ENABLE_TENSORRT"] = "False"
    production_sample_oracle = _production_eager_oracle()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[n2-transformer-export] output_dir={output_dir}")
    print(f"[n2-transformer-export] checkpoint={args.checkpoint}")
    print("[n2-transformer-export] running policy-shaped transformer call probe...")
    model, sample_paths, captured_samples = _run_policy_probe(args, output_dir)
    # The captured production outputs above remain immutable CPU samples.  Only
    # the adapter execution and ONNX trace below use the exportable SDPA path.
    os.environ["ENABLE_TENSORRT"] = "True"

    wanted_joint = args.mode in ("production", "joint_capture")
    wanted_action = args.mode in ("production", "action_cached")
    if wanted_joint and "joint_full_capture" not in captured_samples:
        raise RuntimeError("No joint_full_capture transformer call was captured")
    if wanted_action and "action_cached" not in captured_samples:
        raise RuntimeError("No action_cached transformer call was captured")

    original_flex_attention = None
    dense_mask = None
    segmented_layout = None
    coarse_segmented_layout = None
    token_merge = None
    token_merge_layers = None
    if wanted_joint:
        packed = captured_samples["joint_full_capture"]["packed"]
        if packed.attention_mask is None:
            raise RuntimeError("Captured joint_full_capture payload has no attention mask")
        token_merge = captured_samples["joint_full_capture"].get("token_merge")
        if model.config.token_merge_kernel is not None and token_merge is None:
            raise RuntimeError("token-merge checkpoint did not pass a plan to the joint call")
        if token_merge is not None:
            token_merge_layers = (token_merge.start_layer, token_merge.end_layer)
        dense_mask = _dense_token_mask_from_block_mask(packed.attention_mask)
        (
            original_flex_attention,
            segmented_layout,
            coarse_segmented_layout,
        ) = _install_segmented_sdpa_flex_patch(
            packed.modality_ids,
            dense_mask,
            num_cameras=model.max_n_cameras,
            action_horizon=int(model.config.action_cfg["action_horizon"]),
            latent_shape=_model_latent_shape(
                model,
                conditioning_frames=contract.conditioning_frames,
                image_size=contract.image_size,
            ),
            total_latent_frames=contract.total_latent_frames,
            token_merge=token_merge,
        )

    joint_adapter = None
    joint_inputs = None
    joint_output = None
    action_adapter = None
    action_inputs = None
    conditioning_stack = ExitStack()
    try:
        conditioning_metadata = conditioning_stack.enter_context(
            export_conditioning(
                model.transformer,
                torch.cat([call["sigmas"] for call in model._conditioning_probe_calls]),
            )
        )
        conditioning_metadata["observed_call_count"] = len(model._conditioning_probe_calls)
        conditioning_metadata["probe_calls"] = [
            {**call, "sigmas": call["sigmas"].float().tolist()}
            for call in model._conditioning_probe_calls
        ]
        if wanted_joint:
            joint_payload = captured_samples["joint_full_capture"]
            packed = joint_payload["packed"]
            video_token_indices, non_video_token_indices = _resolve_routing_indices(
                packed.modality_ids,
                getattr(packed, "video_token_indices", None),
                getattr(packed, "non_video_token_indices", None),
            )
            joint_adapter = N2TransformerJointCaptureExportAdapter(
                model.transformer,
                attention_mask=packed.attention_mask,
                pad=int(packed.pad),
                video_token_indices=video_token_indices,
                non_video_token_indices=non_video_token_indices,
                token_merge=token_merge,
            ).eval()
            joint_inputs = _joint_inputs(joint_payload)

        if wanted_action:
            action_payload = captured_samples["action_cached"]
            packed = action_payload["packed"]
            video_token_indices, non_video_token_indices = _resolve_routing_indices(
                packed.modality_ids,
                getattr(packed, "video_token_indices", None),
                getattr(packed, "non_video_token_indices", None),
            )
            if token_merge_layers is None:
                action_adapter = N2TransformerActionCachedExportAdapter(
                    model.transformer, video_token_indices, non_video_token_indices
                ).eval()
            else:
                action_adapter = N2TransformerTokenMergeActionCachedExportAdapter(
                    model.transformer,
                    video_token_indices,
                    non_video_token_indices,
                    token_merge_layers,
                ).eval()

        manifests: dict[str, Any] = {}
        if joint_adapter is not None and joint_inputs is not None:
            joint_adapter.transformer = model.transformer
            print("[n2-transformer-export] exporting joint-capture ONNX...")
            joint_output = _run_adapter(joint_adapter, joint_inputs)
            joint_output_names = (
                ["output", "output2", "output3"]
                if token_merge_layers is None
                else [
                    "output",
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ]
            )
            joint_input_names = [
                "latents",
                "timesteps",
                "rope_cos",
                "rope_sin",
                "context",
                "modality_ids",
            ]
            joint_reference_path = output_dir / "samples/joint_export_source.pt"
            joint_extras = {}
            if token_merge is not None:
                joint_extras["coarse_token_idx"] = token_merge.reduced_token_indices
            _save_export_reference(
                joint_reference_path,
                oracle=TRANSFORMER_EXPORT_JOINT_ORACLE,
                input_names=joint_input_names,
                inputs=joint_inputs,
                output_names=joint_output_names,
                output=joint_output,
                extras=joint_extras,
            )
            sample_paths["joint_export_source"] = joint_reference_path
            export = _export_onnx(
                adapter=joint_adapter,
                inputs=joint_inputs,
                output_path=output_dir / "n2_transformer_joint.onnx",
                input_names=joint_input_names,
                output_names=joint_output_names,
                opset=args.opset,
            )
            manifests["joint_capture"] = {
                **export,
                "sample_path": str(joint_reference_path),
                "production_sample_path": str(sample_paths.get("joint_full_capture")),
                "input_records": [_tensor_record(item) for item in joint_inputs],
                "output": _output_record(joint_output),
                "attention_mask_type": type(
                    captured_samples["joint_full_capture"]["packed"].attention_mask
                ).__name__,
                "dense_mask": _tensor_record(dense_mask),
                "joint_mask_export": args.joint_mask_export,
                "segmented_layout": (segmented_layout.to_manifest() if segmented_layout else None),
                "coarse_segmented_layout": (
                    coarse_segmented_layout.to_manifest()
                    if coarse_segmented_layout is not None
                    else None
                ),
                "pad": int(captured_samples["joint_full_capture"]["packed"].pad),
            }
            _write_json(
                output_dir / "n2_transformer_joint_export.json",
                manifests["joint_capture"],
            )

        if action_adapter is not None:
            if joint_output is None:
                raise RuntimeError("action export requires the matched joint export output")
            action_payload = captured_samples["action_cached"]
            captured_action_inputs = _action_inputs(
                action_payload,
                token_merge_layers=token_merge_layers,
            )
            source_cache = _static_cache_outputs(
                joint_output,
                captured_samples["joint_full_capture"],
                token_merge_layers=token_merge_layers,
            )
            action_inputs = (*captured_action_inputs[: -len(source_cache)], *source_cache)
            action_adapter.transformer = model.transformer
            print("[n2-transformer-export] exporting action-cached ONNX...")
            output = _run_adapter(action_adapter, action_inputs)
            action_cache_input_names = (
                ["kv_cache_key", "kv_cache_value"]
                if token_merge_layers is None
                else [
                    "kv_cache_key_fine",
                    "kv_cache_value_fine",
                    "kv_cache_key_coarse",
                    "kv_cache_value_coarse",
                ]
            )
            action_input_names = [
                "latents",
                "timesteps",
                "rope_cos",
                "rope_sin",
                "context",
                "modality_ids",
                *action_cache_input_names,
            ]
            action_reference_path = output_dir / "samples/action_export_source.pt"
            _save_export_reference(
                action_reference_path,
                oracle=TRANSFORMER_EXPORT_ACTION_ORACLE,
                input_names=action_input_names,
                inputs=action_inputs,
                output_names=["output1"],
                output=output,
            )
            sample_paths["action_export_source"] = action_reference_path
            export = _export_onnx(
                adapter=action_adapter,
                inputs=action_inputs,
                output_path=output_dir / "n2_transformer_action_cache.onnx",
                input_names=action_input_names,
                output_names=["output1"],
                opset=args.opset,
            )
            manifests["action_cached"] = {
                **export,
                "sample_path": str(action_reference_path),
                "production_sample_path": str(sample_paths.get("action_cached")),
                "input_records": [_tensor_record(item) for item in action_inputs],
                "output": _output_record(output),
                "kv_cache_inputs": {
                    name: _tensor_record(value)
                    for name, value in zip(
                        action_cache_input_names,
                        action_inputs[-len(action_cache_input_names) :],
                        strict=True,
                    )
                },
            }
            _write_json(
                output_dir / "n2_transformer_action_cache_export.json",
                manifests["action_cached"],
            )

        manifest = _runtime_manifest(args, num_cameras=model.max_n_cameras)
        manifest.update(
            {
                "output_dir": str(output_dir),
                "production_sample_oracle": production_sample_oracle,
                "conditioning": conditioning_metadata,
                "engine_oracles": {
                    "joint": TRANSFORMER_EXPORT_JOINT_ORACLE,
                    "action_cache": TRANSFORMER_EXPORT_ACTION_ORACLE,
                },
                "sample_paths": {key: str(value) for key, value in sample_paths.items()},
                "validation_sample_sha256": {
                    "transformer_joint_export": sha256_file(sample_paths["joint_export_source"]),
                    "transformer_action_export": sha256_file(sample_paths["action_export_source"]),
                    "transformer_joint_production": sha256_file(sample_paths["joint_full_capture"]),
                    "transformer_action_production": sha256_file(sample_paths["action_cached"]),
                },
                "token_merge_kernel": (
                    list(model.config.token_merge_kernel)
                    if model.config.token_merge_kernel is not None
                    else None
                ),
                "token_merge_layers": (
                    list(token_merge_layers) if token_merge_layers is not None else None
                ),
                "exports": manifests,
                "captured_layouts": {
                    name: _captured_layout_record(payload)
                    for name, payload in captured_samples.items()
                },
                "notes": [
                    "Transformer weights and graph I/O are BF16.",
                    "Segmented attention executes in BF16; eager parity is qualified separately.",
                ],
            }
        )
        _write_json(output_dir / "n2_transformer_export.json", manifest)
    finally:
        conditioning_stack.close()
        _restore_flex_patch(original_flex_attention)

    print("[n2-transformer-export] done")
    print(f"[n2-transformer-export] manifest={output_dir / 'n2_transformer_export.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
