# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""VideoTokenizer: video-specific tokenization extracted from CosmosTransformer3DModel.

Owns patchify, unpatchify, learnable positional embeddings, and view embeddings.
These are the pretrained Cosmos modules that operate on (B, C, T, H, W) video tensors,
converting them to/from (B, L, D) token sequences.

Does NOT own RoPE — that lives on GrootN2 alongside action_rope/state_rope so
the SequencePacker can concatenate all modalities' RoPE uniformly.
"""

import dataclasses

import torch
from torch import nn
from torchvision import transforms

from groot.core.training.models.n2.components.transformer_cosmos import (
    CosmosLearnablePositionalEmbed,
    CosmosPatchEmbed,
    CosmosTransformerConfig,
)


class VideoTokenizer(nn.Module):
    """Video-specific tokenization: patchify, unpatchify, learnable pos embed.

    Extracted from CosmosTransformer3DModel. Holds the pretrained Cosmos weights
    for patch_embed, proj_out, learnable_pos_embed, and view_embed.
    Does NOT own RoPE — that lives on GrootN2 as video_rope.
    """

    def __init__(
        self,
        config: CosmosTransformerConfig,
        bottleneck_kernel: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.base_patch_size = config.patch_size
        self.bottleneck_kernel = bottleneck_kernel
        if bottleneck_kernel is not None:
            if (
                not isinstance(bottleneck_kernel, list | tuple)
                or len(bottleneck_kernel) != 3
                or any(type(size) is not int or size < 1 for size in bottleneck_kernel)
            ):
                raise ValueError("bottleneck_kernel must contain three positive integers")
            if bottleneck_kernel[0] != 1:
                raise ValueError("bottleneck_kernel must be spatial-only: (1, kh, kw)")
            config = dataclasses.replace(
                config,
                patch_size=tuple(
                    patch * bottleneck
                    for patch, bottleneck in zip(config.patch_size, bottleneck_kernel, strict=True)
                ),
            )
            if config.extra_pos_embed_type == "learnable":
                raise NotImplementedError(
                    "vae_bottleneck_kernel does not resample learnable positional tables"
                )
        self.config = config
        hidden_size = config.num_attention_heads * config.attention_head_dim

        # 1. Patch embedding
        patch_in_ch = config.in_channels + 1  # +1 for padding mask channel
        patch_in_ch += config.view_embed_dim
        self.patch_embed = CosmosPatchEmbed(patch_in_ch, hidden_size, config.patch_size, bias=False)

        # 2. Learnable positional embeddings (14B only)
        self.learnable_pos_embed = None
        if config.extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size, config.max_size, config.patch_size
            )

        # 3. View embeddings (multiview only)
        self.view_embed = None
        if config.view_embed_dim > 0:
            self.view_embed = nn.Embedding(config.max_n_cameras, config.view_embed_dim)

        # 4. Output projection (inverse of patchify)
        self.proj_out = nn.Linear(
            hidden_size,
            config.patch_size[0]
            * config.patch_size[1]
            * config.patch_size[2]
            * config.out_channels,
            bias=False,
        )

        if bottleneck_kernel is not None:
            base_patch_embed = CosmosPatchEmbed(
                patch_in_ch, hidden_size, self.base_patch_size, bias=False
            )
            base_proj_out = nn.Linear(
                hidden_size,
                self.base_patch_size[0]
                * self.base_patch_size[1]
                * self.base_patch_size[2]
                * config.out_channels,
                bias=False,
            )
            with torch.no_grad():
                self.patch_embed.proj.weight.copy_(
                    self._expand_patch_embed_weight(base_patch_embed.proj.weight)
                )
                self.proj_out.weight.copy_(self._expand_proj_out_weight(base_proj_out.weight))

    def _expand_patch_embed_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bottleneck_kernel is None:
            return weight
        p_t, p_h, p_w = self.base_patch_size
        k_t, k_h, k_w = self.bottleneck_kernel
        in_channels = weight.shape[1] // (p_t * p_h * p_w)
        # Expand in fp32: bf16 intermediates lose uniform-block equivalence.
        dtype = weight.dtype
        weight = weight.float().unflatten(1, (in_channels, p_t, p_h, p_w))
        # Preserve the block mean and encode within-block deviations in distinct
        # channel-rolled directions, matching the upstream bottleneck warm start.
        num_sub = k_t * k_h * k_w
        hidden = weight.shape[0]
        shifts = [(i + 1) * (hidden // (num_sub + 1)) for i in range(num_sub)]
        rolled = torch.stack([weight.roll(s, dims=0) for s in shifts])
        blocks = (weight + rolled - rolled.mean(0)) / num_sub
        blocks = blocks.unflatten(0, (k_t, k_h, k_w))
        blocks = blocks.permute(3, 4, 0, 5, 1, 6, 2, 7)
        return (
            blocks.reshape(hidden, in_channels, k_t * p_t, k_h * p_h, k_w * p_w)
            .flatten(1)
            .to(dtype)
        )

    def _expand_proj_out_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bottleneck_kernel is None:
            return weight
        p_t, p_h, p_w = self.base_patch_size
        k_t, k_h, k_w = self.bottleneck_kernel
        out_channels = weight.shape[0] // (p_t * p_h * p_w)
        weight = weight.unflatten(0, (p_h, p_w, p_t, out_channels))
        return weight.repeat(k_h, k_w, k_t, 1, 1).flatten(0, 3)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.bottleneck_kernel is not None:
            k_t, k_h, k_w = self.bottleneck_kernel
            num_sub = k_t * k_h * k_w
            patch_key = f"{prefix}patch_embed.proj.weight"
            patch_shape = self.patch_embed.proj.weight.shape
            out_key = f"{prefix}proj_out.weight"
            out_shape = self.proj_out.weight.shape
            # Expand only base-geometry tensors. Already-trained bottleneck
            # weights pass through unchanged; other mismatches remain errors.
            if (
                patch_key in state_dict
                and state_dict[patch_key].shape == (patch_shape[0], patch_shape[1] // num_sub)
                and state_dict[patch_key].shape != patch_shape
            ):
                state_dict[patch_key] = self._expand_patch_embed_weight(state_dict[patch_key])
            if (
                out_key in state_dict
                and state_dict[out_key].shape == (out_shape[0] // num_sub, out_shape[1])
                and state_dict[out_key].shape != out_shape
            ):
                state_dict[out_key] = self._expand_proj_out_weight(state_dict[out_key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def patchify(
        self,
        video: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        n_cameras: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Convert (B, C, T, H, W) video to (B, L, D) tokens.

        Returns:
            tokens: (B, L, D) patchified video tokens
            extra_pos_emb: (B, L, D) learnable positional embedding, or None
            shape_info: dict with spatial dims needed for unpatchify
        """
        B, C, T, H, W = video.shape
        hidden_states = video

        # 1. Concatenate condition_mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        # 2. Concatenate padding_mask
        if padding_mask is None:
            padding_mask = hidden_states.new_zeros(1, 1, H, W)
        pm = transforms.functional.resize(
            padding_mask,
            list(hidden_states.shape[-2:]),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        hidden_states = torch.cat([hidden_states, pm.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1)

        # 3. View embeddings
        # patch_embed reserves these channels even for a single camera.
        if self.view_embed is not None:
            T_per_view = T // n_cameras
            view_ids = torch.arange(n_cameras, device=hidden_states.device).repeat_interleave(
                T_per_view
            )
            ve = self.view_embed(view_ids).to(hidden_states.dtype)
            ve = ve.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, D_ve, T, 1, 1)
            ve = ve.expand(B, -1, -1, H, W)
            hidden_states = torch.cat([hidden_states, ve], dim=1)

        # 4. Learnable positional embedding
        extra_pos_emb = (
            self.learnable_pos_embed(hidden_states)
            if self.learnable_pos_embed is not None
            else None
        )

        # 6. Patchify
        p_t, p_h, p_w = self.config.patch_size
        post_T, post_H, post_W = T // p_t, H // p_h, W // p_w
        tokens = self.patch_embed(hidden_states).flatten(1, 3)  # (B, L, D)

        shape_info = {
            "B": B,
            "T": T,
            "H": H,
            "W": W,
            "post_T": post_T,
            "post_H": post_H,
            "post_W": post_W,
            "p_t": p_t,
            "p_h": p_h,
            "p_w": p_w,
        }

        return tokens, extra_pos_emb, shape_info

    def unpatchify(self, tokens: torch.Tensor, shape_info: dict) -> torch.Tensor:
        """Convert (B, L, D) tokens back to (B, C_out, T, H, W) video.

        Applies proj_out then reshapes. The permutation order matches the diffusers
        convention (not the intuitive inverse of patching).
        """
        p_t = shape_info["p_t"]
        p_h = shape_info["p_h"]
        p_w = shape_info["p_w"]
        post_T = shape_info["post_T"]
        post_H = shape_info["post_H"]
        post_W = shape_info["post_W"]

        tokens = self.proj_out(tokens)
        tokens = tokens.unflatten(2, (p_h, p_w, p_t, -1))
        tokens = tokens.unflatten(1, (post_T, post_H, post_W))
        # NOTE: permutation order matches diffusers (not the intuitive inverse of patching)
        tokens = tokens.permute(0, 7, 1, 6, 2, 4, 3, 5)
        tokens = tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return tokens
