# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import warnings
from collections import namedtuple
from types import MethodType
from typing import Iterable, List, Optional, Set, Tuple, Union

# https://github.com/Dao-AILab/flash-attention/blob/v0.2.8/flash_attn/flash_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import checkpoint_seq, create_model, register_model
from timm.models.vision_transformer import (
    Attention,
    Block,
)
from timm.models.vision_transformer import LayerScale as TIMMLayerScale
from timm.models.vision_transformer import (
    VisionTransformer,
)
from timm.models.vision_transformer import (
    _create_vision_transformer as _timm_create_vision_transformer,
)
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

####


try:  # v1
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:  # v2
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
    )

from flash_attn.bert_padding import pad_input, unpad_input


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv,
        key_padding_mask=None,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device
                )
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(
                        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
                    ),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


def _flash_attn(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(0, 2, 1, 3, 4)  # [B, 3, N, num_heads, head_dim]

    if not isinstance(self.q_norm, nn.Identity):
        qkv[:, 0] = self.q_norm(qkv[:, 0])
        qkv[:, 1] = self.k_norm(qkv[:, 1])

    qkv = rearrange(qkv, "b t s h d -> b s t h d")

    context, _ = self.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=False)

    x = rearrange(context, "b s h d -> b s (h d)")
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def forward(self, x: torch.Tensor) -> torch.Tensor:
    assert (
        x.dtype == torch.bfloat16
    ), "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
    result = self._flash_attn(x)
    return result


def replace_vit_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    Attention.forward = forward
    Attention.inner_attn = FlashAttention(attention_dropout=0.0)
    Attention._flash_attn = _flash_attn


replace_vit_attn_with_flash_attn()
####


input_dim_t = Union[int, Tuple[int, int]]

try:
    # raise ImportError()
    from indirect_grid_sample import indirect_grid_sample
except ImportError:
    indirect_grid_sample = None


class ClsToken(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
    ):
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens

        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (num_tokens % register_multiple)

            scale = ndim**-0.5
            self.token = nn.Parameter(torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def disable(self):
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat(
            [
                token,
                x,
            ],
            dim=1,
        )

        return x

    def no_weight_decay(self):
        return [
            "token",
        ]


class ViTPatchGenerator(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: input_dim_t,
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[input_dim_t] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
        patch_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)

        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims)

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc

        factory = dict(device=device, dtype=dtype)

        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim

        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size, embed_dim, bias=patch_bias, **factory)

        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, **factory) * scale
            )

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )

        self.patch_normalizer = nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def apply_cls_token(self):
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self):
        return [
            "pos_embed",
        ]

    def _load_projection(self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1
            ], "Unable to interpolate non-square patch size"

            src_proj_weight = rearrange(
                src_proj_weight, "b (c h w) -> b c h w", c=3, h=src_patch_size, w=src_patch_size
            )
            src_proj_weight = F.interpolate(
                src_proj_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_proj_weight = rearrange(src_proj_weight, "b c h w -> b (c h w)")
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.im_to_patches(x)
        patches = self.embedder(patches)
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = (
                torch.rand(patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device)
                > self.pos_dropout
            )
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs
        )
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = (
                    torch.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale)
                    + min_scale
                )
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (aspect_max - aspect_min)
                    + aspect_min
                )

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)

                lin_x = torch.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[
                    None, None
                ].expand(batch_size, input_dims[0], -1)
                lin_y = torch.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[
                    None, :, None
                ].expand(batch_size, -1, input_dims[1])

                lin_xy = torch.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                # i_rows, i_cols = input_dims
                # p_rows, p_cols = pos_embed.shape[2:]
                # if i_rows <= p_rows and i_cols <= p_cols:
                #     left = (p_cols - i_cols) // 2
                #     top = (p_rows - i_rows) // 2
                #     pos_embed = pos_embed[..., top:top+i_rows, left:left+i_cols]
                # else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(), size=(max_dim, max_dim), align_corners=True, mode="bilinear"
                ).to(pos_embed.dtype)

                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=True, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):
    def __init__(self, patch_size: int, embed_dim: int, bias: bool = False, **factory):
        super().__init__(3 * (patch_size**2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


def _take_indices(
    num_blocks: int,
    n: Optional[Union[int, List[int], Tuple[int]]],
) -> Tuple[Set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def _enable_cpe_for_timm_vit(
    model: VisionTransformer,
    max_img_size: Union[int, Tuple[int, int]] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int = Optional[None],
    num_registers: int = Optional[None],
):
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)
    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)


def enable_cpe(
    model: nn.Module,
    *args,
    **kwargs,
):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
    else:
        raise ValueError(f"CPE not supported for this model type: {type(model)}")


###


class Dinov2LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.grandma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.grandma) if self.inplace else x * self.grandma

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Huggingface is absurd and it will rename strings that contain `gamma`, which means that the normal DINO implementation
        # of LayerScale won't work with HFHub. So we rename the variable to 'grandma', and support loading checkpoints in either
        # format
        key_a = f"{prefix}gamma"
        key_b = f"{prefix}grandma"
        if key_a in state_dict:
            gamma = state_dict[key_a]
        elif key_b in state_dict:
            gamma = state_dict[key_b]
        else:
            if strict:
                raise KeyError(f"Couldn't find the key {key_a} nor {key_b} in the state dict!")
            else:
                missing_keys.append(key_a)
                missing_keys.append(key_b)
                unexpected_keys.extend(state_dict.keys())
                gamma = None

        if gamma is not None:
            self.grandma.data.copy_(gamma)


def _create_vision_transformer(*args, **kwargs):
    model = _timm_create_vision_transformer(*args, **kwargs)
    _patch_layer_scale(model)
    return model


def _patch_layer_scale(model: VisionTransformer):
    def replace_ls(old_ls: TIMMLayerScale):
        new_ls = Dinov2LayerScale(old_ls.gamma.shape[0], inplace=old_ls.inplace)
        new_ls.load_state_dict(old_ls.state_dict())
        return new_ls

    # Monkey patch: Replace TIMM's LayerScale with our modified DINOv2 one, that uses a param name
    # other than gamma, so that HFHub doesn't mess with it!
    for mod in model.modules():
        if isinstance(mod, Block):
            if isinstance(mod.ls1, TIMMLayerScale):
                mod.ls1 = replace_ls(mod.ls1)
            if isinstance(mod.ls2, TIMMLayerScale):
                mod.ls2 = replace_ls(mod.ls2)
    pass


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16, weight_init="skip")
    if pretrained:
        # There is no pretrained version of ViT-H/16, but we can adapt a ViT-H/14 for this purpose
        model = _create_vision_transformer(
            "vit_huge_patch14_224", pretrained=True, **dict(model_args, **kwargs)
        )
    else:
        model = _create_vision_transformer(
            "vit_huge_patch16_224", pretrained=False, **dict(model_args, **kwargs)
        )
    return model


####


class RADIOModelBase(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        max_resolution: int,
    ):
        super().__init__()

        self.model = model
        self._patch_size = patch_size
        self._max_resolution = max_resolution

    @property
    def num_cls_tokens(self) -> int:
        if hasattr(self.model, "num_cls_tokens"):
            return self.model.num_cls_tokens

        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_cls_tokens
        elif self.model.global_pool == "avg":
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        if self._patch_size is not None:
            return self._patch_size
        if hasattr(self.model, "patch_size"):
            return self.model.patch_size
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.patch_size
        return None

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def blocks(self) -> Iterable[nn.Module]:
        blocks = getattr(self.model, "blocks", None)
        if blocks is not None:
            return blocks
        return None

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def forward(
        self, x: torch.Tensor, feature_fmt: str = "NLC"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        """

        y = self.model.forward_features(x)
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return y[:, patch_gen.num_skip :]
        return y


def create_model_from_args(args) -> nn.Module:
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # Skip weight initialization unless it's explicitly requested.
    weight_init = args.model_kwargs.pop("weight_init", "skip")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        weight_init=weight_init,
        **args.model_kwargs,
    )

    if hasattr(model, "norm") and not getattr(args, "model_norm", False):
        model.norm = nn.Identity()

    model.head = nn.Identity()

    if args.cpe_max_size is not None:
        uq_teachers = set(t["name"] for t in args.teachers)
        enable_cpe(
            model,
            args.cpe_max_size,
            num_cls_tokens=len(uq_teachers) if args.cls_token_per_teacher else 1,
            register_multiple=getattr(args, "register_multiple", None),
            num_registers=getattr(args, "cpe_num_registers", None),
        )

    return model


####


class RADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = "radio_v2.5-h",
        patch_size: Optional[int] = None,
        max_resolution: Optional[int] = None,
        model_type: Optional[str] = "radio",
        hidden_size: Optional[int] = 1280,
        **kwargs,
    ):
        self.args = args
        if version == "radio_v2.5-h":
            resource = dict(
                url="https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h.pth.tar?download=true",
                patch_size=16,
                max_resolution=2048,
                vitdet_num_global=4,
            )
        self.patch_size = patch_size or resource["patch_size"]
        self.max_resolution = max_resolution or resource["max_resolution"]
        self.model_type = model_type
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.model_type
        output["hidden_size"] = self.hidden_size
        return output


class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    def __init__(self, config: RADIOConfig):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config

        model = create_model_from_args(args)

        self.radio_model = RADIOModelBase(
            model,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
        )

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def num_summary_tokens(self) -> int:
        return self.radio_model.num_summary_tokens

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    def forward(self, pixel_values: torch.Tensor, output_hidden_states=False, return_dict=True):
        y = self.radio_model.forward(pixel_values.to(self.dtype))
        assert not output_hidden_states
        if return_dict:
            return ModelOutput(
                last_hidden_state=y,
                hidden_states=None,
            )
        else:
            return y
