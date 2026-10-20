# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Parameters-only bf16 casts (N2 ``rope_fp32``).

``Module.to(dtype=...)`` and DeepSpeed < 0.19.7 (``module.bfloat16()``) cast parameters and
buffers alike, which rounds the fp32 constants a model computes with: the N2 action/state RoPE
tables (``ActionRoPE1D.freqs`` / ``seq``), the Qwen2.5-VL rotary ``inv_freq`` and the Wan VAE
latent ``mean`` / ``inv_std``. RoPE angles then drift by up to 0.13 rad at positions 32-47.
DeepSpeed >= 0.19.7 casts parameters only. A model trained that way (``rope_fp32``) must be
served the same way, and vice versa: the legacy whole-model cast stays the default.
"""

from __future__ import annotations

import importlib.util
import types

import torch
from torch import nn


def cast_parameters_only(
    module: nn.Module,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Move ``module`` to ``device`` and cast its floating-point parameters to ``dtype``.

    Buffers move with the module but keep their dtype. This is DeepSpeed's tensor-level cast
    (``p.data = p.data.to(dtype)``), so a model cast here matches one the engine cast. Each
    parameter is cast as it moves, so ``device`` never holds a copy of it in the old dtype.
    """

    def cast(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, nn.Parameter) and tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    # Module.to's per-tensor walk, which hands each parameter to cast as an nn.Parameter.
    return module._apply(cast)  # noqa: SLF001


def cast_for_inference(
    model: nn.Module,
    *,
    rope_fp32: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """The inference cast a checkpoint was trained under: parameters only, or the whole model."""
    if rope_fp32:
        return cast_parameters_only(model, device=device, dtype=dtype)
    return model.to(device=device, dtype=dtype)


def install_params_only_bf16_cast() -> bool:
    """Make DeepSpeed < 0.19.7 cast parameters only, as >= 0.19.7 does. Idempotent.

    0.18.8's ``DeepSpeedEngine._configure_distributed_model`` runs ``self.module.bfloat16()``.
    For the duration of that call, the module's ``bfloat16`` is shadowed with
    ``cast_parameters_only``. Returns True if the patch was installed by this call; False when
    DeepSpeed is absent, already >= 0.19.7, or already patched.
    """
    if importlib.util.find_spec("deepspeed") is None:
        return False
    import deepspeed
    from deepspeed.runtime.engine import DeepSpeedEngine
    from packaging.version import Version

    if Version(deepspeed.__version__) >= Version("0.19.7"):
        return False
    original = DeepSpeedEngine._configure_distributed_model  # noqa: SLF001
    if getattr(original, "params_only_bf16_cast", False):
        return False

    def _configure_distributed_model(self, model):
        model.bfloat16 = types.MethodType(
            lambda m: cast_parameters_only(m, dtype=torch.bfloat16), model
        )
        try:
            return original(self, model)
        finally:
            del model.bfloat16

    _configure_distributed_model.params_only_bf16_cast = True
    DeepSpeedEngine._configure_distributed_model = _configure_distributed_model  # noqa: SLF001
    return True
