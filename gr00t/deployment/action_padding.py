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

"""``strict_action_padding_mask`` on the TensorRT paths.

With the flag on, the PyTorch action head zeroes the padded action dimensions and
keeps the padded horizon tokens out of the DiT self-attention. The TensorRT loops
apply the same masks around their engines, and the DiT engine of a strict
checkpoint takes the token mask as a ``hidden_attention_mask`` input. The masks
and the engine check live here so they can be tested without TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import _expand_action_mask


HIDDEN_ATTENTION_MASK = "hidden_attention_mask"


@dataclass(frozen=True)
class StrictActionMasks:
    """Masks the strict PyTorch action head applies in its denoising loop."""

    action: torch.Tensor  # (B, action_horizon, action_dim), in the actions dtype
    tokens: torch.Tensor  # (B, action_horizon, 1) bool, applied to the action features
    hidden: torch.Tensor  # (B, state tokens + action_horizon) bool, the DiT input


def strict_action_masks(
    config, action_input, actions: torch.Tensor, num_state_tokens: int
) -> StrictActionMasks | None:
    """Build the strict-mode masks for ``actions``, or None when the flag is off."""
    if not config.strict_action_padding_mask:
        return None
    action_mask = _expand_action_mask(action_input.get("action_mask"), actions)
    if action_mask is None:
        raise ValueError("action_mask is required when strict_action_padding_mask is enabled")
    tokens = action_mask.any(dim=-1)
    state = torch.ones(tokens.shape[0], num_state_tokens, dtype=torch.bool, device=tokens.device)
    return StrictActionMasks(
        action=action_mask,
        tokens=tokens.unsqueeze(-1),
        hidden=torch.cat((state, tokens), dim=1),
    )


def assert_dit_engine_matches_action_padding(config, engine_inputs, context: str) -> None:
    """Refuse a DiT engine exported with a different ``strict_action_padding_mask``.

    Args:
        config: Model config carrying ``strict_action_padding_mask``.
        engine_inputs: Input tensor names of the loaded DiT engine.
        context: Name of the TensorRT path, interpolated into the error.
    """
    strict = bool(config.strict_action_padding_mask)
    takes_mask = HIDDEN_ATTENTION_MASK in engine_inputs
    if strict != takes_mask:
        raise ValueError(
            f"{context}: the checkpoint has strict_action_padding_mask={strict}, but its DiT "
            f"engine {'takes' if takes_mask else 'has no'} a {HIDDEN_ATTENTION_MASK} input, so "
            "it was exported from a checkpoint with the other setting. Re-export the engine "
            "from this checkpoint."
        )
