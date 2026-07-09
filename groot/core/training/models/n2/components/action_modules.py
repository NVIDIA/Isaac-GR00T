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
"""Action/state encoding modules for joint video+action training.

Ported from DreamCosmos cosmos_submodule.py with N2 conventions:
- ActionRoPE1D outputs (cos, sin) each (L, D_head) matching Cosmos RoPE format
- All modules use CategorySpecificLinear for multi-embodiment support
"""

import torch
from torch import nn
import torch.nn.functional as F

from groot.core.training.models.modules.flowmatching_modules import SinusoidalPositionalEncoding, swish


class CategorySpecificLinear(nn.Module):
    """Per-embodiment linear layer. Maintains separate weight matrices indexed by category ID."""

    def __init__(self, num_categories: int, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, output_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim), cat_ids: (B,) -> (B, T, out_dim)."""
        W = torch.index_select(self.W, 0, cat_ids).to(x.dtype)
        b = torch.index_select(self.b, 0, cat_ids).to(x.dtype)
        return torch.bmm(x, W) + b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """2-layer MLP with ReLU, per-embodiment weights."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class ActionEncoder(nn.Module):
    """Projects (noisy_action, diffusion_timestep) -> transformer hidden dim.

    3-stage with timestep fusion: W1(action) concat time_emb -> W2+silu -> W3.
    Per-embodiment weights via CategorySpecificLinear.

    Optionally adds a *learned temporal positional embedding* along the action
    sequence (time) axis to the projected action tokens (after W1, before
    time-emb concat). This is multiplicative-RoPE-agnostic — RoPE still
    applies inside attention; this just gives the encoder its own position
    signal in the residual stream. Enabled with ``use_temporal_pos_emb=True``;
    zero-init so the model output is identical at step 0 (safe ablation init).
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        num_embodiments: int = 1,
        use_temporal_pos_emb: bool = False,
        action_horizon: int = 48,
    ):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_dim)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_dim, hidden_dim)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.use_temporal_pos_emb = use_temporal_pos_emb
        if use_temporal_pos_emb:
            # (1, T, hidden_dim) — broadcasts over batch, indexed by seq position.
            # Zero-init keeps step-0 output identical to the no-pos-emb baseline.
            self.temporal_pos_emb = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))

    def forward(
        self,
        actions: torch.Tensor,  # (B, T, action_dim)
        timesteps: torch.Tensor,  # (B, T) diffusion sigma per token
        cat_ids: torch.Tensor,  # (B,) embodiment index
    ) -> torch.Tensor:  # (B, T, hidden_dim)
        a_emb = self.W1(actions, cat_ids)  # (B, T, hidden_dim)
        if self.use_temporal_pos_emb:
            # Slice to actual T in case runtime horizon < table length.
            a_emb = a_emb + self.temporal_pos_emb[:, : a_emb.shape[1], :].to(dtype=a_emb.dtype)
        tau_emb = self.pos_encoding(timesteps * 1000.0).to(dtype=a_emb.dtype)  # (B, T, hidden_dim)
        x = torch.cat([a_emb, tau_emb], dim=-1)  # (B, T, 2*hidden_dim)
        x = swish(self.W2(x, cat_ids))  # (B, T, hidden_dim)
        return self.W3(x, cat_ids)  # (B, T, hidden_dim)


class StateEncoder(nn.Module):
    """Projects robot state -> transformer hidden dim. No timestep (state is observed)."""

    def __init__(self, state_dim: int, hidden_dim: int, num_embodiments: int = 1):
        super().__init__()
        self.mlp = CategorySpecificMLP(num_embodiments, state_dim, hidden_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,  # (B, T, state_dim)
        cat_ids: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, T, hidden_dim)
        return self.mlp(state, cat_ids)


class ActionDecoder(nn.Module):
    """Projects transformer hidden -> action space."""

    def __init__(self, hidden_dim: int, action_dim: int, num_embodiments: int = 1):
        super().__init__()
        self.mlp = CategorySpecificMLP(num_embodiments, hidden_dim, hidden_dim, action_dim)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T, hidden_dim)
        cat_ids: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, T, action_dim)
        return self.mlp(tokens, cat_ids)


class ActionRoPE1D(nn.Module):
    """1D rotary position embedding for action/state token sequences.

    Output format matches CosmosRotaryPosEmbed: (cos, sin) each (L, D_head).
    Uses the Cosmos "real_unbind" convention (freqs doubled before cos/sin).
    """

    def __init__(self, head_dim: int, max_len: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        half_dim = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("freqs", freqs)
        self.register_buffer("seq", torch.arange(max_len, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, D) -> (cos, sin) each (L, D_head)."""
        L = x.shape[1]
        # (L, D//2) outer product of positions and frequencies
        half_emb = torch.outer(self.seq[:L].to(x.device), self.freqs.to(x.device))
        # Double to match Cosmos convention: [half, half] -> (L, D)
        emb = torch.cat([half_emb, half_emb], dim=-1)
        return torch.cos(emb).float(), torch.sin(emb).float()
