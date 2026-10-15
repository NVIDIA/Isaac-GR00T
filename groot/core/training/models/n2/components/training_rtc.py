# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tensor construction for N2 train-time real-time chunking (RTC)."""

import torch


@torch.compiler.disable()
def sample_rtc_delays(
    batch_size: int,
    max_delay: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample one inclusive ``[0, max_delay]`` clean-prefix length per item."""
    return torch.randint(0, max_delay + 1, (batch_size,), device=device)


def build_rtc_training_tensors(
    actions: torch.Tensor,
    noise: torch.Tensor,
    action_mask: torch.Tensor,
    delays: torch.Tensor,
    sampled_t: torch.Tensor,
    *,
    cond_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact clean prefixes, noisy postfixes, and a postfix-only loss mask."""
    batch_size, action_horizon, _ = actions.shape
    positions = torch.arange(action_horizon, device=delays.device)
    prefix_mask = positions.unsqueeze(0) < delays.unsqueeze(1)

    postfix_t = sampled_t.reshape(batch_size, 1).expand(batch_size, action_horizon)
    prefix_t = torch.full_like(postfix_t, 1.0 - cond_sigma)
    per_step_t = torch.where(prefix_mask, prefix_t, postfix_t)
    per_step_sigma = 1.0 - per_step_t

    expanded_t = per_step_t.unsqueeze(-1).to(actions.dtype)
    noisy_actions = (1.0 - expanded_t) * noise + expanded_t * actions
    noisy_actions = torch.where(prefix_mask.unsqueeze(-1), actions, noisy_actions)
    loss_mask = torch.where(prefix_mask.unsqueeze(-1), False, action_mask)
    return noisy_actions, per_step_t, per_step_sigma, loss_mask
