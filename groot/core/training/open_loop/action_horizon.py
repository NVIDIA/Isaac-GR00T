# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""How many steps of each predicted action chunk open-loop evaluation consumes.

Kept free of heavy imports so that callers which cannot afford
``open_loop_eval``'s matplotlib and torch dependencies -- the OSS evaluator and
its CPU-only test job -- can still reach the rule.
"""

from __future__ import annotations

from typing import Any


def resolve_action_horizon(modality_configs: Any, configured: int | None) -> int:
    """Steps to consume from each action chunk the policy predicts.

    ``configured`` pins the horizon; ``None`` follows the checkpoint, whose
    chunk length is its number of action delta indices.

    Only ``None`` means "unset", so a configured ``0`` raises rather than
    quietly becoming the checkpoint's horizon. Both it and an empty chunk would
    otherwise reach the rollout as a zero stride, and ``range`` would blame
    itself for a modality config that is the real fault.

    A pin longer than the chunk raises as well. The rollout would read past the
    end of every chunk and surface a bare ``IndexError`` from inside the loop,
    when the statement worth making is that this checkpoint cannot answer the
    horizon it was asked for.
    """
    chunk = len(modality_configs["action"].delta_indices)
    horizon, source = (chunk, "checkpoint chunk") if configured is None else (configured, "pinned")
    if horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {horizon} from {source}")
    if horizon > chunk:
        raise ValueError(
            f"{source} action horizon {horizon} exceeds the checkpoint's action chunk "
            f"of {chunk} steps; the checkpoint cannot be scored at this horizon"
        )
    return horizon
