# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolve shared closed-loop success budgets to individual action steps."""

from collections.abc import Sequence
from decimal import Decimal

DEFAULT_SUCCESS_STEP_BUDGETS = (0.25, 0.5, 0.75, 1.0)


def resolve_success_step_budgets(
    budgets: Sequence[int | float] | None, max_episode_steps: int | None
) -> tuple[int, ...]:
    """Accept integer steps or float fractions; floor fractions to at least one step.

    Omitted budgets use defaults only with an explicit episode limit.
    Preserve order and collapse thresholds that coincide after rounding.
    """
    if budgets is None:
        budgets = DEFAULT_SUCCESS_STEP_BUDGETS if max_episode_steps is not None else ()
    resolved = []
    seen = set()
    for budget in budgets:
        if type(budget) is int and budget > 0:
            steps = budget
        elif type(budget) is float and 0 < budget <= 1:
            if max_episode_steps is None:
                raise ValueError(
                    "fractional success_step_budgets require max_episode_steps; "
                    "set --max-episode-steps or use integer step budgets"
                )
            steps = max(1, int(Decimal(str(budget)) * max_episode_steps))
        else:
            raise ValueError(
                "success_step_budgets must contain positive integer steps or float fractions in (0, 1]"
            )
        if (type(budget), budget) in seen:
            raise ValueError("success_step_budgets must not contain duplicates")
        seen.add((type(budget), budget))
        if max_episode_steps is not None and steps > max_episode_steps:
            raise ValueError("success_step_budgets must not exceed max_episode_steps")
        if steps not in resolved:
            resolved.append(steps)
    return tuple(resolved)
