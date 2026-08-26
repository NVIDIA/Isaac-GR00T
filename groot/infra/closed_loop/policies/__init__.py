# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Policy adapters for portable closed-loop evaluation."""

from __future__ import annotations

from typing import Any


def load_n2_policy(
    *,
    checkpoint: str,
    cache_root: str | None,
    options: dict[str, Any],
) -> Any:
    """Construct the supported N2 policy without importing model code in the driver."""
    from groot.infra.closed_loop.policies.n2 import N2RolloutPolicy

    return N2RolloutPolicy(checkpoint=checkpoint, cache_root=cache_root, options=options)
