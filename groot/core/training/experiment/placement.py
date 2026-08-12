# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Ray placement helpers for distributed training."""

from __future__ import annotations


def gpu_placement_bundles(
    num_gpus: int,
    dataloader_num_workers: int,
    *,
    head_node_resource: str | None = None,
) -> list[dict[str, float | int]]:
    """Build GPU bundles, optionally anchoring only the first to the head."""
    bundles: list[dict[str, float | int]] = [
        {"CPU": dataloader_num_workers, "GPU": 1} for _ in range(num_gpus)
    ]
    if bundles and head_node_resource:
        # Anchor the placement group to the head without constraining the
        # remaining bundles, which must stay free to span additional nodes.
        bundles[0][head_node_resource] = 0.001
    return bundles
