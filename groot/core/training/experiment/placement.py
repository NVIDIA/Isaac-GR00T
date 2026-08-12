# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
