# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Node-local checkpoint filesystem policy."""

from pathlib import Path


def node_local_checkpoint_output(output_dir: str | Path) -> bool:
    """Classify this path, not other paths used by a node-local training run."""
    node_local_root = Path("/tmp").resolve()  # noqa: S108 - classify, not create
    return Path(output_dir).resolve().is_relative_to(node_local_root)
