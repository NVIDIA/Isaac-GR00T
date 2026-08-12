# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""DirSyncJob: tracks one in-progress distributed s5cmd directory sync."""

import dataclasses


@dataclasses.dataclass
class DirSyncJob:
    """Tracks one in-progress distributed s5cmd directory sync."""

    s3_base: str  # normalized, no trailing slash
    local_dir: str
