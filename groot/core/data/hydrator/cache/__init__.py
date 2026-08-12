# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Cache module for local file storage and management.

This module provides cache management functionality for the data hydrator,
including download orchestration, pluggable eviction policies, and file lifecycle tracking.

IMPORTANT: This is an INTERNAL module. Users should never import from here directly.
All cache operations are managed through the Hydrator service layer.
"""

from groot.core.data.hydrator.cache.eviction_policy import (
    EvictionPolicy,
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    SizeBasedEvictionPolicy,
)
from groot.core.data.hydrator.cache.manager import CacheManager

__all__ = [
    "CacheManager",
    "EvictionPolicy",
    "LRUEvictionPolicy",
    "FIFOEvictionPolicy",
    "SizeBasedEvictionPolicy",
]
