# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Helpers for identifying dataset path formats used by omni loaders.

The standard PackDS path convention uses a trailing ``.packds`` suffix.
Some already-published datasets also exist under suffixless prefixes inside
``yam_packds_*`` buckets/directories, so omni still recognizes those.
"""

from __future__ import annotations

from pathlib import Path


def is_packds_path(path: str | Path) -> bool:
    """Return True when ``path`` should be treated as a PackDS dataset path.

    Recognises:
    - Paths ending with ``.packds``
    - Paths inside ``yam_packds_*`` buckets/directories
    - ``dc://`` DeltaCAT URIs (all DeltaCAT tables are PackDS-backed)
    """
    normalized = str(path).rstrip("/")
    if normalized.startswith("dc://"):
        return True
    if normalized.endswith(".packds"):
        return True

    parts = [part for part in normalized.split("/") if part]
    return any(part.startswith("yam_packds") for part in parts)
