# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Import-light resolution of the N2 observation-video codec."""

from __future__ import annotations

import os

N2_CODEC_ROUNDTRIPS = ("h264", "none", "mp4v")


def resolve_n2_codec_roundtrip(codec_roundtrip: str = "mp4v") -> str:
    """Resolve the effective serving codec without importing the N2 runtime.

    ``GROOT_N2_CODEC_ROUNDTRIP`` is the only supported environment override.
    Reject retired or misspelled roundtrip variables instead of silently using a
    different codec from the one requested by the caller.
    """
    stale = sorted(
        name
        for name in os.environ
        if name.startswith("GROOT_N2_")
        and name.endswith("_ROUNDTRIP")
        and name != "GROOT_N2_CODEC_ROUNDTRIP"
    )
    if stale:
        raise ValueError(
            f"unrecognized roundtrip env var(s) {stale}; this policy reads only "
            f"GROOT_N2_CODEC_ROUNDTRIP (one of {N2_CODEC_ROUNDTRIPS}). A retired "
            "boolean spelling maps 0 -> 'none' and 1 -> 'h264'."
        )
    codec = os.environ.get("GROOT_N2_CODEC_ROUNDTRIP", codec_roundtrip).lower()
    if codec not in N2_CODEC_ROUNDTRIPS:
        raise ValueError(f"codec_roundtrip must be one of {N2_CODEC_ROUNDTRIPS}, got {codec!r}")
    return codec
