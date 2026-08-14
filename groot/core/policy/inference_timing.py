# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Wall-clock segment logging for remote policy inference."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def inference_timing_enabled() -> bool:
    """Return whether inference timing logs are enabled."""
    raw = os.environ.get("GROOT_INFERENCE_TIMING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def log_inference_timing(tag: str, /, **segments_s: float) -> None:
    """Log one timing line: ``[InferenceTiming:<tag>] a=1.2ms b=3.4ms ...``."""
    if not inference_timing_enabled():
        return
    parts = " ".join(f"{name}={sec * 1000:.1f}ms" for name, sec in segments_s.items())
    logger.info("[InferenceTiming:%s] %s", tag, parts)
