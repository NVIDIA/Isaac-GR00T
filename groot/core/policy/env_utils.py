# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed environment settings shared by policy and runtime setup."""

import os


def env_bool(name: str, *, default: bool) -> bool:
    """Parse a Boolean environment setting without accepting ambiguous values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (expected 1/0/true/false/yes/no)")


def env_bool_optional(name: str) -> bool | None:
    """Parse a Boolean environment setting, or return ``None`` when it is unset."""
    if os.environ.get(name) is None:
        return None
    return env_bool(name, default=False)
