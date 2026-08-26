# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Storage utilities and optional package-local MSC configuration."""

from collections.abc import Mapping
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_msc_telemetry() -> ModuleType | None:
    """Import package-local MSC telemetry support when this distribution provides it."""
    telemetry = Path(__file__).with_name("_msc_telemetry.py")
    if telemetry.is_file():
        return importlib.import_module("._msc_telemetry", __name__)
    return None


_MSC_TELEMETRY = _load_msc_telemetry()


def load_msc_telemetry_settings(
    *,
    credentials_file: str,
    profile: str,
) -> Mapping[str, Any] | None:
    """Load package-local MSC telemetry settings, or disable telemetry when unavailable."""
    if _MSC_TELEMETRY is not None:
        return _MSC_TELEMETRY.load_msc_telemetry_settings(
            credentials_file=credentials_file,
            profile=profile,
        )
    return None


__all__ = ["load_msc_telemetry_settings"]
