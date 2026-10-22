# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Entry-point lookup that also works in source-only runs."""

from __future__ import annotations

from configparser import ConfigParser
from importlib import metadata, resources


def find_entry_points(group: str, name: str, *, missing_ok: bool) -> list[metadata.EntryPoint]:
    """Return the entry points registered as ``name`` in ``group``.

    Installed packages advertise them through package metadata. Source-only runs
    (including --no-install-project containers) read ``groot/entry_points.txt``, the
    manifest that generates that metadata; the OSS package does not ship it. When the
    manifest exists but does not list ``name`` under ``group``, return nothing if
    ``missing_ok``, otherwise raise ``KeyError``.
    """
    found = list(metadata.entry_points(group=group, name=name))
    if found:
        return found
    manifest = resources.files("groot").joinpath("entry_points.txt")
    if not manifest.is_file():
        return []
    config = ConfigParser(interpolation=None)
    config.read_string(manifest.read_text(encoding="utf-8"))
    if missing_ok and not config.has_option(group, name):
        return []
    return [metadata.EntryPoint(name=name, value=config[group][name], group=group)]
