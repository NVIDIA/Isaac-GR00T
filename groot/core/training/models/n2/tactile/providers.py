# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Lazy HB1 factory discovery, shared by direct model and policy construction."""

from __future__ import annotations

from configparser import ConfigParser
from importlib import metadata, resources
from typing import TYPE_CHECKING

from groot.core.training.models.n2.tactile.guards import TactileUnsupportedError

if TYPE_CHECKING:
    from torch import nn

_GROUP = "groot.n2.deform"


def build_hb1_autoencoder() -> nn.Module:
    """Build the full autoencoder; its decoder's RNG use is part of N2 initialization.

    Installed packages advertise the factory through Python entry points. Source-only
    runs (including --no-install-project containers) use the same manifest that generates
    that metadata. The OSS package ships neither the manifest nor the implementation.
    Discovery and imports happen only when deformation construction requests this call.
    Provider import/construction errors propagate unchanged.
    """
    providers = list(metadata.entry_points(group=_GROUP, name="hb1"))
    if not providers:
        manifest = resources.files("groot").joinpath("entry_points.txt")
        if manifest.is_file():
            config = ConfigParser(interpolation=None)
            config.read_string(manifest.read_text(encoding="utf-8"))
            providers = [metadata.EntryPoint(name="hb1", value=config[_GROUP]["hb1"], group=_GROUP)]
    if not providers:
        raise TactileUnsupportedError(
            "HB1 deformation requires the internal GR00T HB1 provider; it is not included in OSS."
        )
    if len(providers) != 1:
        raise RuntimeError(f"Expected one HB1 deformation provider, found {len(providers)}")
    return providers[0].load()()
