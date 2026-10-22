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

from typing import TYPE_CHECKING

from groot.core.training.models.n2.tactile.guards import TactileUnsupportedError
from groot.core.utils.entry_points import find_entry_points

if TYPE_CHECKING:
    from torch import nn

_GROUP = "groot.n2.deform"


def build_hb1_autoencoder() -> nn.Module:
    """Build the full autoencoder; its decoder's RNG use is part of N2 initialization.

    The factory is an entry point; the OSS package ships neither its registration nor
    the implementation. A source manifest that does not register it is an error, not an
    absent provider. Discovery and imports happen only when deformation construction
    requests this call. Provider import/construction errors propagate unchanged.
    """
    providers = find_entry_points(_GROUP, "hb1", missing_ok=False)
    if not providers:
        raise TactileUnsupportedError(
            "HB1 deformation requires the internal GR00T HB1 provider; it is not included in OSS."
        )
    if len(providers) != 1:
        raise RuntimeError(f"Expected one HB1 deformation provider, found {len(providers)}")
    return providers[0].load()()
