# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Training extensions provided outside groot.core.

``training.extensions`` maps an extension name to its options. Each name resolves
to a provider registered under the ``groot.training.extensions`` entry-point group.
The provider is called with the options and returns a ``TrainingExtension``. A run
that names an extension this install does not provide fails when its config is
validated, before any model or dataset is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from groot.core.utils.entry_points import find_entry_points

if TYPE_CHECKING:
    from transformers import TrainerCallback

    from groot.core.training.configs.base_config import Config

_GROUP = "groot.training.extensions"


class TrainingExtension(Protocol):
    def checkpoint_callbacks(self, *, config: Config, trainer: Any) -> list[TrainerCallback]:
        """Callbacks that read each saved checkpoint.

        They are registered before the checkpoint upload callbacks, which may
        delete the local checkpoint.
        """


def load_training_extensions(
    extensions: Mapping[str, Mapping[str, Any]],
) -> dict[str, TrainingExtension]:
    return {name: _provider(name)(dict(options)) for name, options in extensions.items()}


def _provider(name: str):
    providers = find_entry_points(_GROUP, name, missing_ok=True)
    if not providers:
        raise ValueError(
            f"training.extensions.{name} is not available in this install: no provider is "
            f"registered under the {_GROUP!r} entry-point group. Remove it from the config."
        )
    if len(providers) != 1:
        raise RuntimeError(
            f"Expected one {name!r} training extension provider, found {len(providers)}"
        )
    return providers[0].load()
