# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Synthesize the ``tactile``/``deform`` modality metadata of a LeRobot dataset.

Tactile datasets keep the per-finger F6 force slices under the ``state`` group of
meta/modality.json (``<finger>_f6``) and the deform-map videos under ``video``
(``tactile_deform_<finger>_freq20``). Training configs address them as separate
``tactile``/``deform`` modalities, so the loader derives those groups from the base ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from groot.core.data import types as data_types

# Tactile force reuses observation.state slices; deform maps reuse the video columns.
DERIVED_MODALITY_SOURCE_GROUPS = {"tactile": "state", "deform": "video"}


def derive_tactile_modality_meta(
    modality_meta: dict[str, dict[str, Any]],
    modality_configs: Mapping[str, data_types.ModalityConfig],
    *,
    dataset_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Add ``tactile``/``deform`` groups built from ``state``/``video`` to ``modality_meta``.

    A group is derived only when its modality is requested and the dataset does not
    already ship a non-empty group of that name. The derived group holds exactly the
    requested keys, in request order, and shares its entries with the source group.

    Args:
        modality_meta: Parsed meta/modality.json. Updated in place.
        modality_configs: Requested modality configs; only ``modality_keys`` is read.
        dataset_path: Named in the error message, to locate the dataset in a mixture.

    Returns:
        ``modality_meta`` itself, so the call works with or without reassignment.

    Raises:
        KeyError: If a requested tactile key is not in the ``state`` group.
    """
    for modality, source_group in DERIVED_MODALITY_SOURCE_GROUPS.items():
        if modality not in modality_configs:
            continue
        if modality_meta.get(modality):
            continue  # the dataset ships an explicit group
        source_meta = modality_meta.get(source_group, {})
        keys = modality_configs[modality].modality_keys
        missing = [key for key in keys if key not in source_meta]
        if missing and modality == "tactile":
            # Without a state slice the loader would only log a warning and drop that
            # finger's column, so the tactile window silently loses fingers.
            raise KeyError(
                f"Tactile keys {missing} are not in the modality.json '{source_group}' group. "
                f"Available keys: {list(source_meta)}. Dataset: {dataset_path}"
            )
        derived = {}
        for key in keys:
            if key in source_meta:
                derived[key] = source_meta[key]
            else:
                # Same canonical name the loader resolves when a video entry has no alias.
                derived[key] = {"original_key": f"observation.images.{key}"}
        if derived:
            modality_meta[modality] = derived
    return modality_meta
