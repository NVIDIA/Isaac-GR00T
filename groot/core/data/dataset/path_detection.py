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

``uses_lance`` decides whether a run is a LanceDS run. The run-level lance hooks (the launch stamp,
the pin check, the actors' environment, the fork check) and their imports are gated on it: the OSS
export ships no lance package.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

from groot.core.data.dataset.recipe_fields import LANCE_ENTRY_FIELDS

#: The data-level LanceDS fields: setting any makes uses_lance true (lance_pins checks only
#: lance_registry_as_of).
_DATA_LANCE_FIELDS = ("lance_datasets_root", "lance_step_filters", "lance_registry_as_of")


def is_lance_dataset_path(path: str | Path) -> bool:
    """Return True when ``path`` is a local LanceDS store.

    Stores are identified structurally: a directory containing the
    ``step.lance`` table (see groot/core/data/dataset/lance/read/format.py). A path that cannot be
    read is no store rather than an error (``os.path.isdir``): a run without LanceDS must not fail
    on a path it does not read itself, as the Ray driver does not read its datasets. An ``s3://``
    store is declared, not detected: :func:`is_lance_source`.
    """
    return os.path.isdir(os.path.join(path, "step.lance"))


def is_lance_source(path: str | Path, lance_fields: Mapping[str, Any] | None) -> bool:
    """Return True when a training source is read as a LanceDS store.

    ``lance_fields`` are the source's resolved LANCE_ENTRY_FIELDS (lance_pins.lance_fields), None
    where it has none. A local path is a store by its structure (:func:`is_lance_dataset_path`). An
    ``s3://`` path is never probed, so LeRobot on s3 costs no request: it is a store when its entry
    gives it a lance field (a pin, a view or a filter list).
    """
    if str(path).startswith("s3://"):
        return any(value is not None for value in (lance_fields or {}).values())
    return is_lance_dataset_path(path)


def uses_lance(data: Any) -> bool:
    """True when ``data`` (the run's ``DataConfig``) involves LanceDS.

    That is, a data-level lance field is set, an entry sets a lance field
    (lance_pins.check_lance_training_pins refuses one on any other path), a training path, once
    resolved, is a LanceDS store or a directory holding stores (which resolve_all_dataset_paths
    expands into one path per store), or a validation path is a store (build_val_datasets expands
    none). A mapping without a path (the vision-language sources some recipes list) names no store.
    The rows of a DeltaCAT table are read only at build: DatasetFactory.build refuses a store among
    them when this is False. Never raises on a path it cannot read (see is_lance_dataset_path).
    """
    # Here, not at the top: data_paths imports boto3, which this module's importers need not load.
    from groot.core.utils.storage.data_paths import (
        expand_lerobot_dataset_paths,
        resolve_dataset_path,
    )

    if any(getattr(data, field) is not None for field in _DATA_LANCE_FIELDS):
        return True
    for ds in data.datasets:
        if any(getattr(ds, field) is not None for field in LANCE_ENTRY_FIELDS):
            return True
        if ds.val_dataset_path is not None and is_lance_dataset_path(ds.val_dataset_path):
            return True
        for source in ds.dataset_paths:
            path = (
                source.get("path") or source.get("uri") if isinstance(source, Mapping) else source
            )
            if path is None:
                continue
            root = resolve_dataset_path(str(path))
            # resolve_all_dataset_paths expands neither remote URIs nor PackDS roots, and an s3://
            # store sets a lance field (above): no store. A .packds root may be a store itself.
            if str(root).startswith(("s3://", "dc://")):
                continue
            try:
                expanded = (
                    [root] if str(root).endswith(".packds") else expand_lerobot_dataset_paths(root)
                )
            except OSError:  # load-bearing: for is_lance_dataset_path's reason
                expanded = []
            if any(is_lance_dataset_path(p) for p in expanded):
                return True
    return False


def is_packds_path(path: str | Path) -> bool:
    """Return True when ``path`` should be treated as a PackDS dataset path.

    Recognises:
    - Paths ending with ``.packds``
    - Paths inside ``yam_packds_*`` buckets/directories
    - ``dc://`` DeltaCAT URIs (all DeltaCAT tables are PackDS-backed)

    A LanceDS store under such a name is still read as a store: check :func:`is_lance_source` first.
    """
    normalized = str(path).rstrip("/")
    if normalized.startswith("dc://"):
        return True
    if normalized.endswith(".packds"):
        return True

    parts = [part for part in normalized.split("/") if part]
    return any(part.startswith("yam_packds") for part in parts)
