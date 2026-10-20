# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Fields of a ``data.datasets`` entry that hold a value per dataset path, resolved one way.

The trainer (``DataConfig`` and ``DatasetFactory.build``) and the lance write tools
(``lance/write/recipe_entries.py``) call the same functions, so a recipe the trainer refuses is
refused by the tools too, and one it accepts resolves to the same value per path. Standard library
only: the write tools import this on CPU pods, in a venv without the training stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The per-entry LanceDS fields, which the factory resolves per source: for the loader, and to
#: refuse them on a path that does not read them (a field missed here would be dropped unrefused).
LANCE_ENTRY_FIELDS = (
    "lance_view_id",
    "lance_as_of",
    "lance_step_version",
    "lance_media_version",
    "lance_episodes_version",
    "lance_step_filter",
)


def per_source(mapping: Mapping, source_uris: list[str], *, field: str) -> dict:
    """{uri: value} for the sources a {dataset_path: value} entry field names.

    A key matches a uri exactly or as a path suffix (launch paths are absolute, yaml keys
    relative); uris no key names are absent (unselected). A key naming no source or several, or
    a source named twice, is refused: each would apply a selection the yaml did not mean.
    """
    resolved = {}
    bad = []
    for key, value in mapping.items():
        hits = [u for u in source_uris if u == key or str(u).endswith("/" + key)]
        if len(hits) != 1:
            bad.append(key)
        elif hits[0] in resolved:
            raise ValueError(f"{field} has two keys for {hits[0]!r}; keep one")
        else:
            resolved[hits[0]] = value
    if bad:
        raise ValueError(
            f"{field} keys not matching exactly one of this entry's dataset_paths:"
            f" {sorted(bad)} (dataset_paths: {list(source_uris)})"
        )
    # '' is neither null nor an omitted key; it is almost certainly a broken template
    # interpolation -- refuse it.
    empty = sorted(k for k, v in mapping.items() if v is not None and not v)
    if empty:
        raise ValueError(f"{field} has empty values for: {empty}")
    return resolved


def expand_filter_sets(value: Any, sets: Mapping[str, Any], dataset_paths: Any) -> Any:
    """An entry's ``lance_step_filter`` with each filter-set name replaced by the list it names.

    ``value`` is a filter list, the name of a set in ``sets`` (``data.lance_step_filters``), or a
    ``{dataset_path: either}`` mapping; ``dataset_paths`` names the entry in the refusal of a set
    ``sets`` does not define.
    """
    per_path = dict(value) if isinstance(value, dict) else {None: value}
    for key, name in per_path.items():
        if isinstance(name, str):
            if name not in sets:
                raise ValueError(
                    f"dataset {dataset_paths}: lance_step_filter names {name!r}, "
                    f"which lance_step_filters does not define (known: {sorted(sets)})"
                )
            per_path[key] = list(sets[name])
    return per_path if isinstance(value, dict) else per_path[None]
