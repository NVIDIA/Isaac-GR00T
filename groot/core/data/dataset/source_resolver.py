# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolve configured dataset sources before dataset construction.

This module keeps external catalog lookups out of concrete dataset loaders.  A
dataset config can keep using plain paths, or it can point at a DeltaCAT table
whose rows contain concrete LeRobot or PackDS dataset roots.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import os
from typing import Any, Literal

from groot.core.data.dataset.path_detection import is_packds_path

DatasetSourceFormat = Literal["lerobot", "packds", "deltacat"]


@dataclass(frozen=True)
class ResolvedDatasetSource:
    """A concrete dataset root after resolving any catalog indirection."""

    uri: str
    format: DatasetSourceFormat
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_packds_path(path: str) -> bool:
    return is_packds_path(path)


def _infer_format(path: str | None = None, path_column: str | None = None) -> DatasetSourceFormat:
    if path is not None and str(path).startswith("dc://"):
        return "deltacat"
    if path is not None and _is_packds_path(path):
        return "packds"
    if path_column is not None and "packds" in path_column.lower():
        return "packds"
    return "lerobot"


def _normalize_format(value: Any, path_column: str | None = None) -> DatasetSourceFormat:
    if value is None:
        return _infer_format(path_column=path_column)
    normalized = str(value).lower()
    if normalized not in ("lerobot", "packds", "deltacat"):
        raise ValueError(
            f"Unsupported dataset source format {value!r}; "
            "expected 'lerobot', 'packds', or 'deltacat'"
        )
    return normalized  # type: ignore[return-value]


def _make_deltacat_client() -> Any:
    server_url = os.environ.get("DELTACAT_SERVER_URL", "").strip()
    bearer_token = os.environ.get("DELTACAT_BEARER_TOKEN", "").strip()
    if not server_url:
        raise RuntimeError("DELTACAT_SERVER_URL is required to resolve DeltaCAT dataset sources")
    if not bearer_token:
        raise RuntimeError("DELTACAT_BEARER_TOKEN is required to resolve DeltaCAT dataset sources")

    try:
        from deltacat_client import Client
    except ImportError as exc:
        raise RuntimeError(
            "deltacat-client is required to resolve DeltaCAT dataset sources"
        ) from exc

    timeout = float(os.environ.get("DELTACAT_CLIENT_TIMEOUT", "120"))
    return Client(server_url=server_url, bearer_token=bearer_token, timeout=timeout)


def _table_to_rows(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_pylist"):
        rows = table.to_pylist()
        return [dict(row) for row in rows]
    if hasattr(table, "to_dict"):
        return [dict(row) for row in table.to_dict(orient="records")]
    if isinstance(table, Mapping):
        columns = {str(key): list(value) for key, value in table.items()}
        if not columns:
            return []
        row_count = len(next(iter(columns.values())))
        return [{key: values[i] for key, values in columns.items()} for i in range(row_count)]
    return [dict(row) for row in table]


def _column_names(table: Any, rows: list[dict[str, Any]]) -> list[str]:
    names = getattr(table, "column_names", None)
    if names is not None:
        return list(names)
    if rows:
        return list(rows[0].keys())
    return []


def _row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = row.get(key)
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _default_path_column(source_format: DatasetSourceFormat, columns: list[str]) -> str:
    if source_format == "packds":
        candidates = ("packds_uri", "packds_path", "dataset_path", "path", "uri")
    elif source_format == "deltacat":
        candidates = ("deltacat_uri", "dc_uri", "dataset_path", "path", "uri")
    else:
        candidates = ("lerobot_path", "dataset_path", "path", "uri")
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "DeltaCAT dataset source table is missing a path column. "
        f"Tried {list(candidates)}; available columns are {columns}."
    )


def _deltacat_spec(source: Mapping[str, Any]) -> Mapping[str, Any] | None:
    nested = source.get("deltacat")
    if isinstance(nested, Mapping):
        return nested
    source_type = source.get("type") or source.get("source")
    if source_type == "deltacat":
        return source
    return None


def _resolve_deltacat_table(spec: Mapping[str, Any]) -> list[ResolvedDatasetSource]:
    namespace = spec.get("namespace")
    table_name = spec.get("table")
    if not namespace or not table_name:
        raise ValueError("DeltaCAT dataset sources require 'namespace' and 'table'")

    path_column = spec.get("path_column")
    source_format = _normalize_format(spec.get("format"), path_column=path_column)
    read_kwargs = dict(spec.get("read_kwargs") or {})
    read_kwargs.setdefault("read_as", "pyarrow")
    filters = spec.get("filters") or {}
    limit = spec.get("limit")
    if filters:
        read_kwargs.setdefault("filter_predicate", dict(filters))
    if limit is not None:
        read_kwargs.setdefault("limit", int(limit))

    table = _make_deltacat_client().catalog.read(
        namespace=str(namespace),
        table=str(table_name),
        **read_kwargs,
    )
    rows = _table_to_rows(table)
    if filters:
        rows = [row for row in rows if _row_matches_filters(row, filters)]

    if limit is not None:
        rows = rows[: int(limit)]

    columns = _column_names(table, rows)
    selected_column = str(path_column or _default_path_column(source_format, columns))
    if not rows:
        raise ValueError(
            f"DeltaCAT table {namespace}.{table_name} produced no dataset paths for column {selected_column!r}"
        )

    resolved: list[ResolvedDatasetSource] = []
    for row in rows:
        uri = row.get(selected_column)
        if not uri:
            raise ValueError(
                f"DeltaCAT table {namespace}.{table_name} has an empty {selected_column!r} value"
            )
        resolved.append(
            ResolvedDatasetSource(
                uri=str(uri),
                format=source_format,
                metadata={
                    "source": "deltacat",
                    "namespace": str(namespace),
                    "table": str(table_name),
                    "path_column": selected_column,
                },
            )
        )
    return resolved


def _resolve_packds_deltacat_locator(uri: str) -> list[ResolvedDatasetSource]:
    """Resolve a PackDS dc:// locator using the PackDS source resolver."""
    from groot.core.data.packds import sources as packds_sources

    resolved = packds_sources.resolve_source(uri, deltacat_client=_make_deltacat_client())
    return [
        ResolvedDatasetSource(
            uri=resolved.packds_uri,
            format="packds",
            metadata={"source": "deltacat", **resolved.metadata},
        )
    ]


def resolve_dataset_source(source: Any) -> list[ResolvedDatasetSource]:
    """Resolve one dataset source config item into concrete dataset roots."""
    if isinstance(source, Mapping):
        spec = _deltacat_spec(source)
        if spec is not None:
            return _resolve_deltacat_table(spec)
        path = source.get("path") or source.get("uri")
        if path is None:
            raise ValueError(f"Unsupported dataset source mapping: {source!r}")
        path_str = str(path)
        return [
            ResolvedDatasetSource(
                uri=path_str,
                format=(
                    _normalize_format(source.get("format"), path_column=None)
                    if source.get("format") is not None
                    else _infer_format(path=path_str)
                ),
                metadata={"source": "path"},
            )
        ]

    path_str = str(source)
    if path_str.startswith("dc://"):
        return [
            ResolvedDatasetSource(
                uri=path_str,
                format="deltacat",
                metadata={"source": "deltacat"},
            )
        ]
    return [
        ResolvedDatasetSource(
            uri=path_str,
            format=_infer_format(path=path_str),
            metadata={"source": "path"},
        )
    ]


def resolve_dataset_sources(sources: Iterable[Any]) -> list[ResolvedDatasetSource]:
    """Resolve all configured dataset sources, expanding table rows if needed."""
    resolved: list[ResolvedDatasetSource] = []
    for source in sources:
        resolved.extend(resolve_dataset_source(source))
    return resolved
