# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Backend discovery and open helpers for PackDS readers.

`PackDSReader` uses the classes in this module to do two backend-specific tasks:

- discover candidate table representations under a PackDS path
- open a chosen representation as a readable Arrow dataset

All higher-level read logic stays in `PackDSReader`. Backends exist only to isolate
layout detection and dataset-opening behavior that differs by storage format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import lance
import pyarrow.dataset as pads
import pyarrow.fs as pafs

from groot.core.data.packds import constants, utils


@dataclass(frozen=True)
class ResolvedTable:
    """A concrete table representation discovered on disk or remote storage.

    Attributes:
        table_name: Logical table name, currently always ``"steps"``.
        backend_key: Registry key used to look up the backend implementation
            that knows how to open this representation.
        format_name: Canonical user-facing format name used for preference
            selection, e.g. ``"lance"`` or ``"parquet"``.
        uri: Fully resolved local path or remote URI for the table directory.
    """

    table_name: str
    backend_key: str
    format_name: str
    uri: str


class PackDSBackend:
    """Base class for PackDS read backends.

    A backend class is intentionally minimal. It is responsible only for:

    - discovering paths that match its on-disk layout
    - opening a discovered representation as a ``pyarrow.dataset.Dataset``

    The reader owns all generic logic such as filtering, projection, decoding,
    validation, and batch materialization.
    """

    backend_key: ClassVar[str]
    format_name: ClassVar[str]

    @classmethod
    def open_dataset(
        cls,
        resolved: ResolvedTable,
        *,
        storage_options: dict[str, str] | None,
    ) -> pads.Dataset:
        """Open a discovered representation as a readable Arrow dataset."""
        raise NotImplementedError

    @staticmethod
    def _join_uri(base: str | Path, child: str) -> str:
        """Join a child name onto a local path or remote URI."""
        base = str(base)
        if utils.is_remote(base):
            return base.rstrip("/") + "/" + child
        return str(Path(base) / child)

    @staticmethod
    def _table_dir_name(table_name: str, format_name: str) -> str:
        """Return the canonical directory name for a named table format."""
        return f"{table_name}.{format_name}"

    @staticmethod
    def _parse_named_table_dir(name: str) -> tuple[str, str] | None:
        """Parse ``{table_name}.{format}`` directory names."""
        parts = name.split(".")
        if len(parts) != 2:
            return None
        table_name, format_name = parts
        if format_name not in constants.TABLE_FORMATS:
            return None
        return table_name, format_name

    @staticmethod
    def _contains_only_files_with_suffix(path: Path, suffix: str) -> bool:
        """Check simple file-only directory layouts such as parquet."""
        if not path.is_dir():
            return False
        entries = list(path.iterdir())
        if not entries:
            return False
        return all(entry.is_file() and entry.name.endswith(suffix) for entry in entries)

    @classmethod
    def _direct_local_match(
        cls,
        path: Path,
        *,
        table_name: str,
    ) -> ResolvedTable | None:
        """Try to interpret *path* itself as this backend's table directory."""
        del path, table_name
        return None

    @classmethod
    def _child_local_matches(cls, root: Path, *, table_name: str) -> list[ResolvedTable]:
        """Discover child table directories below a PackDS root directory."""
        matches: list[ResolvedTable] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            resolved = cls._direct_local_match(child, table_name=table_name)
            if resolved is not None:
                matches.append(resolved)
        return matches

    @classmethod
    def discover(cls, path: str | Path, *, table_name: str) -> list[ResolvedTable]:
        """Discover candidate representations for this backend."""
        path_str = str(path)
        if utils.is_remote(path_str):
            return cls._discover_remote(path_str, table_name=table_name)

        path_obj = Path(path_str)
        direct = cls._direct_local_match(path_obj, table_name=table_name)
        if direct is not None:
            return [direct]
        if not path_obj.is_dir():
            return []
        return cls._child_local_matches(path_obj, table_name=table_name)

    @classmethod
    def _discover_remote(cls, path: str, *, table_name: str) -> list[ResolvedTable]:
        """Discover remote candidates for this backend."""
        raise NotImplementedError


class LanceBackend(PackDSBackend):
    """Backend for PackDS' canonical Lance-backed table layout.

    PackDS writes the LanceDB-managed table into ``steps.lance``. Even though
    the writable implementation is backed by the ``lancedb`` package, the
    on-disk representation is a Lance dataset directory, so the canonical
    PackDS format name is simply ``"lance"``.
    """

    backend_key = constants.TABLE_FORMAT_LANCE
    format_name = constants.TABLE_FORMAT_LANCE

    @staticmethod
    def _is_lance_dataset_layout(path: Path) -> bool:
        """Recognize a managed Lance dataset directory layout."""
        if not path.is_dir():
            return False
        child_names = {child.name for child in path.iterdir()}
        return {"data", "_transactions", "_versions"}.issubset(child_names)

    @classmethod
    def _direct_local_match(cls, path: Path, *, table_name: str) -> ResolvedTable | None:
        parsed = cls._parse_named_table_dir(path.name)
        if parsed is None:
            return None
        found_table_name, format_name = parsed
        if found_table_name != table_name or format_name != cls.format_name:
            return None
        if cls._is_lance_dataset_layout(path):
            return ResolvedTable(table_name, cls.backend_key, cls.format_name, str(path))
        return None

    @classmethod
    def _discover_remote(cls, path: str, *, table_name: str) -> list[ResolvedTable]:
        direct_name = path.rstrip("/").split("/")[-1]
        parsed = cls._parse_named_table_dir(direct_name)
        if parsed is not None:
            found_table_name, format_name = parsed
            if found_table_name == table_name and format_name == cls.format_name:
                return [ResolvedTable(table_name, cls.backend_key, cls.format_name, path)]
        return [
            ResolvedTable(
                table_name,
                cls.backend_key,
                cls.format_name,
                cls._join_uri(path, cls._table_dir_name(table_name, cls.format_name)),
            )
        ]

    @classmethod
    def open_dataset(
        cls,
        resolved: ResolvedTable,
        *,
        storage_options: dict[str, str] | None,
    ) -> pads.Dataset:
        kwargs = {}
        if storage_options is not None:
            kwargs["storage_options"] = storage_options
        return lance.dataset(resolved.uri, **kwargs)


class ParquetBackend(PackDSBackend):
    """Backend for ``steps.parquet`` directories.

    This backend exists because parquet tables can be opened through Arrow's
    generic dataset API without any Lance-specific dependency at read time.
    """

    backend_key = constants.TABLE_FORMAT_PARQUET
    format_name = constants.TABLE_FORMAT_PARQUET

    @classmethod
    def _is_parquet_layout(cls, path: Path) -> bool:
        return cls._contains_only_files_with_suffix(path, ".parquet")

    @classmethod
    def _direct_local_match(cls, path: Path, *, table_name: str) -> ResolvedTable | None:
        parsed = cls._parse_named_table_dir(path.name)
        if parsed is None:
            return None
        found_table_name, format_name = parsed
        if found_table_name != table_name or format_name != cls.format_name:
            return None
        if cls._is_parquet_layout(path):
            return ResolvedTable(table_name, cls.backend_key, cls.format_name, str(path))
        return None

    @classmethod
    def _discover_remote(cls, path: str, *, table_name: str) -> list[ResolvedTable]:
        direct_name = path.rstrip("/").split("/")[-1]
        parsed = cls._parse_named_table_dir(direct_name)
        if parsed is not None:
            found_table_name, format_name = parsed
            if found_table_name == table_name and format_name == cls.format_name:
                return [ResolvedTable(table_name, cls.backend_key, cls.format_name, path)]
        return [
            ResolvedTable(
                table_name,
                cls.backend_key,
                cls.format_name,
                cls._join_uri(path, cls._table_dir_name(table_name, cls.format_name)),
            )
        ]

    @classmethod
    def _filesystem_and_source(
        cls,
        uri: str,
        *,
        storage_options: dict[str, str] | None,
    ) -> tuple[pafs.FileSystem | None, str]:
        """Translate a parquet URI into Arrow dataset open arguments."""
        if not utils.is_remote(uri):
            return None, uri

        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise NotImplementedError(f"Unsupported remote parquet scheme: {parsed.scheme!r}")

        opts = storage_options or {}
        endpoint = opts.get("aws_endpoint") or None
        scheme = "http" if opts.get("allow_http") == "true" else "https"
        if endpoint and "://" in endpoint:
            endpoint = endpoint.split("://", 1)[1]

        filesystem = pafs.S3FileSystem(
            access_key=opts.get("aws_access_key_id"),
            secret_key=opts.get("aws_secret_access_key"),
            region=opts.get("aws_region") or None,
            endpoint_override=endpoint,
            scheme=scheme,
        )
        return filesystem, f"{parsed.netloc}{parsed.path}"

    @classmethod
    def open_dataset(
        cls,
        resolved: ResolvedTable,
        *,
        storage_options: dict[str, str] | None,
    ) -> pads.Dataset:
        filesystem, source = cls._filesystem_and_source(
            resolved.uri,
            storage_options=storage_options,
        )
        kwargs = {"format": "parquet"}
        if filesystem is not None:
            kwargs["filesystem"] = filesystem
        return pads.dataset(source, **kwargs)


# Registry consulted by ``PackDSReader`` after source resolution.
# Keys are table-backend identities, not canonical user-facing format names.
TABLE_BACKENDS: dict[str, type[PackDSBackend]] = {
    LanceBackend.backend_key: LanceBackend,
    ParquetBackend.backend_key: ParquetBackend,
}

# Backward-compatible alias for older imports.
BACKENDS = TABLE_BACKENDS
