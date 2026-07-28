"""SQL-filtered DeltaCAT PackDS sampling utilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import os
import random
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
import polars as pl

from groot.core.data.packds import constants
from groot.core.data.packds.video import get_video_codec

DEFAULT_DELTACAT_BETA_SERVER_URL = (
    "http://gear-data-infra-iris-beta-alb-1753350942.us-east-2.elb.amazonaws.com"
)
DEFAULT_STEPS_URI = "dc://default/xdof/yam_airplane_post_training_steps"
SOURCE_VIDEO_BLOB_HEADER = "PACKDS_SOURCE_VIDEO_BLOB_REFERENCE_V1"
CHUNK_SIDECAR_BLOB_HEADER = "PACKDS_CHUNK_SIDECAR_REFERENCE_V1"


_BORROWED_CLIENT_LOCK = threading.Lock()
_AMBIENT_S3_ENDPOINT_LOCK = threading.Lock()
_RETRYABLE_CATALOG_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Max episode ids per ``episode_id IN (...)`` metadata read. The ids ride in the
# GET query string, so an unbounded list over a large dataset builds a multi-MB
# URL the catalog server rejects with HTTP 414. Ids carry a long task prefix
# (~90 URL-encoded chars each), so 32 keeps the URL a few KB, under the server cap.
_STEP_COUNT_ID_CHUNK = 32

# Env-gated per-shard build instrumentation (GROOT_PACKDS_SHARD_TIMING=1).
# Accumulates fetch vs decode seconds (SUMMED across decode threads) and the
# number of cache-miss fetches, so a caller can attribute shard-build cost to
# S3 fetch vs CPU decode vs cache misses. Times sum across concurrent threads,
# so they exceed wall time -- the RATIO (fetch:decode) and counts are the signal.
_TIMING_LOCK = threading.Lock()
_TIMING = {"fetch_s": 0.0, "decode_s": 0.0, "n_fetch": 0, "fetch_bytes": 0}


def _shard_timing_enabled() -> bool:
    return os.environ.get("GROOT_PACKDS_SHARD_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _timing_add(
    *, fetch_s: float = 0.0, decode_s: float = 0.0, n_fetch: int = 0, fetch_bytes: int = 0
) -> None:
    with _TIMING_LOCK:
        _TIMING["fetch_s"] += fetch_s
        _TIMING["decode_s"] += decode_s
        _TIMING["n_fetch"] += n_fetch
        _TIMING["fetch_bytes"] += fetch_bytes


def timing_snapshot_reset() -> dict:
    with _TIMING_LOCK:
        snap = dict(_TIMING)
        _TIMING["fetch_s"] = 0.0
        _TIMING["decode_s"] = 0.0
        _TIMING["n_fetch"] = 0
        _TIMING["fetch_bytes"] = 0
        return snap


def _s3_client_config() -> Any:
    """Botocore Config for the video-blob S3 clients.

    The video sidecars are fetched as MANY small per-frame/per-chunk GET
    requests, decoded in parallel across ``DELTACAT_PACKDS_DECODE_WORKERS``
    threads that all SHARE one cached client. botocore's default connection
    pool is only 10, so it -- not the thread count -- caps how many GETs run
    concurrently; with thousands of blobs per shard that throttles shard
    builds and starves the GPU. Size the pool to the decode parallelism (env
    ``DELTACAT_PACKDS_S3_MAX_POOL_CONNECTIONS``, default 64).
    """
    from botocore.config import Config

    pool = int(os.environ.get("DELTACAT_PACKDS_S3_MAX_POOL_CONNECTIONS", "64") or 64)
    return Config(
        # H100 CI exports AWS_ENDPOINT_URL for PDX/MSC access. S3 Express
        # CreateSession is resolved lazily during get_object, so suppressing the
        # env only during client construction is not enough.
        ignore_configured_endpoint_urls=True,
        max_pool_connections=max(10, pool),
        retries={"max_attempts": 5, "mode": "adaptive"},
    )


def _ambient_s3_fallback_disabled() -> bool:
    """Whether to refuse ambient AWS creds for blob reads.

    When set, forces the in-plan vended-lease path so the managed-cred path can
    be validated locally without silently depending on ~/.aws / IMDS.
    """
    return os.environ.get("DELTACAT_DISABLE_AMBIENT_S3_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _ignore_vended_s3_express_endpoint_enabled() -> bool:
    """Whether H100 reads should bypass non-resolvable PDX S3 Express endpoints."""
    return os.environ.get(
        "DELTACAT_PACKDS_IGNORE_VENDED_S3_EXPRESS_ENDPOINT", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }


def _is_s3_express_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in {"s3", "s3a"} and parsed.netloc.endswith("--x-s3")


def _is_pdx_s3_endpoint(endpoint_url: str | None) -> bool:
    if not endpoint_url:
        return False
    host = urlparse(endpoint_url).hostname or endpoint_url.split("/", 1)[0]
    return host == "pdx.s8k.io" or host.endswith(".pdx.s8k.io")


@contextmanager
def _without_ambient_s3_endpoint_urls():
    """Temporarily remove global S3 endpoint overrides for DeltaCAT-owned reads.

    CI exports AWS_ENDPOINT_URL/S3_ENDPOINT_URL for PDX/MSC access. DeltaCAT
    read plans vend their own S3 authority, including S3 Express routing; if
    Lance/object_store or boto3 sees the ambient endpoint it can route an
    S3 Express bucket through PDX and sign with the wrong region.
    """
    with _AMBIENT_S3_ENDPOINT_LOCK:
        saved = {
            key: os.environ.pop(key)
            for key in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "S3_ENDPOINT_URL")
            if key in os.environ
        }
        try:
            yield
        finally:
            os.environ.update(saved)


@dataclass(frozen=True)
class DeltaCatTableRef:
    """Parsed DeltaCAT table URI."""

    catalog: str
    namespace: str
    table: str
    table_version: str | None = None

    @classmethod
    def from_uri(cls, uri: str) -> DeltaCatTableRef:
        parsed = urlparse(uri)
        if parsed.scheme != "dc":
            raise ValueError(f"Expected a dc:// URI, got {uri!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if not parsed.netloc or len(parts) < 2:
            raise ValueError(
                "DeltaCAT URIs must be dc://<catalog>/<namespace>/<table>[/<table_version>]"
            )
        if len(parts) > 3:
            raise ValueError(
                f"DeltaCAT table refs support at most one table_version path segment: {uri!r}"
            )
        return cls(
            catalog=parsed.netloc,
            namespace=parts[0],
            table=parts[1],
            table_version=parts[2] if len(parts) == 3 else None,
        )


@dataclass(frozen=True)
class VideoBlobReference:
    """Inline MP4 bytes or an external source-video/chunk object reference.

    DeltaCAT rows can store a full source-video object reference or a chunk
    sidecar byte-range reference. The reader treats both as object descriptors,
    then deduplicates and caches by object URI, range, size, and optional hash
    before fetching or decoding.
    """

    camera: str
    codec: str
    byte_size: int
    sha256: str | None = None
    object_uri: str | None = None
    object_byte_range_start: int | None = None
    object_byte_range_end: int | None = None
    raw_reference: dict[str, Any] = field(default_factory=dict)
    inline_blob: bytes | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_cell(cls, camera: str, value: bytes | bytearray | memoryview) -> VideoBlobReference:
        raw = bytes(value)
        header_bytes, separator, payload_bytes = raw.partition(b"\n")
        header = header_bytes.decode("utf-8", errors="replace") if separator else ""
        if not separator:
            return cls(
                camera=camera,
                codec="h264",
                byte_size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                inline_blob=raw,
            )
        if header not in {SOURCE_VIDEO_BLOB_HEADER, CHUNK_SIDECAR_BLOB_HEADER}:
            if header.startswith("PACKDS_") and header.endswith("_REFERENCE_V1"):
                raise ValueError(
                    "DeltaCAT PackDS video blob reference cells must use "
                    f"{SOURCE_VIDEO_BLOB_HEADER!r} or {CHUNK_SIDECAR_BLOB_HEADER!r}."
                )
            return cls(
                camera=camera,
                codec="h264",
                byte_size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                inline_blob=raw,
            )

        payload = json.loads(payload_bytes.decode("utf-8"))
        return cls._from_reference_payload(camera, header, payload)

    @classmethod
    def _from_reference_payload(
        cls,
        camera: str,
        header: str,
        payload: dict[str, Any],
    ) -> VideoBlobReference:
        object_uri = payload.get("uri") or payload.get("object_uri")
        if not object_uri:
            raise ValueError(f"{header} payload is missing 'uri'.")

        range_start: int | None = None
        range_end: int | None = None
        if header == CHUNK_SIDECAR_BLOB_HEADER:
            start_value = payload.get("object_byte_range_start")
            end_value = payload.get("object_byte_range_end")
            if start_value is None or end_value is None:
                raise ValueError(
                    f"{CHUNK_SIDECAR_BLOB_HEADER} payload is missing object byte range fields."
                )
            range_start = int(start_value)
            range_end = int(end_value)
            if range_start < 0 or range_end < range_start:
                raise ValueError(
                    f"{CHUNK_SIDECAR_BLOB_HEADER} payload has invalid byte range "
                    f"{range_start}-{range_end}."
                )

        byte_size = payload.get("size", payload.get("byte_size"))
        if byte_size is None:
            if range_start is not None and range_end is not None:
                byte_size = range_end - range_start + 1
            else:
                raise ValueError(f"{header} payload is missing 'size'.")
        return cls(
            camera=camera,
            codec=str(payload.get("codec") or "h264"),
            byte_size=int(byte_size),
            sha256=payload.get("sha256"),
            object_uri=str(object_uri),
            object_byte_range_start=range_start,
            object_byte_range_end=range_end,
            raw_reference=payload,
        )

    @property
    def cache_key(self) -> str:
        if self.inline_blob is not None:
            return f"inline:{self.sha256}:{self.byte_size}"
        if self.object_byte_range_start is None:
            byte_range = "full"
        else:
            byte_range = f"{self.object_byte_range_start}-{self.object_byte_range_end}"
        shape_hash = self.raw_reference.get("chunk_shape_hash", "")
        return (
            f"s3-source:{self.object_uri}:{byte_range}:{self.byte_size}:{self.sha256 or shape_hash}"
        )


@dataclass(frozen=True)
class DeltaCatDecodedStep:
    """One DeltaCAT step row with decoded RGB video frames."""

    row: dict[str, Any]
    decoded_video: dict[str, np.ndarray]
    video_references: dict[str, VideoBlobReference]

    @property
    def episode_id(self) -> str:
        return str(self.row["episode_id"])

    @property
    def step_index(self) -> int:
        return int(self.row["step_index"])


@dataclass(frozen=True)
class DeltaCatDecodedEpisode:
    """All decoded steps returned for one episode."""

    episode_id: str
    steps: list[DeltaCatDecodedStep]


class _ObjectLRUCache:
    """Tiny byte-weighted LRU cache."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            value, weight = item
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: Any, weight: int) -> None:
        with self._lock:
            if self._max_bytes <= 0 or weight > self._max_bytes:
                return
            prior = self._items.pop(key, None)
            if prior is not None:
                self._bytes -= prior[1]
            self._items[key] = (value, weight)
            self._bytes += weight
            while self._bytes > self._max_bytes and self._items:
                _old_key, (_old_value, old_weight) = self._items.popitem(last=False)
                self._bytes -= old_weight


def app_episodes_table_uri(steps_table_uri: str) -> str:
    """Return the application `_episodes` sibling URI for a `_steps` table URI."""
    table_ref = DeltaCatTableRef.from_uri(steps_table_uri)
    if not table_ref.table.endswith(constants.STEPS_TABLE_SUFFIX):
        raise ValueError(f"Steps table name must end with {constants.STEPS_TABLE_SUFFIX!r}")
    episodes_table = (
        table_ref.table[: -len(constants.STEPS_TABLE_SUFFIX)] + constants.APP_EPISODES_TABLE_SUFFIX
    )
    uri = f"dc://{table_ref.catalog}/{table_ref.namespace}/{episodes_table}"
    if table_ref.table_version is not None:
        return f"{uri}/{table_ref.table_version}"
    return uri


def _normalize_as_of_timestamp(as_of_timestamp: int | None) -> int | None:
    """Return DeltaCAT's expected epoch-nanosecond `as_of` value.

    Configs record nanoseconds, but this accepts common epoch second,
    millisecond, and microsecond values so hand-written YAML does not point
    DeltaCAT at a pre-table snapshot.
    """
    if as_of_timestamp is None:
        return None
    value = int(as_of_timestamp)
    magnitude = abs(value)
    if magnitude < 100_000_000_000:
        return value * 1_000_000_000
    if magnitude < 100_000_000_000_000:
        return value * 1_000_000
    if magnitude < 100_000_000_000_000_000:
        return value * 1_000
    return value


def _apply_sql(lf: pl.LazyFrame, sql: str | None, *, table_name: str) -> pl.LazyFrame:
    if sql is None or not sql.strip():
        return lf

    query = sql.strip()
    if query.lower().startswith("where "):
        query = query[6:].strip()

    first_token = query.split(maxsplit=1)[0].lower() if query else ""
    if first_token in {"select", "with"}:
        context = pl.SQLContext()
        context.register(table_name, lf)
        context.register("self", lf)
        result = context.execute(query, eager=False)
        return result if isinstance(result, pl.LazyFrame) else result.lazy()

    return lf.filter(pl.sql_expr(query))


_SQL_IDENTIFIER_EXCLUDE = {
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "desc",
    "else",
    "end",
    "false",
    "from",
    "group",
    "in",
    "is",
    "limit",
    "not",
    "null",
    "or",
    "order",
    "select",
    "self",
    "steps",
    "then",
    "true",
    "when",
    "where",
    "with",
}


def _sql_referenced_columns(sql: str | None) -> list[str]:
    """Best-effort SQL identifier extraction for projected predicate columns."""
    if sql is None or not sql.strip():
        return []
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql)
    columns: list[str] = []
    for match in re.finditer(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_:\-.]*)", without_strings):
        token = match.group(1)
        if token.lower() in _SQL_IDENTIFIER_EXCLUDE:
            continue
        if match.end() < len(without_strings) and without_strings[match.end()] == "(":
            continue
        columns.append(token)
    return list(dict.fromkeys(columns))


def _is_valid_action_steps_query(sql: str | None) -> bool:
    """Whether ``sql`` is exactly the canonical "keep valid-action steps" filter.

    The companion ``_episodes`` table aggregates ``num_valid_action_steps =
    sum(has_valid_action)`` per episode at write time, so a step filter that
    selects exactly the ``has_valid_action`` rows has its per-episode row count
    already materialized there. Recognizing that filter lets the loader read the
    aggregate instead of scanning every step row to count. Only the bare column,
    ``= true`` and ``== true`` forms are accepted; any compound predicate
    (``has_valid_action = true AND ...``) falls through to the exact count read.
    """
    if sql is None:
        return False
    query = sql.strip()
    if query.lower().startswith("where "):
        query = query[6:].strip()
    query = query.strip().strip("()").strip()
    # The column token must match exactly (lowercase `has_valid_action`); only the
    # `true` literal is case-insensitive. This keeps the fast path in lockstep with
    # the exact `_apply_sql` Polars filter, whose column names are case-sensitive --
    # an uppercase column errors there, so it must not silently succeed here.
    return re.fullmatch(r"has_valid_action(\s*={1,2}\s*(?i:true))?", query) is not None


def _step_index_plan_predicate(sql: str | None) -> dict[str, Any] | None:
    """Translate common bounded smoke filters into DeltaCAT's JSON predicate."""
    if sql is None:
        return None
    query = sql.strip()
    if query.lower().startswith("where "):
        query = query[6:].strip()
    match = re.fullmatch(r"step_index\s*<\s*(\d+)", query, flags=re.IGNORECASE)
    if match:
        return {"lt": ["step_index", int(match.group(1))]}
    match = re.fullmatch(r"step_index\s*<=\s*(\d+)", query, flags=re.IGNORECASE)
    if match:
        return {"lte": ["step_index", int(match.group(1))]}
    return None


def _and_filter_predicates(*predicates: dict[str, Any] | None) -> dict[str, Any] | None:
    selected = [predicate for predicate in predicates if predicate is not None]
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return {"and": selected}


def _ensure_lazyframe(result: Any) -> pl.LazyFrame:
    if isinstance(result, pl.LazyFrame):
        return result
    if isinstance(result, pl.DataFrame):
        return result.lazy()
    if hasattr(result, "schema") and hasattr(result, "to_batches"):
        return pl.from_arrow(result).lazy()
    raise TypeError(f"Expected a Polars or Arrow table result, got {type(result)!r}")


def _resolve_token() -> str:
    for name in ("DELTACAT_BEARER_TOKEN", "DELTACAT_WORKER_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    raise RuntimeError(
        "Set DELTACAT_BEARER_TOKEN or DELTACAT_WORKER_TOKEN before reading DeltaCAT tables."
    )


def _weighted_size(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, list) and all(isinstance(item, np.ndarray) for item in value):
        return sum(int(item.nbytes) for item in value)
    return 1


def _stable_json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _cache_ttl_fresh(cached_at: float, ttl_s: float) -> bool:
    return ttl_s <= 0 or (time.monotonic() - cached_at) <= ttl_s


def _is_retryable_catalog_exception(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in _RETRYABLE_CATALOG_STATUS_CODES:
        return True

    exc_name = type(exc).__name__.lower()
    return any(
        token in exc_name
        for token in (
            "timeout",
            "transport",
            "connection",
            "connecterror",
            "readerror",
            "remoteprotocolerror",
        )
    )


def _is_retryable_blob_exception(exc: BaseException) -> bool:
    """Return whether an S3 video-blob read failed in a retryable way."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:  # pragma: no cover - botocore is present with DeltaCAT.
        BotoCoreError = ()  # type: ignore[assignment]
        ClientError = ()  # type: ignore[assignment]

    if isinstance(exc, ClientError):
        response = getattr(exc, "response", {}) or {}
        metadata = response.get("ResponseMetadata", {}) or {}
        status_code = metadata.get("HTTPStatusCode")
        error_code = (response.get("Error", {}) or {}).get("Code")
        if status_code in _RETRYABLE_CATALOG_STATUS_CODES:
            return True
        return str(error_code).lower() in {
            "internalerror",
            "requesttimeout",
            "slowdown",
            "throttling",
            "throttlingexception",
            "toomanyrequests",
            "serviceunavailable",
        }

    if isinstance(exc, BotoCoreError):
        return True

    exc_text = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in exc_text
        for token in (
            "temporary failure in name resolution",
            "endpointconnectionerror",
            "connection",
            "connect",
            "dns",
            "timeout",
            "read timed out",
            "connection reset",
            "connection aborted",
        )
    )


class PackDSStepsReader:
    """Read SQL-filtered DeltaCAT PackDS steps with decoded video frames."""

    def __init__(
        self,
        table: str = DEFAULT_STEPS_URI,
        *,
        episode_query: str | None = None,
        steps_query: str | None = None,
        as_of_timestamp: int | None = None,
        seed: int | None = None,
        episodes_table_uri: str | None = None,
        cameras: Sequence[str] | None = None,
        step_columns: Sequence[str] | None = None,
        key_remap: dict[str, str] | None = None,
        server_url: str | None = None,
        timeout_s: int = 120,
        max_blob_cache_bytes: int = 256 * 1024 * 1024,
        max_decoded_cache_bytes: int = 512 * 1024 * 1024,
        decode_workers: int = 1,
    ) -> None:
        self.steps_table_ref = DeltaCatTableRef.from_uri(table)
        self.episodes_table_ref = DeltaCatTableRef.from_uri(
            episodes_table_uri or app_episodes_table_uri(table)
        )
        self.episode_query = episode_query
        self.steps_query = steps_query
        self.as_of_timestamp = _normalize_as_of_timestamp(as_of_timestamp)
        self.cameras = tuple(cameras) if cameras is not None else None
        self.step_columns = tuple(step_columns) if step_columns is not None else None
        self.key_remap = {str(key): str(value) for key, value in (key_remap or {}).items()}
        self.timeout_s = timeout_s
        self._rng = random.Random(seed)
        self._episode_ids: list[str] | None = None
        self._episodes_metadata: list[dict[str, Any]] | None = None
        self._server_url = server_url
        self._max_blob_cache_bytes = max_blob_cache_bytes
        self._blob_cache = _ObjectLRUCache(max_blob_cache_bytes)
        self._max_decoded_cache_bytes = max_decoded_cache_bytes
        self._decoded_cache = _ObjectLRUCache(max_decoded_cache_bytes)
        # Bytes-bounded LRU of collected+forward-filled episode ROWS (no video
        # bytes) keyed by episode_id, so trajectory windows re-use an episode's
        # rows instead of re-planning/re-reading the catalog per window. Env
        # DELTACAT_PACKDS_MAX_EPISODE_ROWS_CACHE_BYTES (default 4 GiB) -- the rows
        # are small, so this typically holds the whole table's episodes.
        self._max_episode_rows_cache_bytes = int(
            os.environ.get("DELTACAT_PACKDS_MAX_EPISODE_ROWS_CACHE_BYTES", str(4 * 1024**3)) or 0
        )
        self._episode_rows_cache_ttl_s = float(
            os.environ.get("DELTACAT_PACKDS_EPISODE_ROWS_CACHE_TTL_S", "1800") or 1800
        )
        self._episode_rows_cache = (
            _ObjectLRUCache(self._max_episode_rows_cache_bytes)
            if self._max_episode_rows_cache_bytes > 0
            else None
        )
        self._max_read_plan_cache_entries = int(
            os.environ.get("DELTACAT_PACKDS_MAX_READ_PLAN_CACHE_ENTRIES", "4096") or 0
        )
        self._read_plan_cache_ttl_s = float(
            os.environ.get("DELTACAT_PACKDS_READ_PLAN_CACHE_TTL_S", "1800") or 1800
        )
        # Plan calls are catalog-bound and idempotent. Cache them per process so
        # repeated shard loads of the same episode do not hammer /plan. The TTL
        # keeps vended plan credentials from being reused indefinitely.
        self._read_plan_cache = (
            _ObjectLRUCache(self._max_read_plan_cache_entries)
            if self._max_read_plan_cache_entries > 0
            else None
        )
        self._decode_workers = max(1, int(decode_workers))
        self._ambient_s3_client: Any | None = None
        self._ambient_s3_client_pid: int | None = None
        self._ambient_s3_client_identity: tuple[Any, ...] | None = None
        self._borrowed_s3_client: Any | None = None
        self._borrowed_s3_client_pid: int | None = None
        self._borrowed_s3_client_plan: Any | None = None
        self._s3_express_no_endpoint_client: Any | None = None
        self._s3_express_no_endpoint_client_pid: int | None = None
        self._s3_express_no_endpoint_client_plan: Any | None = None
        self._s3_express_no_endpoint_client_identity: tuple[Any, ...] | None = None
        self._client: Any | None = None
        self._client_pid: int | None = None

    def _make_client(self, *, server_url: str | None) -> Any:
        try:
            from deltacat_client import Client
        except ImportError as exc:
            raise RuntimeError("Install deltacat-client to use PackDSStepsReader.") from exc

        resolved_server_url = (
            server_url
            or os.environ.get("DELTACAT_SERVER_URL")
            or os.environ.get("DELTACAT_BETA_SERVER_URL")
            or DEFAULT_DELTACAT_BETA_SERVER_URL
        )
        kwargs = {
            "server_url": resolved_server_url,
            "bearer_token": _resolve_token(),
            "timeout": self.timeout_s,
            "catalog": self.steps_table_ref.catalog,
            "namespace": self.steps_table_ref.namespace,
        }
        client_params = inspect.signature(Client).parameters
        return Client(**{key: value for key, value in kwargs.items() if key in client_params})

    def _ensure_client(self) -> Any:
        pid = os.getpid()
        if self._client is None or self._client_pid != pid:
            self._client = self._make_client(server_url=self._server_url)
            self._client_pid = pid
        return self._client

    def _catalog_retry_attempts(self) -> int:
        return max(1, int(os.environ.get("DELTACAT_PACKDS_CATALOG_MAX_ATTEMPTS", "12") or 12))

    def _catalog_retry_base_sleep_s(self) -> float:
        return max(
            0.0, float(os.environ.get("DELTACAT_PACKDS_CATALOG_RETRY_BASE_SLEEP_S", "2.0") or 2.0)
        )

    def _catalog_retry_max_sleep_s(self) -> float:
        return max(
            0.0, float(os.environ.get("DELTACAT_PACKDS_CATALOG_RETRY_MAX_SLEEP_S", "30.0") or 30.0)
        )

    def _blob_retry_attempts(self) -> int:
        return max(1, int(os.environ.get("DELTACAT_PACKDS_BLOB_MAX_ATTEMPTS", "8") or 8))

    def _blob_retry_base_sleep_s(self) -> float:
        return max(
            0.0, float(os.environ.get("DELTACAT_PACKDS_BLOB_RETRY_BASE_SLEEP_S", "0.5") or 0.5)
        )

    def _blob_retry_max_sleep_s(self) -> float:
        return max(
            0.0, float(os.environ.get("DELTACAT_PACKDS_BLOB_RETRY_MAX_SLEEP_S", "8.0") or 8.0)
        )

    def _reset_cached_s3_clients(self) -> None:
        self._ambient_s3_client = None
        self._ambient_s3_client_pid = None
        self._ambient_s3_client_identity = None
        self._borrowed_s3_client = None
        self._borrowed_s3_client_pid = None
        self._borrowed_s3_client_plan = None
        self._s3_express_no_endpoint_client = None
        self._s3_express_no_endpoint_client_pid = None
        self._s3_express_no_endpoint_client_plan = None
        self._s3_express_no_endpoint_client_identity = None

    def _with_catalog_retries(self, label: str, call: Any) -> Any:
        attempts = self._catalog_retry_attempts()
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_catalog_exception(exc):
                    raise
                sleep_s = min(
                    self._catalog_retry_max_sleep_s(),
                    self._catalog_retry_base_sleep_s() * (2 ** (attempt - 1)),
                )
                if sleep_s > 0:
                    sleep_s *= 0.75 + random.random() * 0.5
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                status = f" status={status_code}" if status_code is not None else ""
                print(  # noqa: T201
                    f"[deltacat-packds] retrying {label} after {type(exc).__name__}{status} "
                    f"(attempt {attempt}/{attempts}, sleep={sleep_s:.1f}s)",
                    flush=True,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
                # Build a fresh client after transport/server-side errors; this
                # avoids reusing an httpx connection that may have been dropped
                # by the ALB while keeping the retry scoped to one idempotent
                # catalog request.
                self._client = None
                self._client_pid = None

        raise AssertionError("unreachable")

    def _with_blob_retries(self, label: str, call: Any) -> bytes:
        attempts = self._blob_retry_attempts()
        for attempt in range(1, attempts + 1):
            try:
                return call()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_blob_exception(exc):
                    raise
                sleep_s = min(
                    self._blob_retry_max_sleep_s(),
                    self._blob_retry_base_sleep_s() * (2 ** (attempt - 1)),
                )
                if sleep_s > 0:
                    sleep_s *= 0.75 + random.random() * 0.5
                print(  # noqa: T201
                    f"[deltacat-packds] retrying {label} after {type(exc).__name__}: "
                    f"{str(exc)[:300]} (attempt {attempt}/{attempts}, sleep={sleep_s:.1f}s)",
                    flush=True,
                )
                self._reset_cached_s3_clients()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        raise AssertionError("unreachable")

    def _catalog_read(self, **kwargs: Any) -> Any:
        def call() -> Any:
            read = self._ensure_client().catalog.read
            read_params = inspect.signature(read).parameters
            if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in read_params.values()):
                return read(**kwargs)
            return read(**{key: value for key, value in kwargs.items() if key in read_params})

        with _without_ambient_s3_endpoint_urls():
            return self._with_catalog_retries("catalog.read", call)

    def _catalog_query_partition_stats(self, **kwargs: Any) -> Any:
        def call() -> Any:
            query_partition_stats = self._ensure_client().catalog.query_partition_stats
            query_params = inspect.signature(query_partition_stats).parameters
            if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in query_params.values()):
                return query_partition_stats(**kwargs)
            return query_partition_stats(
                **{key: value for key, value in kwargs.items() if key in query_params}
            )

        return self._with_catalog_retries("catalog.query_partition_stats", call)

    def _catalog_plan(self, **kwargs: Any) -> Any:
        def call() -> Any:
            plan = self._ensure_client().catalog.plan
            plan_params = inspect.signature(plan).parameters
            if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in plan_params.values()):
                return plan(**kwargs)
            return plan(**{key: value for key, value in kwargs.items() if key in plan_params})

        return self._with_catalog_retries("catalog.plan", call)

    def _catalog_plan_cached(self, **kwargs: Any) -> Any:
        cache = self._read_plan_cache
        if cache is None:
            return self._catalog_plan(**kwargs)
        cache_key = _stable_json_key(kwargs)
        cached = cache.get(cache_key)
        if cached is not None:
            plan, cached_at = cached
            if _cache_ttl_fresh(float(cached_at), self._read_plan_cache_ttl_s):
                return plan

        plan = self._catalog_plan(**kwargs)
        cache.put(cache_key, (plan, time.monotonic()), 1)
        return plan

    def _cached_episode_rows(self, episode_id: str) -> tuple[Any, pl.DataFrame] | None:
        cache = self._episode_rows_cache
        cached = cache.get(episode_id) if cache is not None else None
        if cached is None:
            return None
        if len(cached) == 3:
            plan, steps, cached_at = cached
        else:
            # Compatibility for readers pickled from older in-flight jobs.
            plan, steps = cached
            cached_at = time.monotonic()
        if _cache_ttl_fresh(float(cached_at), self._episode_rows_cache_ttl_s):
            return plan, steps
        return None

    def _put_episode_rows_cache(self, episode_id: str, plan: Any, steps: pl.DataFrame) -> None:
        cache = self._episode_rows_cache
        if cache is None:
            return
        cache.put(
            str(episode_id),
            (plan, steps, time.monotonic()),
            max(1, int(steps.estimated_size())),
        )

    def preload_episode_rows(
        self,
        episode_ids: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Populate the per-episode row cache with grouped DeltaCAT reads.

        Shard construction already knows the episodes it will touch. Planning
        those ids in small groups preserves downstream sampling order while
        replacing many first-miss single-episode /plan calls with one call per
        group.
        """
        if self._episode_rows_cache is None:
            return

        missing_ids = [
            episode_id
            for episode_id in dict.fromkeys(str(episode_id) for episode_id in episode_ids)
            if self._cached_episode_rows(episode_id) is None
        ]
        if not missing_ids:
            return

        if batch_size is None:
            batch_size = int(
                os.environ.get("DELTACAT_PACKDS_EPISODE_ROWS_PREFETCH_BATCH_SIZE", "64") or 64
            )
        batch_size = max(1, int(batch_size))

        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start : start + batch_size]
            plan, lf = self.steps_lazyframe_for_episode_ids(batch_ids)
            steps = lf.collect().sort(["episode_id", "step_index"])
            empty = steps.head(0)
            by_episode: dict[str, pl.DataFrame] = {}
            if steps.height:
                for episode_steps in steps.partition_by("episode_id", maintain_order=True):
                    by_episode[str(episode_steps["episode_id"][0])] = episode_steps.sort(
                        "step_index"
                    )
            for episode_id in batch_ids:
                self._put_episode_rows_cache(episode_id, plan, by_episode.get(episode_id, empty))

    def episodes_lazyframe(self) -> pl.LazyFrame:
        """Return the episodes table after applying `episode_query`."""
        columns = [
            "episode_id",
            "length",
            "task_name",
            "tasks",
            "incomplete",
            *_sql_referenced_columns(self.episode_query),
        ]
        read_kwargs: dict[str, Any] = {
            "namespace": self.episodes_table_ref.namespace,
            "table": self.episodes_table_ref.table,
            "table_version": self.episodes_table_ref.table_version,
            "as_of": self.as_of_timestamp,
            "columns": list(dict.fromkeys(columns)),
            "read_as": "polars",
        }
        # App _episodes tables are companion-style metadata tables in DeltaCAT.
        # The explicit opt-in avoids the PackDS full-scan guard while keeping
        # scoped steps reads below unchanged.
        read_kwargs["allow_full_companion_scan"] = True
        result = self._catalog_read(**read_kwargs)
        lf = _apply_sql(_ensure_lazyframe(result), self.episode_query, table_name="episodes")
        schema = lf.collect_schema()
        if "incomplete" in schema:
            lf = lf.filter(~pl.col("incomplete").fill_null(False))
        return lf

    def get_episodes_metadata(self) -> list[dict[str, Any]]:
        """Return episode metadata in the shape expected by episode loaders."""
        if self._episodes_metadata is not None:
            return [dict(row) for row in self._episodes_metadata]

        if self._can_use_partition_stats_episode_metadata():
            self._episodes_metadata = self._episodes_metadata_from_partition_stats()
            self._episode_ids = [str(row["trajectory_id"]) for row in self._episodes_metadata]
            return [dict(row) for row in self._episodes_metadata]

        lf = self.episodes_lazyframe()
        schema = lf.collect_schema()
        columns = ["episode_id", "length"]
        if "task_name" in schema:
            columns.append("task_name")
        elif "tasks" in schema:
            columns.append("tasks")

        rows = lf.select(columns).collect().to_dicts()
        metadata: list[dict[str, Any]] = []
        for row in rows:
            task_name = row.get("task_name")
            tasks_value = row.get("tasks")
            if task_name:
                tasks = [str(task_name)]
            elif isinstance(tasks_value, list):
                tasks = [str(task) for task in tasks_value if task]
            elif tasks_value:
                tasks = [str(tasks_value)]
            else:
                tasks = []
            metadata.append(
                {
                    "trajectory_id": str(row["episode_id"]),
                    "length": int(row["length"]),
                    "tasks": tasks,
                    "sub_tasks": [],
                }
            )
        self._episodes_metadata = metadata
        return [dict(row) for row in self._episodes_metadata]

    def get_episode_ids(self) -> list[str]:
        """Return the episode ids selected by `episode_query`."""
        if self._episode_ids is not None:
            return list(self._episode_ids)

        if self._can_use_partition_stats_episode_metadata():
            self.get_episodes_metadata()
            return list(self._episode_ids or [])

        lf = self.episodes_lazyframe()
        schema = lf.collect_schema()
        if "episode_id" not in schema:
            raise ValueError("episode_query must preserve an 'episode_id' column.")
        ids = (
            lf.select("episode_id")
            .drop_nulls()
            .unique(maintain_order=True)
            .collect()
            .get_column("episode_id")
            .to_list()
        )
        self._episode_ids = [str(episode_id) for episode_id in ids]
        return list(self._episode_ids)

    def _can_use_partition_stats_episode_metadata(self) -> bool:
        """Use fast step-partition stats for current simple YAM episode filters."""
        query = (self.episode_query or "").strip()
        if not query:
            return True
        if query.lower().startswith("where "):
            query = query[6:].strip()
        query = query.strip().strip("()").strip()
        yam_eq = r"""(?:embodiment\s*={1,2}\s*['"]YAM['"]|['"]YAM['"]\s*={1,2}\s*embodiment)"""
        return re.fullmatch(yam_eq, query, flags=re.IGNORECASE) is not None

    def _episodes_metadata_from_partition_stats(self) -> list[dict[str, Any]]:
        """Build episode ids and lengths from the steps table partition index.

        Some managed sidecar tables have a slow companion `_episodes` table. For
        simple all-YAM training, the steps table is identity partitioned by
        episode_id, so partition stats give the same ids and lengths without a
        companion-table scan.
        """
        rows: list[dict[str, Any]] = []
        after_token: str | None = None
        while True:
            result = self._catalog_query_partition_stats(
                namespace=self.steps_table_ref.namespace,
                table=self.steps_table_ref.table,
                catalog=self.steps_table_ref.catalog,
                limit=1000,
                after_token=after_token,
            )
            result_rows = getattr(result, "rows", None)
            if result_rows is None and isinstance(result, dict):
                result_rows = result.get("rows")
            for stats_row in result_rows or []:
                if not stats_row.get("is_current", True):
                    continue
                partition_values_json = stats_row.get("partition_values_json")
                if not partition_values_json:
                    continue
                partition_values = json.loads(partition_values_json)
                if not partition_values:
                    continue
                record_count = stats_row.get("record_count")
                if record_count is None:
                    continue
                rows.append(
                    {
                        "trajectory_id": str(partition_values[0]),
                        "length": int(record_count),
                        "tasks": [],
                        "sub_tasks": [],
                    }
                )
            after_token = getattr(result, "next_token", None)
            if after_token is None and isinstance(result, dict):
                after_token = result.get("next_token")
            if not after_token:
                break

        rows.sort(key=lambda row: row["trajectory_id"])
        return rows

    def steps_lazyframe_for_episode_ids(
        self, episode_ids: Sequence[str]
    ) -> tuple[Any, pl.LazyFrame]:
        """Return a DeltaCAT read plan and step LazyFrame for selected episode ids."""
        ids = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
        if not ids:
            raise ValueError("Provide at least one episode_id.")

        columns = self._step_read_columns()
        episode_predicate = {"in": ["episode_id", ids]}
        filter_predicate = _and_filter_predicates(
            episode_predicate,
            _step_index_plan_predicate(self.steps_query),
        )
        plan = self._catalog_plan_cached(
            namespace=self.steps_table_ref.namespace,
            table=self.steps_table_ref.table,
            table_version=self.steps_table_ref.table_version,
            as_of=self.as_of_timestamp,
            columns=columns,
            filter_predicate=filter_predicate,
            include_files=True,
        )
        result = self._catalog_read(plan=plan, read_as="polars")
        lf = _ensure_lazyframe(result).sort(["episode_id", "step_index"])
        lf = self._with_forward_filled_video_blobs(lf)
        lf = _apply_sql(lf, self.steps_query, table_name="steps")
        return plan, lf

    def lowdim_lazyframe_for_episode_ids(
        self,
        episode_ids: Sequence[str],
        columns: Sequence[str],
    ) -> tuple[Any, pl.LazyFrame]:
        """Return filtered low-dimensional steps without video/blob columns.

        This is used by training-start statistics sampling. Request every
        configured low-dimensional state/action column already known to this
        reader, the identity columns needed to align rows, and any scalar
        columns named by the step SQL filter. q01/q99 preflight must not read
        source-video references, blob bytes, frame indices, JPEG/video columns,
        or decoded frames.
        """
        ids = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
        if not ids:
            raise ValueError("Provide at least one episode_id.")

        configured_lowdim_columns = [
            column
            for column in (self.step_columns or ())
            if column.startswith(f"{constants.STATE_PREFIX}:")
            or column.startswith(f"{constants.ACTION_PREFIX}:")
        ]
        canonical_columns = ["episode_id", "step_index", *configured_lowdim_columns, *columns]
        source_by_canonical = {
            column: self._source_step_column(column) for column in canonical_columns
        }
        requested_columns = list(
            dict.fromkeys(
                [
                    *source_by_canonical.values(),
                    *_sql_referenced_columns(self.steps_query),
                ]
            )
        )
        forbidden = [
            column
            for column in requested_columns
            if column.startswith(f"{constants.VIDEO_PREFIX}:")
            or column.startswith(f"{constants.VIDEO_BLOB_PREFIX}:")
            or column.startswith(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:")
            or "blob" in column
        ]
        if forbidden:
            raise ValueError(
                f"DeltaCAT low-dimensional stats reads must not request video/blob columns: {forbidden}"
            )

        filter_predicate = _and_filter_predicates(
            {"in": ["episode_id", ids]},
            _step_index_plan_predicate(self.steps_query),
        )
        plan = self._catalog_plan_cached(
            namespace=self.steps_table_ref.namespace,
            table=self.steps_table_ref.table,
            table_version=self.steps_table_ref.table_version,
            as_of=self.as_of_timestamp,
            columns=requested_columns,
            filter_predicate=filter_predicate,
            include_files=True,
        )
        result = self._catalog_read(plan=plan, read_as="polars")
        lf = _ensure_lazyframe(result).sort(["episode_id", "step_index"])
        lf = self._with_canonical_lowdim_columns(lf, source_by_canonical)
        lf = _apply_sql(lf, self.steps_query, table_name="steps")
        return plan, lf

    def get_next_steps(self, batch_size: int = 64) -> list[DeltaCatDecodedStep]:
        """Return a random batch of decoded steps."""
        if batch_size <= 0:
            return []

        batch: list[DeltaCatDecodedStep] = []
        while len(batch) < batch_size:
            _episode_id, plan, steps = self._random_nonempty_episode_dataframe(self._rng)
            take = min(batch_size - len(batch), steps.height)
            row_indices = self._rng.sample(range(steps.height), take)
            rows = [steps.row(row_index, named=True) for row_index in row_indices]
            batch.extend(self._decode_step_rows_deduped(rows, plan))
        return batch

    def _collect_episode_rows(self, episode_id: str) -> tuple[Any, pl.DataFrame, bool, float]:
        """Collect and forward-fill one episode's rows, reusing the shard cache.

        The rows contain state/action and video reference cells, not decoded
        video frames. This is the right granularity for single-turn training:
        one episode is split across several shards, so repeated full-episode
        loads should decode on demand but should not keep hitting DeltaCAT /plan
        and /read for the same row metadata.
        """
        episode_id = str(episode_id)
        cached = self._cached_episode_rows(episode_id)
        if cached is not None:
            plan, steps = cached
            return plan, steps, True, 0.0

        t0 = time.perf_counter()
        plan, lf = self.steps_lazyframe_for_episode_ids([episode_id])
        steps = lf.collect().sort("step_index")
        read_s = time.perf_counter() - t0
        self._put_episode_rows_cache(episode_id, plan, steps)
        return plan, steps, False, read_s

    def get_steps_for_episode(self, episode_id: str) -> list[DeltaCatDecodedStep]:
        """Return all decoded filtered steps for `episode_id`."""
        plan, steps, _cache_hit, _read_s = self._collect_episode_rows(episode_id)
        return self.decode_steps_dataframe(steps, plan)

    def get_steps_for_episode_window(
        self, episode_id: str, start_step: int, num_steps: int
    ) -> list[DeltaCatDecodedStep]:
        """Decode only a contiguous row window of an episode.

        This is the fast path for trajectory training. Identical result to
        ``get_steps_for_episode(episode_id)`` followed by a row slice
        ``[start_step, start_step + num_steps)`` (i.e. EpisodeData.slice), but
        only the window's rows are video-decoded instead of the whole episode.
        The step LazyFrame is collected in full first (cheap: state/action + blob
        REFERENCE columns, no video bytes) so the per-episode forward-fill of
        sparse video-blob references (``_with_forward_filled_video_blobs``, applied
        inside ``steps_lazyframe_for_episode_ids``) is preserved before slicing --
        windowing the rows BEFORE forward-fill would drop references defined
        earlier in the episode and silently lose video. ``DataFrame.slice`` clamps
        the length at the episode end (matching the trajectory loader's tail
        window), so no out-of-range handling is needed.
        """
        # Per-episode collected-rows cache (bytes-bounded LRU over MANY episodes).
        # The catalog.plan + row read for an episode is the expensive, endpoint-
        # bound op; trajectory windows are randomly sharded across episodes, so a
        # single-entry cache misses constantly. The episode ROWS are small
        # (state/action + blob REFERENCES, no video bytes), so caching every
        # episode's collected+forward-filled rows lets each episode be planned/read
        # at most ONCE per worker -- after warmup, windows are decode-only with
        # ZERO catalog calls. The byte-limited decoded LRU handles frame reuse.
        timing = _shard_timing_enabled()
        import time as _t

        plan, steps, cache_hit, read_s = self._collect_episode_rows(episode_id)

        window = steps.slice(int(start_step), int(num_steps))
        t1 = _t.perf_counter()
        decoded = self.decode_steps_dataframe(window, plan)
        if timing:
            print(  # noqa: T201
                f"[traj-window] ep_rows={steps.height} win=[{start_step},{start_step + num_steps}) "
                f"row_read={'hit' if cache_hit else f'{read_s:.1f}s'} "
                f"decode={_t.perf_counter() - t1:.1f}s decoded_steps={len(decoded)}",
                flush=True,
            )
        return decoded

    def get_step_counts_by_episode(self, episode_ids: Sequence[str]) -> dict[str, int]:
        """Return filtered row counts for selected episodes without decoding video.

        Fast path: a pure ``has_valid_action`` step filter selects exactly the
        rows the companion ``_episodes`` table already aggregated as
        ``num_valid_action_steps`` (``sum(has_valid_action)`` per episode). Read
        that pre-aggregated count -- a partition-pruned metadata read of one row
        per episode -- instead of scanning and transferring every step row of
        every selected episode just to count it. Episodes whose companion row
        predates the column (null/absent) fall back to the exact row-count read,
        so the result is identical either way.
        """
        ids = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
        if not ids:
            return {}
        # Both read paths encode ``episode_id IN (ids)`` as a GET query param, so a
        # single call over a large dataset (thousands of episodes) builds a
        # multi-MB URL the catalog server rejects with HTTP 414. Chunk the ids and
        # merge -- the per-episode counts are independent, so the union is identical
        # to one unchunked call.
        if len(ids) > _STEP_COUNT_ID_CHUNK:
            merged: dict[str, int] = {}
            for start in range(0, len(ids), _STEP_COUNT_ID_CHUNK):
                merged.update(
                    self.get_step_counts_by_episode(ids[start : start + _STEP_COUNT_ID_CHUNK])
                )
            return merged
        if _is_valid_action_steps_query(self.steps_query):
            valid_counts = self._num_valid_action_steps(ids)
            missing = [episode_id for episode_id in ids if episode_id not in valid_counts]
            counts = dict.fromkeys(ids, 0)
            counts.update(valid_counts)
            if missing:
                counts.update(self._count_steps_by_full_read(missing))
            return counts
        return self._count_steps_by_full_read(ids)

    def _num_valid_action_steps(self, episode_ids: Sequence[str]) -> dict[str, int]:
        """Per-episode ``num_valid_action_steps`` from the companion `_episodes` table.

        CONTRACT: this fast path is only correct because the writer maintains
        ``num_valid_action_steps == count(has_valid_action = true)`` for each episode
        at the same snapshot (``episodes.py`` folds ``sum(has_valid_action)`` over the
        same step rows at finalize). If a writer ever stored a different aggregate
        here, counts would diverge from the decode path's row height. The null/absent
        fallback below does NOT catch a wrong-but-present value -- it is a snapshot
        invariant, validated live for this table (count == num_valid_action_steps).

        Returns counts only for episodes whose companion row carries a non-null
        ``num_valid_action_steps``; episodes missing the column or value are
        omitted so the caller can fall back to an exact step-row count for them.
        The read is filtered by ``episode_id`` (the episodes table's identity
        partition key) so it reads one metadata row per episode rather than the
        whole companion table.
        """
        ids = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
        if not ids:
            return {}
        column = "num_valid_action_steps"
        result = self._catalog_read(
            namespace=self.episodes_table_ref.namespace,
            table=self.episodes_table_ref.table,
            table_version=self.episodes_table_ref.table_version,
            as_of=self.as_of_timestamp,
            columns=["episode_id", column],
            filter_predicate={"in": ["episode_id", ids]},
            read_as="polars",
            allow_full_companion_scan=True,
        )
        lf = _ensure_lazyframe(result)
        schema = lf.collect_schema()
        if column not in schema or "episode_id" not in schema:
            return {}
        episodes_df = (
            lf.select(["episode_id", column])
            .filter(pl.col(column).is_not_null() & pl.col("episode_id").is_in(ids))
            .collect()
        )
        return {str(row["episode_id"]): int(row[column]) for row in episodes_df.to_dicts()}

    def _count_steps_by_full_read(self, episode_ids: Sequence[str]) -> dict[str, int]:
        """Count filtered steps per episode via a full step-row read (no video)."""
        ids = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
        if not ids:
            return {}
        filter_predicate = _and_filter_predicates(
            {"in": ["episode_id", ids]},
            _step_index_plan_predicate(self.steps_query),
        )
        plan = self._catalog_plan_cached(
            namespace=self.steps_table_ref.namespace,
            table=self.steps_table_ref.table,
            table_version=self.steps_table_ref.table_version,
            as_of=self.as_of_timestamp,
            columns=list(
                dict.fromkeys(
                    ["episode_id", "step_index", *_sql_referenced_columns(self.steps_query)]
                )
            ),
            filter_predicate=filter_predicate,
            include_files=True,
        )
        result = self._catalog_read(plan=plan, read_as="polars")
        lf = _apply_sql(_ensure_lazyframe(result), self.steps_query, table_name="steps")
        counts_df = (
            lf.select(["episode_id", "step_index"])
            .group_by("episode_id")
            .agg(pl.len().alias("count"))
            .collect()
        )
        counts = dict.fromkeys(ids, 0)
        counts.update({str(row["episode_id"]): int(row["count"]) for row in counts_df.to_dicts()})
        return counts

    def _source_step_column(self, column: str) -> str:
        for prefix in ("state:", "action:"):
            if column.startswith(prefix):
                return f"{prefix}{self.key_remap.get(column.removeprefix(prefix), column.removeprefix(prefix))}"
        if column.startswith("annotation:"):
            canonical_key = f"annotation.{column.removeprefix('annotation:')}"
            mapped = self.key_remap.get(canonical_key, column.removeprefix("annotation:"))
            if mapped.startswith("annotation:"):
                return mapped
            if mapped.startswith("annotation."):
                return f"annotation:{mapped.removeprefix('annotation.')}"
            return f"annotation:{mapped}"
        return self.key_remap.get(column, column)

    def _with_canonical_lowdim_columns(
        self,
        lf: pl.LazyFrame,
        source_by_canonical: dict[str, str],
    ) -> pl.LazyFrame:
        schema = lf.collect_schema()
        aliases = [
            pl.col(source).alias(canonical)
            for canonical, source in source_by_canonical.items()
            if source != canonical and source in schema
        ]
        return lf.with_columns(aliases) if aliases else lf

    def decode_steps_dataframe(
        self,
        steps: pl.DataFrame,
        plan: Any,
    ) -> list[DeltaCatDecodedStep]:
        """Decode a collected steps DataFrame using the supplied DeltaCAT plan."""
        rows = steps.to_dicts()
        return self._decode_step_rows_deduped(rows, plan)

    def _decode_step_rows_deduped(
        self,
        rows: list[dict[str, Any]],
        plan: Any,
    ) -> list[DeltaCatDecodedStep]:
        row_requests: list[
            tuple[dict[str, Any], dict[str, VideoBlobReference], dict[str, tuple[str, int]]]
        ] = []
        references_by_key: dict[str, VideoBlobReference] = {}

        for row in rows:
            references: dict[str, VideoBlobReference] = {}
            frame_requests: dict[str, tuple[str, int]] = {}
            for camera in self._row_cameras(row):
                blob_column = f"{constants.VIDEO_BLOB_PREFIX}:{camera}"
                frame_index_column = f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{camera}"
                cell = row.get(blob_column)
                if cell is None:
                    continue
                reference = VideoBlobReference.from_cell(camera, cell)
                references[camera] = reference
                frame_index = int(row.get(frame_index_column) or 0)
                references_by_key.setdefault(reference.cache_key, reference)
                frame_requests[camera] = (reference.cache_key, frame_index)
            row_requests.append((row, references, frame_requests))

        # Bounded-chunk tables repeat one source-object reference across the
        # rows that share a small MP4. Decode each unique reference once per
        # collected batch and let the byte-limited LRU caches handle reuse
        # across later calls.
        def decode_reference(item: tuple[str, VideoBlobReference]) -> tuple[str, list[np.ndarray]]:
            key, reference = item
            return key, self._decode_reference(reference, plan, decoded_cache=self._decoded_cache)

        workers = min(self._decode_workers, max(1, len(references_by_key)))
        if workers == 1:
            decoded_by_key = dict(map(decode_reference, references_by_key.items()))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                decoded_by_key = dict(executor.map(decode_reference, references_by_key.items()))

        decoded_steps: list[DeltaCatDecodedStep] = []
        for row, references, frame_requests in row_requests:
            decoded_video: dict[str, np.ndarray] = {}
            for camera, (cache_key, frame_index) in frame_requests.items():
                frames = decoded_by_key[cache_key]
                if frame_index < 0 or frame_index >= len(frames):
                    raise IndexError(
                        f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{camera}={frame_index} "
                        f"is outside decoded blob frame count {len(frames)} for {cache_key}."
                    )
                decoded_video[camera] = frames[frame_index]
            decoded_steps.append(
                DeltaCatDecodedStep(
                    row=row,
                    decoded_video=decoded_video,
                    video_references=references,
                )
            )
        return decoded_steps

    def _random_nonempty_episode_dataframe(
        self,
        rng: random.Random,
    ) -> tuple[str, Any, pl.DataFrame]:
        episode_ids = self.get_episode_ids()
        if not episode_ids:
            raise RuntimeError("episode_query selected zero episodes.")

        for episode_id in rng.sample(episode_ids, len(episode_ids)):
            plan, steps, _cache_hit, _read_s = self._collect_episode_rows(episode_id)
            if steps.height:
                return episode_id, plan, steps
        raise RuntimeError("steps_query selected zero steps across all selected episodes.")

    def _step_read_columns(self) -> list[str] | None:
        if self.step_columns is None:
            return None

        columns = list(dict.fromkeys(self.step_columns))
        required = ["episode_id", "step_index"]
        if self.cameras is not None:
            for camera in self.cameras:
                required.append(f"{constants.VIDEO_BLOB_PREFIX}:{camera}")
                required.append(f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{camera}")
        required.extend(_sql_referenced_columns(self.steps_query))
        for column in required:
            if column not in columns:
                columns.append(column)
        return columns

    def _with_forward_filled_video_blobs(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        schema = lf.collect_schema()
        if self.cameras is None:
            blob_columns = [
                name for name in schema if name.startswith(f"{constants.VIDEO_BLOB_PREFIX}:")
            ]
        else:
            blob_columns = [
                f"{constants.VIDEO_BLOB_PREFIX}:{camera}"
                for camera in self.cameras
                if f"{constants.VIDEO_BLOB_PREFIX}:{camera}" in schema
            ]
        if not blob_columns:
            return lf
        return lf.with_columns(
            [
                pl.col(column).fill_null(strategy="forward").over("episode_id").alias(column)
                for column in blob_columns
            ]
        )

    def _row_cameras(self, row: dict[str, Any]) -> list[str]:
        if self.cameras is not None:
            return list(self.cameras)
        prefix = f"{constants.VIDEO_BLOB_PREFIX}:"
        return sorted(column.removeprefix(prefix) for column in row if column.startswith(prefix))

    def _decode_step_row(
        self,
        row: dict[str, Any],
        plan: Any,
        *,
        decoded_cache: _ObjectLRUCache | None,
    ) -> DeltaCatDecodedStep:
        decoded_video: dict[str, np.ndarray] = {}
        references: dict[str, VideoBlobReference] = {}

        for camera in self._row_cameras(row):
            blob_column = f"{constants.VIDEO_BLOB_PREFIX}:{camera}"
            frame_index_column = f"{constants.VIDEO_BLOB_FRAME_INDEX_PREFIX}:{camera}"
            cell = row.get(blob_column)
            if cell is None:
                continue

            reference = VideoBlobReference.from_cell(camera, cell)
            references[camera] = reference
            frame_index = int(row.get(frame_index_column) or 0)
            frames = self._decode_reference(reference, plan, decoded_cache=decoded_cache)
            if frame_index < 0 or frame_index >= len(frames):
                raise IndexError(
                    f"{frame_index_column}={frame_index} is outside decoded "
                    f"blob frame count {len(frames)} for {reference.cache_key}."
                )
            decoded_video[camera] = frames[frame_index]

        return DeltaCatDecodedStep(
            row=row,
            decoded_video=decoded_video,
            video_references=references,
        )

    def _decode_reference(
        self,
        reference: VideoBlobReference,
        plan: Any,
        *,
        decoded_cache: _ObjectLRUCache | None,
    ) -> list[np.ndarray]:
        cached = decoded_cache.get(reference.cache_key) if decoded_cache is not None else None
        if cached is not None:
            return cached

        if _shard_timing_enabled():
            import time as _t

            t0 = _t.perf_counter()
            blob = self._read_reference_blob(reference, plan)
            t1 = _t.perf_counter()
            frames = get_video_codec(reference.codec).decode_blob(blob)
            t2 = _t.perf_counter()
            _timing_add(fetch_s=t1 - t0, decode_s=t2 - t1, n_fetch=1, fetch_bytes=len(blob))
        else:
            blob = self._read_reference_blob(reference, plan)
            frames = get_video_codec(reference.codec).decode_blob(blob)
        if decoded_cache is not None:
            decoded_cache.put(reference.cache_key, frames, _weighted_size(frames))
        return frames

    def _read_reference_blob(self, reference: VideoBlobReference, plan: Any) -> bytes:
        cached = self._blob_cache.get(reference.cache_key)
        if cached is not None:
            return cached

        if reference.inline_blob is not None:
            blob = reference.inline_blob
        else:
            blob = self._read_s3_reference_blob(reference, plan)

        if reference.sha256 is not None:
            actual_sha256 = hashlib.sha256(blob).hexdigest()
            if actual_sha256 != reference.sha256:
                raise ValueError(
                    f"Video blob sha256 mismatch for {reference.cache_key}: "
                    f"expected {reference.sha256}, got {actual_sha256}"
                )
        if len(blob) != reference.byte_size:
            raise ValueError(
                f"Video blob size mismatch for {reference.cache_key}: "
                f"expected {reference.byte_size}, got {len(blob)}"
            )

        self._blob_cache.put(reference.cache_key, blob, len(blob))
        return blob

    def _s3_client_for_reference(self, object_uri: str, plan: Any) -> Any:
        try:
            access = self._resolve_s3_access(plan, object_uri)
        except RuntimeError:
            access = None

        if access is not None:
            s3 = self._s3_express_client_without_vended_endpoint(access, object_uri, plan)
            if s3 is not None:
                return s3

            from deltacat_io_core.s3_storage_adapter import boto3_s3_client_for_s3_access

            with _without_ambient_s3_endpoint_urls():
                s3 = boto3_s3_client_for_s3_access(access)
            if s3 is None:
                raise RuntimeError(
                    f"DeltaCAT read plan did not provide S3 access for {object_uri!r}."
                )
            return s3

        # The object is not authorized by any plan-file binding -- e.g. an
        # S3 Express video sidecar whose binding (aws_s3_express_data) is
        # configured `default_chain` and therefore gets no vended lease in
        # the plan. The lease the plan DID vend for the table's home binding
        # is project-scoped and can read the sidecar bucket (incl. the S3
        # Express CreateSession), so reuse it before dropping to ambient
        # creds. NOTE: expedient fix -- the durable fix is to make the
        # sidecar binding vend its own lease (credential_source != default_chain).
        s3 = self._borrowed_lease_s3_client(plan, object_uri)
        if s3 is not None:
            return s3
        if _ambient_s3_fallback_disabled():
            raise RuntimeError(
                "No vended lease in the read plan could authorize "
                f"{object_uri!r}, and ambient S3 fallback is "
                "disabled via DELTACAT_DISABLE_AMBIENT_S3_FALLBACK."
            )
        return self._get_ambient_s3_client(object_uri)

    def _s3_express_client_without_vended_endpoint(
        self, access: Any, object_uri: str, plan: Any
    ) -> Any | None:
        """Build an S3 Express client that ignores ambient endpoint overrides.

        H100 CI can resolve AWS' native S3 Express virtual-hosted endpoint but
        cannot use the PDX endpoint exported for checkpoint/object-store access.
        The access envelope may either omit endpoint_url or vend a PDX forwarding
        hostname. In both cases keep the authority-owned credentials and region,
        but let boto3 derive the AWS S3 Express endpoint from the directory
        bucket name.
        """
        endpoint_url = getattr(access, "endpoint_url", None)
        if (
            not _ignore_vended_s3_express_endpoint_enabled()
            or not _is_s3_express_uri(object_uri)
            or (endpoint_url is not None and not _is_pdx_s3_endpoint(endpoint_url))
        ):
            return None

        creds = getattr(access, "credentials", None)
        if not isinstance(creds, dict):
            creds = {}
        if not creds:
            return None

        from deltacat_io_core.filesystem_support import infer_s3_express_region

        region = (
            getattr(access, "region", None)
            or creds.get("region")
            or infer_s3_express_region(object_uri)
        )
        if not region:
            return None
        access_key = creds.get("access_key_id") or creds.get("aws_access_key_id")
        secret_key = creds.get("secret_access_key") or creds.get("aws_secret_access_key")
        session_token = creds.get("session_token") or creds.get("aws_session_token")
        identity = (
            getattr(access, "binding_id", None),
            getattr(access, "binding_revision", None),
            getattr(access, "root_name", None),
            endpoint_url,
            region,
            access_key,
            session_token,
            _stable_json_key(creds),
        )
        pid = os.getpid()
        if (
            self._s3_express_no_endpoint_client is not None
            and self._s3_express_no_endpoint_client_pid == pid
            and self._s3_express_no_endpoint_client_plan is plan
            and self._s3_express_no_endpoint_client_identity == identity
        ):
            return self._s3_express_no_endpoint_client

        with _BORROWED_CLIENT_LOCK, _without_ambient_s3_endpoint_urls():
            if access_key and secret_key:
                import boto3

                client = boto3.client(
                    "s3",
                    region_name=region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    config=_s3_client_config(),
                )
            else:
                from deltacat_io_core.s3_storage_adapter import (
                    create_boto3_s3_client_for_access,
                    storage_access_ref_from_s3_access,
                )

                access_ref = storage_access_ref_from_s3_access(access)
                if access_ref is None:
                    return None
                stripped_access_ref = replace(access_ref, endpoint_url=None, region=region)
                client = create_boto3_s3_client_for_access(
                    stripped_access_ref,
                    client_kwargs={"config": _s3_client_config()},
                )
                if client is None:
                    return None
        self._s3_express_no_endpoint_client = client
        self._s3_express_no_endpoint_client_pid = pid
        self._s3_express_no_endpoint_client_plan = plan
        self._s3_express_no_endpoint_client_identity = identity
        return client

    def _read_s3_reference_blob(self, reference: VideoBlobReference, plan: Any) -> bytes:
        if reference.object_uri is None:
            raise ValueError(f"Reference {reference!r} does not describe an S3 object.")

        parsed = urlparse(reference.object_uri)
        request = {
            "Bucket": parsed.netloc,
            "Key": parsed.path.lstrip("/"),
        }
        if reference.object_byte_range_start is not None:
            request["Range"] = (
                f"bytes={reference.object_byte_range_start}-{reference.object_byte_range_end}"
            )

        def read_once() -> bytes:
            s3 = self._s3_client_for_reference(reference.object_uri, plan)
            response = s3.get_object(**request)
            return response["Body"].read()

        return self._with_blob_retries(f"s3.get_object {reference.cache_key}", read_once)

    def _borrowed_lease_s3_client(self, plan: Any, object_uri: str) -> Any:
        """Build an S3 client for an object outside the plan's bound files.

        Reuses a lease the plan already vended for another (project-scoped)
        binding -- but only for objects within a root the plan declares (an
        unleased sidecar binding), never for arbitrary foreign buckets. Returns
        None if the object is not in a known root or no plan file carries usable
        credentials. boto3 is given no endpoint_url so it auto-detects the zonal
        S3 Express endpoint and performs CreateSession from the bare ``--x-s3``
        bucket name.
        """
        # Cache the client per (process, plan object). The vended lease is a
        # short-lived STS credential; the reader re-plans per episode, so keying
        # on the plan refreshes the client (and its creds) as plans rotate.
        # Compare by identity (`is`), not id(), since CPython recycles id()
        # values after GC (a fresh plan could collide with a freed one).
        pid = os.getpid()
        if (
            self._borrowed_s3_client is not None
            and self._borrowed_s3_client_pid == pid
            and self._borrowed_s3_client_plan is plan
        ):
            return self._borrowed_s3_client

        # Only borrow for objects within a data root the plan declares (the
        # unleased-sidecar case); refuse to lend a vended lease to a bucket the
        # plan does not describe.
        root = self._matching_data_root(plan, object_uri)
        if root is None:
            return None

        all_files = getattr(plan, "all_files", None)
        if not callable(all_files):
            return None
        try:
            plan_files = list(all_files())
        except Exception:  # noqa: BLE001
            return None

        lease_creds: dict[str, Any] | None = None
        for file_info in plan_files:
            file_path = getattr(file_info, "path", None) or getattr(
                file_info, "canonical_path", None
            )
            if not file_path:
                continue
            try:
                access = self._resolve_s3_access(plan, file_path)
            except RuntimeError:
                continue
            creds = getattr(access, "credentials", None)
            if (
                isinstance(creds, dict)
                and creds.get("access_key_id")
                and creds.get("secret_access_key")
            ):
                lease_creds = creds
                break
        if lease_creds is None:
            return None

        import boto3

        region = root.get("region") or lease_creds.get("region") or "us-east-2"
        # Suppress any AWS_ENDPOINT_URL[_S3]/S3_ENDPOINT_URL override (gear
        # forwards pdx.s8k.io for swift) during construction so boto3
        # auto-detects the zonal S3 Express endpoint from the bare "--x-s3"
        # bucket name. The override is restored immediately.
        with _BORROWED_CLIENT_LOCK, _without_ambient_s3_endpoint_urls():
            client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=lease_creds["access_key_id"],
                aws_secret_access_key=lease_creds["secret_access_key"],
                aws_session_token=lease_creds.get("session_token"),
                config=_s3_client_config(),
            )
        self._borrowed_s3_client = client
        self._borrowed_s3_client_pid = pid
        self._borrowed_s3_client_plan = plan
        return client

    @staticmethod
    def _matching_data_root(plan: Any, object_uri: str) -> dict[str, Any] | None:
        """Plan data_root dict whose prefix contains object_uri, else None."""
        data_roots = getattr(plan, "data_roots", None)
        if data_roots is None:
            return None
        try:
            items = list(data_roots.items())
        except (AttributeError, TypeError):
            return None
        for _name, root in items:
            try:
                rd = dict(root)
            except (TypeError, ValueError):
                rd = {"root": getattr(root, "root", None), "region": getattr(root, "region", None)}
            prefix = rd.get("root")
            if prefix and object_uri.startswith(prefix):
                return rd
        return None

    def _get_ambient_s3_client(self, object_uri: str | None = None) -> Any:
        pid = os.getpid()
        suppress_endpoint_overrides = object_uri is not None and _is_s3_express_uri(object_uri)
        if suppress_endpoint_overrides:
            from deltacat_io_core.filesystem_support import infer_s3_express_region

            region_name = infer_s3_express_region(object_uri) or "us-east-2"
        else:
            region_name = (
                os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-2"
            )

        identity = (pid, region_name, suppress_endpoint_overrides)
        if self._ambient_s3_client is None or self._ambient_s3_client_identity != identity:
            import boto3

            context = (
                _without_ambient_s3_endpoint_urls()
                if suppress_endpoint_overrides
                else nullcontext()
            )
            with context:
                self._ambient_s3_client = boto3.client(
                    "s3", region_name=region_name, config=_s3_client_config()
                )
            self._ambient_s3_client_pid = pid
            self._ambient_s3_client_identity = identity
        return self._ambient_s3_client

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        state["_client_pid"] = None
        state["_ambient_s3_client"] = None
        state["_ambient_s3_client_pid"] = None
        state["_ambient_s3_client_identity"] = None
        state["_borrowed_s3_client"] = None
        state["_borrowed_s3_client_pid"] = None
        state["_borrowed_s3_client_plan"] = None
        state["_s3_express_no_endpoint_client"] = None
        state["_s3_express_no_endpoint_client_pid"] = None
        state["_s3_express_no_endpoint_client_plan"] = None
        state["_s3_express_no_endpoint_client_identity"] = None
        state["_blob_cache"] = None
        state["_decoded_cache"] = None
        state["_episode_rows_cache"] = None
        state["_read_plan_cache"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
        self._client_pid = None
        self._ambient_s3_client = None
        self._ambient_s3_client_pid = None
        self._ambient_s3_client_identity = None
        self._borrowed_s3_client = None
        self._borrowed_s3_client_pid = None
        self._borrowed_s3_client_plan = None
        self._s3_express_no_endpoint_client = None
        self._s3_express_no_endpoint_client_pid = None
        self._s3_express_no_endpoint_client_plan = None
        self._s3_express_no_endpoint_client_identity = None
        self._blob_cache = _ObjectLRUCache(self._max_blob_cache_bytes)
        self._decoded_cache = _ObjectLRUCache(self._max_decoded_cache_bytes)
        _max_ep_rows = getattr(self, "_max_episode_rows_cache_bytes", 0)
        self._episode_rows_cache_ttl_s = getattr(
            self,
            "_episode_rows_cache_ttl_s",
            float(os.environ.get("DELTACAT_PACKDS_EPISODE_ROWS_CACHE_TTL_S", "1800") or 1800),
        )
        self._episode_rows_cache = _ObjectLRUCache(_max_ep_rows) if _max_ep_rows > 0 else None
        self._max_read_plan_cache_entries = getattr(
            self,
            "_max_read_plan_cache_entries",
            int(os.environ.get("DELTACAT_PACKDS_MAX_READ_PLAN_CACHE_ENTRIES", "4096") or 0),
        )
        self._read_plan_cache_ttl_s = getattr(
            self,
            "_read_plan_cache_ttl_s",
            float(os.environ.get("DELTACAT_PACKDS_READ_PLAN_CACHE_TTL_S", "1800") or 1800),
        )
        self._read_plan_cache = (
            _ObjectLRUCache(self._max_read_plan_cache_entries)
            if self._max_read_plan_cache_entries > 0
            else None
        )

    @staticmethod
    def _resolve_s3_access(plan: Any, object_uri: str) -> Any:
        from deltacat_io_core.s3_storage_adapter import resolve_plan_file_s3_access

        errors: list[str] = []
        for file_info in plan.all_files():
            try:
                return resolve_plan_file_s3_access(
                    file_info,
                    object_uri,
                    storage_bindings=plan.storage_bindings,
                    data_roots=plan.data_roots,
                    credentials=plan.credentials,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Could not resolve DeltaCAT S3 authority for {object_uri!r}.\n" + "\n".join(errors)
        )
