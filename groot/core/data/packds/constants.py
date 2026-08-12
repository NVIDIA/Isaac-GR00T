# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Internal constants for PackDS format.

These constants define the internal structure of the PackDS format and should NOT
be exported or used outside this package. External code should use the PackDS class
API which encapsulates these details.
"""

import pyarrow as pa

# Format version - increment when making breaking changes to the format
PACKDS_FORMAT_VERSION = 6

# Table naming
TABLE_NAME = "steps"
EPISODES_TABLE_NAME = "episodes"
TABLE_FORMAT_LANCE = "lance"
TABLE_FORMAT_PARQUET = "parquet"
TABLE_FORMATS = (
    TABLE_FORMAT_LANCE,
    TABLE_FORMAT_PARQUET,
)
DEFAULT_TABLE_PREFERENCE = TABLE_FORMATS

# DeltaCAT companion-table suffixes.  The PackDS DeltaCAT writer
# requires the steps-table name to end with ``STEPS_TABLE_SUFFIX`` so
# the application-managed episodes-table name can be derived
# deterministically by swapping the suffix.
#
# - ``APP_EPISODES_TABLE_SUFFIX`` — application-managed sibling that
#   stores per-episode aggregate metadata (length, duration, task_name,
#   etc.).  Written directly via ``mode="merge"``; schema is allowed to
#   evolve.
# - ``AUTO_COMPANION_TABLE_SUFFIX`` — appended by DeltaCAT itself to
#   maintain a discovery companion alongside the steps table.  Frozen
#   schema, never written by us; the application ``_episodes`` table is
#   the source of truth in our pipeline.
STEPS_TABLE_SUFFIX = "_steps"
APP_EPISODES_TABLE_SUFFIX = "_episodes"
AUTO_COMPANION_TABLE_SUFFIX = "__episodes"

# Canonical Lance-backed PackDS table directory.
LANCE_TABLE_DIR = "steps.lance"

# Dataset extension
PACKDS_EXTENSION = ".packds"

# Column prefixes for different modalities
VIDEO_PREFIX = "video"
VIDEO_BLOB_PREFIX = "video_blob"
VIDEO_BLOB_FRAME_INDEX_PREFIX = "video_blob_frame_index"
STATE_PREFIX = "state"
ACTION_PREFIX = "action"

# Core required columns in the PackDS table
# episode_id (str) is the primary episode identifier
REQUIRED_COLUMNS = frozenset(
    {
        "episode_id",
        "step_index",
        "step_time_ns",
        "original_step_time_ns",
        "has_valid_action",
    }
)

# Default min-batch size for the PackDS Lance writer's flush threshold.
#
# WHY THIS MATTERS — read before changing:
# Every ``PackDSLanceDB`` flush calls ``lance.write_dataset(mode="append")``
# on the steps table, which writes a *new* manifest version that contains
# every fragment in the dataset.  The cost of one append is therefore
# O(N_existing_fragments), so the total cost of writing M batches grows
# as O(M^2) — the manifest history under ``steps.lance/_versions/`` also
# grows by one file per commit.
#
# This default used to be 9000 (set explicitly by the conversion path)
# but was silently lowered to 30 when commit ea2e6cc75a (April 2026)
# refactored ``WRITE_BATCH = 9000`` into ``ConversionConfig`` and let
# the field inherit ``DEFAULT_WRITE_BATCH_SIZE = 30``.  At 30 rows per
# flush a ~1M-step conversion produces ~33,000 manifest commits and
# tens of thousands of small per-fragment writes, which:
#   - on the local-staging path (typical) accumulates a 100+ GiB
#     ``_versions/`` directory of redundant manifests and quadratic
#     wall time (the orchestrator was observed stuck for hours on
#     episodes that should have taken seconds);
#   - on the direct-to-S3 path produces the sub-100 KiB request storm
#     that triggers SwiftStack HTTP 429s and pages the SRE team —
#     i.e. exactly the disruption that motivated this docstring.
#
# 9000 keeps each flush large enough that fragment count stays
# bounded for typical conversions (~100s of fragments, not tens of
# thousands) while still giving Ray workers a reasonable per-write
# memory ceiling.  Production conversion configs should leave this
# alone unless they have a specific reason to flush smaller — and if
# they do, they should be sized so that ``total_rows / min_batch_size``
# stays in the low thousands at most.
DEFAULT_WRITE_BATCH_SIZE = 9000

# S3 streaming-uploader tunables.  These set the defaults baked into
# ``PackDSWriteConfig`` and used by ``_StreamingUploader``; per-script
# overrides should pass them via the ``PackDSLanceDB`` constructor
# (``s3_upload_threshold_bytes`` / ``s3_upload_workers``).
#
# - ``S3_UPLOAD_THRESHOLD_BYTES``: minimum bytes accumulated before a
#   batch is submitted to the upload pool.  Larger values mean fewer,
#   bigger uploads (better amortisation of subprocess + commit
#   overhead) but a longer tail under ``flush()``.
# - ``S3_UPLOAD_WORKERS``: max concurrent ``s5cmd`` subprocesses.
#   Bumping helps only if the writer keeps the queue deep enough to
#   feed them; otherwise idle workers waste process slots.
S3_UPLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
S3_UPLOAD_WORKERS = 4

# Stuck-upload watchdog thresholds.  An ``s5cmd`` upload is
# declared stuck if it hasn't finished by
#
#     warn_after = max(WARN_FLOOR_S, total_bytes / MIN_THROUGHPUT_B_PER_S)
#     kill_after = warn_after + KILL_GRACE_S
#
# This scales the budget with the size of the batch (a 256 MiB
# bundle gets 256·1024/400 ≈ 655s before warning), so small
# fragments aren't held to the same wall-clock as large ones — a
# zero-progress hang on a small upload still escapes promptly.
#
# - ``MIN_THROUGHPUT_B_PER_S = 800 KiB/s`` is below any healthy S3
#   path; healthy uploads sustain >>1 MiB/s.  This only fires on
#   actual hangs.
# - ``WARN_FLOOR_S`` keeps the warning budget reasonable for very
#   small batches — TLS handshake + first byte can legitimately
#   take a few seconds.
# - ``KILL_GRACE_S`` is the extra time after the WARNING before we
#   ``killpg`` the subprocess.  Lets a marginally-slow-but-moving
#   upload finish without bouncing through a retry.
# - ``FLUSH_HEARTBEAT_S`` is the heartbeat cadence in
#   ``_flush_remote_uploads`` while ``wait()`` is blocked, so a
#   long flush doesn't look like a hang.
S3_UPLOAD_MIN_THROUGHPUT_B_PER_S = 800 * 1024
S3_UPLOAD_WARN_FLOOR_S = 30.0
S3_UPLOAD_KILL_GRACE_S = 30.0
S3_UPLOAD_FLUSH_HEARTBEAT_S = 30.0

# Progress-monitor cadence for the ``S5cmdRunner`` watchdog thread.
# - ``PROGRESS_FIRST_DU_S``: how long to let a transfer run before
#   issuing the first ``s5cmd du`` to (a) resolve unknown total
#   size for downloads and (b) recompute size-scaled watchdog
#   thresholds.  Short enough that genuinely tiny transfers
#   complete before we ever spend a round-trip on ``du``.
# - ``PROGRESS_POLL_INTERVAL_S``: cadence of subsequent progress
#   measurements + log lines.  Each tick measures the destination
#   side (local ``scandir`` for downloads, ``s5cmd du`` for
#   uploads) and emits a one-line progress record.
S3_PROGRESS_FIRST_DU_S = 5.0
S3_PROGRESS_POLL_INTERVAL_S = 10.0

# Batch size for DeltaCAT writes.  Each write_batch() call creates one
# DeltaCAT delta, so we accumulate more rows than the Lance fragment
# threshold to avoid an explosion of tiny deltas.
DELTACAT_WRITE_BATCH_SIZE = 9000

# Append/manifest commit retry policy for concurrent writers.
# Tuned for shared-table workloads (e.g. dozens of Ray workers all
# committing to the same ``main_steps`` simultaneously) talking to a
# SwiftStack-backed Lance store whose ``If-None-Match`` semantics are
# loosely compliant — Lance occasionally perceives phantom version
# conflicts on manifest writes and retries internally.
#
# There are two retry layers, each with its own count:
#
# - **Inner** (``COMMIT_INNER_MAX_RETRIES``): passed as
#   ``lance.LanceDataset.commit(max_retries=...)``.  Lance's Rust
#   client retries the manifest CAS internally with exponential
#   backoff up to ``COMMIT_RETRY_MAX_SLEEP_S`` per retry.  Inner
#   retries reuse the same ``lance.dataset`` handle and stale
#   ``read_version``, so they only help with failures the Rust
#   client can resolve in-place (e.g. brief HTTP 503 bursts).
# - **Outer** (``COMMIT_MAX_RETRIES``): the Python wrapper loop in
#   ``_commit_remote`` / ``flush`` re-issues the whole
#   ``lance.commit`` on retryable errors with linear backoff
#   ``COMMIT_RETRY_SLEEP_S * attempt``.  Each outer attempt builds
#   a fresh ``lance.dataset(...)`` and re-reads ``ds.version``, so
#   it self-heals stale-manifest pathologies that the inner loop
#   can't escape from.
#
# We weight outer >> inner because empirically some commits only
# resolve after a fresh dataset handle.  Total retry surface is
# 30 × 5 = 150 attempts; if a commit can't land in 150 attempts
# the worker raises and the episode-level
# ``@retry_on_exception(max_attempts=3)`` re-runs the whole
# episode on a fresh Ray worker, which often clears the
# pathology entirely.  Typical CAS still settles in <10 outer
# attempts.
COMMIT_WARN_RETRIES = 10
COMMIT_MAX_RETRIES = 30
COMMIT_INNER_MAX_RETRIES = 5
COMMIT_RETRY_SLEEP_S = 0.1
COMMIT_RETRY_MAX_SLEEP_S = 30.0

# Default JPEG quality for video encoding
DEFAULT_JPEG_QUALITY = 95

# Default CRF and group size for multi-frame video blob encoding (H.264 or AV1).
# At CRF 18, group_size 16: ~90% smaller than JPEG q95, ~9 dB lower PSNR (~44 vs ~53),
# ~8x slower single-frame decode. Useful for storage-constrained pre-training; for
# online learning or post-training where decode latency and quality matter, use JPEG.
DEFAULT_VIDEO_BLOB_CRF = 18
DEFAULT_VIDEO_BLOB_GROUP_SIZE = 16
DEFAULT_VIDEO_BLOB_CODEC = "av1"

# Encoder presets for video blob encoding and export
H264_ENCODER = "libx264"
H264_PRESET = "veryfast"
AV1_ENCODER = "libsvtav1"
AV1_PRESET = "12"  # SVT-AV1 preset (12 = fastest, 0 = slowest)
AV1_CRF = 28  # Default CRF for AV1 encoding (distinct from H264 CRF above)

# Default steps per second (control frequency in Hz)
DEFAULT_STEPS_PER_SECOND = 30.0

# Columns that receive BTREE scalar indices on finalize
INDEX_COLUMNS = (
    "episode_id",
    "step_index",
    "step_time_ns",
    "original_step_time_ns",
)

# Default chunk size for ``PackDSLanceDB.finalize_episodes``.  Each
# chunk becomes one ``merge_insert`` against ``episodes.lance``;
# 10000 amortises the per-commit overhead while keeping the chunked
# Arrow table small (few hundred KiB at typical schema width).
EPISODES_FINALIZE_CHUNK_SIZE = 10_000

# Validation defaults
VALIDATE_DEFAULT_SAMPLE_SIZE = 100
VALIDATE_BATCH_SIZE = 1000

# ThreadPoolWriteSession idle timeout (seconds).  If no write future
# completes within this window, the session assumes the write is stuck.
WRITE_BUFFER_FLUSH_TIMEOUT_S = 300

# DeltaCAT plan cache TTL (seconds).  Plans are content-addressed so
# safe to cache; credentials may expire sooner (typically 1h STS tokens).
PLAN_CACHE_TTL_S = 300.0

# Episodes table schema. See EPISODES_TABLE_DESIGN.md for rationale.
# episode_index is not stored — row position in the episodes table IS the index.
#
# The DeltaCAT-backed _episodes table is a plain (non-PackDS) Lance table
# identity-partitioned on episode_id with no merge keys. Per-episode
# upserts go through ``mode="replace_partition"``: each call rewrites
# exactly one episode_id partition's single row. There is no notion of
# step_index here — that's the steps table's concern.
#
# All ingest-time configuration that was historically stamped on the
# steps table's ``packds:*`` schema metadata lives here per-episode.
# That means a single shared ``main_steps`` table can hold episodes
# ingested with different source formats / encodings / processing
# pipelines without losing per-episode provenance.
EPISODES_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string(), nullable=False),
    # --- aggregate from steps -------------------------------------
    pa.field("length", pa.int64()),
    pa.field("duration_ns", pa.int64()),
    pa.field("num_valid_action_steps", pa.int64()),
    # First task observed in this episode (temporally earliest non-empty
    # ``annotation:task`` value). Single string — for episodes whose task
    # changes mid-trajectory, downstream consumers should fall back to
    # the per-step ``annotation:task`` column on the steps table.
    pa.field("task_name", pa.string()),
    pa.field("has_video_blobs", pa.bool_()),
    pa.field("has_jpegs", pa.bool_()),
    pa.field("start_time", pa.timestamp("us", tz="UTC")),
    pa.field("modified_time", pa.timestamp("us", tz="UTC")),
    # Mid-rewrite state flag.  The DeltaCAT replace-episode flow upserts
    # this row with ``incomplete=True`` before deleting the steps-table
    # partition, then clears it on the post-write ``finalize_episode``.
    # Readers must treat ``incomplete=True`` (or any past in-flight
    # crash that left the flag stuck) as "episode absent" so concurrent
    # reads don't surface a partially-rewritten partition.  Rows that
    # predate this column have ``incomplete=NULL`` and are treated as
    # ``False`` by the reader.
    pa.field("incomplete", pa.bool_()),
    # --- user-supplied metadata (preserved across re-finalize) ----
    pa.field("vendor", pa.string()),
    pa.field("operator", pa.string()),
    # ``robot`` is the *specific* unit identifier (e.g. ``"YAM-US-01"``),
    # not a general embodiment class. Use ``embodiment`` for the latter.
    pa.field("robot", pa.string()),
    # General embodiment class (e.g. ``"YAM"``, ``"G1"``). Sourced from
    # ``ConversionConfig.embodiment`` first, then input-data metadata,
    # otherwise left empty.
    pa.field("embodiment", pa.string()),
    pa.field("version", pa.string()),
    # --- ingest-time provenance (per-episode, fixed-at-write) -----
    pa.field("packds_version", pa.int64()),
    pa.field("source_format", pa.string()),
    pa.field("source_paths", pa.list_(pa.string())),
    pa.field("steps_per_second", pa.float64()),
    pa.field("jpeg_quality", pa.int64()),
    pa.field("video_blob_crf", pa.int64()),
    pa.field("video_blob_group_size", pa.int64()),
    pa.field("video_blob_codec", pa.string()),
    pa.field("processing", pa.string()),  # JSON-encoded dict
])

# Subset of EPISODES_SCHEMA columns that record ingest-time provenance.
# Treated as fixed-at-write per episode — a re-finalize keeps them in
# sync with the writer that produced the episode's steps. The names
# mirror the legacy ``packds:*`` schema-metadata keys (with the
# ``packds:`` prefix dropped) so existing operator playbooks don't have
# to learn a second naming convention.
EPISODES_PROVENANCE_FIELDS = (
    "packds_version",
    "source_format",
    "source_paths",
    "steps_per_second",
    "jpeg_quality",
    "video_blob_crf",
    "video_blob_group_size",
    "video_blob_codec",
    "processing",
)

# Identity partition scheme by episode_id, used when bootstrapping the
# non-PackDS _episodes table. PackDS-layout tables auto-apply the same
# scheme server-side; for non-PackDS tables we declare it explicitly.
EPISODES_TABLE_PARTITION_SCHEME = {
    "keys": [{"key": ["episode_id"], "transform": "identity"}],
}
