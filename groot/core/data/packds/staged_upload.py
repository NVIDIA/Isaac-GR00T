# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Local staging dir + bulk s5cmd upload for remote PackDS conversion output.

Why this module exists
======================

Conversion writes a Lance dataset by emitting many small fragment
files (one per ``write_batch`` flush, typically ~10–60 MiB each), each
followed by a manifest commit that updates the dataset's version
history.  Targeted at S3 / SwiftStack the per-batch pattern is:

- many small ``PUT`` requests (one per fragment data file),
- many ``HEAD`` + conditional ``PUT If-None-Match`` requests on the
  manifest (one per commit, with internal retries),
- a steady drumbeat of ``GET`` requests reading existing fragments
  during ``optimize``,
- finally the stats and episodes-table writes.

That request shape saturates SwiftStack: the cluster returns
``HTTP 429`` under sustained sub-100 KiB request bursts, and even
without 429s the per-request latency (~50–200 ms) dominates wall
time.  Coalescing the same byte volume into one big bulk upload at
the end is the access pattern SwiftStack actually serves cheaply.

This module provides the bulk-upload primitive.  The conversion
orchestrator (``convert_to_packds``) uses it automatically whenever
the output URI is a remote Lance path: the writer runs against a
local staging directory under ``$TMPDIR/packds_staging/...`` and the
finished dataset is uploaded in one ``S5cmdRunner.cp`` call after
all finalize work completes.  ``PackDSLanceDB._finalize`` refuses to
run against a remote path for the same reason — the per-batch
commit storm is consistently slower than bulk upload after a local
finalize.  DeltaCAT writes always bypass this path — DeltaCAT's own
batching is the right answer for that backend.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from groot.core.data.packds import utils
from groot.core.data.packds.s5cmd_runner import S5cmdRunner
from groot.core.data.packds.sources import resolve_source

logger = logging.getLogger(__name__)


def make_staging_path(remote_uri: str, *, base_dir: str | None = None) -> Path:
    """Pick a unique local staging directory for *remote_uri*.

    The leaf component of *remote_uri* is preserved on disk so a
    stale staging dir is recognisable by name; everything above it is
    namespaced by pid + uuid so two concurrent conversions don't
    collide.

    Args:
        remote_uri: The eventual upload destination
            (e.g. ``s3://bucket/prefix/foo.packds``).
        base_dir: Override the staging root.  Default uses
            ``$TMPDIR/packds_staging``.

    Returns:
        A ``Path`` that does not yet exist on disk.  The caller is
        responsible for creating it.
    """
    name = remote_uri.rstrip("/").rsplit("/", 1)[-1] or "packds_staging"
    root = Path(base_dir or tempfile.gettempdir()) / "packds_staging"
    return root / f"{os.getpid()}_{uuid.uuid4().hex[:8]}" / name


def upload_packds_directory(local_path: Path, remote_uri: str) -> None:
    """Bulk-upload an entire local PackDS directory to *remote_uri*.

    Walks every regular file under *local_path*, builds
    ``(src, dst)`` pairs preserving the relative path layout, and
    issues one :meth:`S5cmdRunner.cp` call.  The runner's watchdog
    + per-op JSON validation enforces "either every file lands or
    we raise" — partial uploads are not a returnable state.

    The s5cmd endpoint and credentials are derived from
    ``resolve_source(remote_uri)`` so the upload uses whatever
    profile the rest of the conversion uses to talk to the
    bucket.

    Raises if *local_path* doesn't exist, contains no files, or the
    underlying s5cmd run fails after retries.
    """
    local_path = Path(local_path)
    if not local_path.is_dir():
        raise FileNotFoundError(f"staging dir {local_path} not found")

    files: list[tuple[str | Path, str]] = []
    for fp in sorted(local_path.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(local_path).as_posix()
        files.append((fp, remote_uri.rstrip("/") + "/" + rel))

    if not files:
        raise RuntimeError(
            f"upload_packds_directory: no files under {local_path}; "
            "refusing to claim the upload succeeded.",
        )

    total_bytes = sum(Path(src).stat().st_size for src, _ in files)
    logger.info(
        "staged-upload: uploading %d file(s), %.2f GiB from %s -> %s",
        len(files), total_bytes / 1024**3, local_path, remote_uri,
    )

    source = resolve_source(remote_uri)
    storage_options = source.storage_options or {}
    env = utils.storage_options_to_s3_env(storage_options)
    endpoint = storage_options.get("aws_endpoint", "")

    # Watchdog params tuned for the staged-upload shape (typically
    # 100s–10000s of small Lance fragment files, total volume up to
    # ~50 GiB):
    #
    # - ``min_throughput_b_per_s = 100 KiB/s`` is much more permissive
    #   than the 800 KiB/s default.  The default assumes healthy
    #   single-stream throughput; staged-upload submits the whole
    #   conversion's byte volume at once and SwiftStack effective
    #   throughput drops sharply under load (we observed ~750 KiB/s
    #   on a single 864 MiB / 1213-file batch that the default
    #   killed prematurely).  100 KiB/s gives ~10× headroom before
    #   the watchdog declares the batch "stuck".
    # - ``warn_floor_s = 60s`` so smallest batches aren't warned at
    #   30s for a TLS handshake.
    # - ``kill_grace_s = 300s`` (5 min) to cushion any single slow
    #   request that would otherwise tip the whole call into a
    #   watchdog kill on a ~boundary-pace SwiftStack.
    #
    # Numbers are deliberately loose: a batch that's truly hung
    # (e.g. credentials revoked, endpoint unreachable) still fails
    # within the size-scaled budget.  A batch that's merely slow
    # because SwiftStack is overloaded gets the whole budget to
    # finish, since killing + retrying just adds load.
    runner = S5cmdRunner(
        endpoint=endpoint,
        env=env,
        min_throughput_b_per_s=100 * 1024,
        warn_floor_s=60.0,
        kill_grace_s=300.0,
    )
    upload_start = time.perf_counter()
    runner.cp(
        files,
        total_bytes=total_bytes,
        expected_count=len(files),
        log_label="staged-upload",
    )
    elapsed = time.perf_counter() - upload_start
    logger.info(
        "staged-upload: complete in %.1fs (%.1f MiB/s effective)",
        elapsed, (total_bytes / 1024**2) / max(elapsed, 0.001),
    )


def cleanup_staging(local_path: Path) -> None:
    """Remove a staging directory.  Idempotent; no-op when absent."""
    local_path = Path(local_path)
    if local_path.exists():
        shutil.rmtree(local_path, ignore_errors=True)
