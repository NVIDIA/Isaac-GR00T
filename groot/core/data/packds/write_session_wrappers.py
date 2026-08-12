# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Composable WriteSession wrappers for batching and background execution.

``BatchingWriteSession`` accumulates small Arrow tables into larger batches.
``ThreadPoolWriteSession`` runs ``write_batch()`` calls in a thread pool.

Both implement the :class:`WriteSession` protocol and delegate to an inner
session, so they compose::

    session = DeltaCatWriteSession(...)
    session = BatchingWriteSession(session, min_batch_size=9000)

    session = create_lance_write_session(...)
    session = BatchingWriteSession(session, min_batch_size=30)
    session = ThreadPoolWriteSession(session, max_workers=1)

See ``WRITE_SESSION_REFACTOR.md`` for the full design rationale and threading
architecture documentation.
"""

from __future__ import annotations

import concurrent.futures

import pyarrow as pa

from groot.core.data.packds import constants
from groot.core.data.packds.write_session import WriteSession


class BatchingWriteSession:
    """WriteSession wrapper that batches small writes.

    Accumulates Arrow tables in a list until *min_batch_size* rows are
    reached, then concatenates and forwards to the inner session.

    Synchronous — no threads, no queues, no timeouts.  Errors propagate
    immediately from the inner session's ``write_batch()``.
    """

    def __init__(self, inner: WriteSession, min_batch_size: int) -> None:
        self._inner = inner
        self._min_batch_size = min_batch_size
        self._pending: list[pa.Table] = []
        self._pending_rows: int = 0

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        merged = pa.concat_tables(self._pending, promote_options="default")
        self._inner.write_batch(merged)
        self._pending.clear()
        self._pending_rows = 0

    def write_batch(self, data: pa.Table) -> None:
        if data.num_rows == 0:
            return
        self._pending.append(data)
        self._pending_rows += data.num_rows
        if self._pending_rows >= self._min_batch_size:
            self._flush_pending()

    def flush(self) -> None:
        self._flush_pending()
        self._inner.flush()

    def finalize(self, metadata: dict[str, str]) -> None:
        self._flush_pending()
        self._inner.finalize(metadata)

    def close(self) -> None:
        self._flush_pending()
        self._inner.close()

    def forget_partition(self, episode_ids: str | list[str]) -> None:
        """Forward to the inner session if it tracks per-partition state.

        ``DeltaCatWriteSession`` uses this to reset its
        "first-write-this-session" bookkeeping for an episode after
        ``finalize_episode``, so that re-writing the same id within
        one session goes back through ``mode="replace_partition"``.
        Inner sessions that don't track this (LanceDB, etc.) just
        accept the call as a no-op.
        """
        forget = getattr(self._inner, "forget_partition", None)
        if forget is not None:
            forget(episode_ids)


class ThreadPoolWriteSession:
    """WriteSession wrapper that runs writes in a background thread pool.

    Submits each ``write_batch()`` as a Future.  ``_wait_all()`` waits for
    all pending futures with a progress-aware timeout: the deadline resets
    every time a future completes, so slow-but-progressing writes never
    time out.

    **Threading layer for write-call parallelism.**  For remote Lance paths,
    this is separate from the S3 upload threading inside
    ``StagedRemoteLanceWriteSession._StreamingUploader``:

    - This pool overlaps *fragment creation* (CPU-bound encoding).
    - The inner ``_StreamingUploader`` pool overlaps *S3 upload* (I/O-bound).

    They serve different purposes and coexist intentionally.  See
    "Threading Architecture" in ``WRITE_SESSION_REFACTOR.md``.

    On write failure: raises immediately from ``_raise_completed_errors()``,
    but does NOT cancel in-flight futures.  ``close()`` cleans up.
    Consistent with docs/error_handling_philosophy.md — crash loud.
    """

    def __init__(
        self,
        inner: WriteSession,
        max_workers: int = 1,
        idle_timeout_s: float = constants.WRITE_BUFFER_FLUSH_TIMEOUT_S,
    ) -> None:
        self._inner = inner
        self._idle_timeout_s = idle_timeout_s
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[concurrent.futures.Future] = []

    def _raise_completed_errors(self) -> None:
        """Check completed futures and raise the first error found."""
        still_pending = []
        for f in self._futures:
            if f.done():
                f.result()  # raises if the write failed
            else:
                still_pending.append(f)
        self._futures = still_pending

    def _wait_all(self) -> None:
        """Wait for all pending futures with a progress-aware timeout."""
        remaining = list(self._futures)
        self._futures.clear()
        while remaining:
            done, remaining_set = concurrent.futures.wait(
                remaining,
                timeout=self._idle_timeout_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            remaining = list(remaining_set)
            if not done:
                raise RuntimeError(
                    f"Write thread made no progress for "
                    f"{self._idle_timeout_s}s — may be stuck. "
                    f"{len(remaining)} writes still pending."
                )
            for f in done:
                f.result()  # raises if the write failed

    def write_batch(self, data: pa.Table) -> None:
        self._raise_completed_errors()
        future = self._pool.submit(self._inner.write_batch, data)
        self._futures.append(future)

    def flush(self) -> None:
        self._wait_all()
        self._inner.flush()

    def finalize(self, metadata: dict[str, str]) -> None:
        self._wait_all()
        self._inner.finalize(metadata)

    def close(self) -> None:
        self._wait_all()
        self._pool.shutdown(wait=False)
        self._inner.close()

    def forget_partition(self, episode_ids: str | list[str]) -> None:
        forget = getattr(self._inner, "forget_partition", None)
        if forget is not None:
            forget(episode_ids)
