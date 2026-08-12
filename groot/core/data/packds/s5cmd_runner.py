# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Strongly-typed wrapper around ``s5cmd`` for both PackDS S3 uploads and downloads.

Why a dedicated abstraction instead of a one-line ``subprocess.run``:

PackDS shells out to ``s5cmd`` from two places:

- The streaming uploader writes Lance fragment files to S3 (one
  ``s5cmd cp local s3://...`` per fragment batch).
- The conversion-format loaders download episode data from S3 to a
  local scratch dir (``s5cmd cp s3://.../episode/* /tmp/...``)
  before reading frames.

A naive wrapper has three failure modes that silently corrupt the
job (we've hit all three in production):

1. **Hang.**  The TCP connection to S3 stays ESTABLISHED, but the
   server never finishes its response and ``s5cmd`` blocks on a
   socket read forever.  Without a watchdog the convert thread sits
   in ``flush()`` (or ``materialize_episode``) indefinitely while
   the rest of the Ray cluster idles.
2. **Orphaned subprocess.**  ``subprocess.Popen`` doesn't put the
   child in a new process group, so SIGTERM to the Python process
   doesn't propagate.  An interrupted convert leaves ``s5cmd``
   still transferring to a now-orphaned destination.
3. **Silent partial success.**  When the user finally ``pkill``-s
   the hung ``s5cmd`` it can exit *zero* (graceful Go shutdown)
   even though some operations were cancelled mid-stream.
   ``check=True`` only guards against non-zero exit, so the caller
   sees "success" and commits manifest entries / treats files as
   present when they actually aren't — the worst possible outcome.

This module addresses all three:

- ``--json`` mode parses an explicit success/error record per
  operation; we reject the whole call if any operation reports an
  error, and (when the caller knows the expected op count) we
  cross-check the success count too.  Glob downloads where the op
  count is unknown still get returncode + error-record validation.
- ``start_new_session=True`` puts ``s5cmd`` in its own process
  group so we can ``killpg`` cleanly.
- A polling watchdog with size-scaled thresholds escalates hung
  subprocesses from WARNING to a hard kill + retry.

Per ``docs/error_handling_philosophy.md``: this code does not
silently recover.  Transient errors retry; once attempts are
exhausted the exception propagates so the caller halts before
acting on a partial result.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from groot.core.data.packds import constants

logger = logging.getLogger(__name__)


class S5cmdError(RuntimeError):
    """Terminal s5cmd failure — retries exhausted, caller should halt."""


class S5cmdRetryableError(S5cmdError):
    """Transient s5cmd failure — caller may retry the same input."""


@dataclass(frozen=True)
class _OpResult:
    """One parsed JSON record from ``s5cmd --json``."""
    succeeded: bool
    detail: str  # destination on success; error message on failure


class S5cmdRunner:
    """Run ``s5cmd cp`` for a batch of ``(src, dst)`` pairs.

    Either side of each pair may be local (``Path`` or ``str``) or
    an ``s3://`` URI — the same wrapper handles uploads
    (``local → s3://``), downloads (``s3:// → local``), and bucket-
    to-bucket copies.

    ``cp(ops)`` either returns successfully — meaning the operation
    completed without any error records from ``s5cmd --json`` — or
    raises :class:`S5cmdError`.  There is no third state.

    Attributes are read-only after construction; the same instance
    is safe to reuse across many calls from a single thread.
    Concurrent ``cp`` calls from multiple threads each spawn an
    independent ``s5cmd`` subprocess — they don't share state.
    """

    def __init__(
        self,
        endpoint: str,
        env: dict[str, str],
        *,
        min_throughput_b_per_s: float = constants.S3_UPLOAD_MIN_THROUGHPUT_B_PER_S,
        warn_floor_s: float = constants.S3_UPLOAD_WARN_FLOOR_S,
        kill_grace_s: float = constants.S3_UPLOAD_KILL_GRACE_S,
        progress_first_du_s: float = constants.S3_PROGRESS_FIRST_DU_S,
        progress_poll_interval_s: float = constants.S3_PROGRESS_POLL_INTERVAL_S,
        max_attempts: int = 2,
        retry_backoff_s: float = 5.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if kill_grace_s <= 0:
            raise ValueError(f"kill_grace_s must be > 0, got {kill_grace_s}")
        if min_throughput_b_per_s <= 0:
            raise ValueError(
                f"min_throughput_b_per_s must be > 0, got {min_throughput_b_per_s}"
            )
        self._endpoint = endpoint
        self._env = env
        self._min_throughput_b_per_s = min_throughput_b_per_s
        self._warn_floor_s = warn_floor_s
        self._kill_grace_s = kill_grace_s
        self._progress_first_du_s = progress_first_du_s
        self._progress_poll_interval_s = progress_poll_interval_s
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s

    def _thresholds_for(self, total_bytes: int) -> tuple[float, float]:
        """Return ``(warn_after_s, kill_after_s)`` for *total_bytes*.

        WARN fires once the elapsed time exceeds the budget implied
        by ``min_throughput_b_per_s`` (or ``warn_floor_s``,
        whichever is larger — small uploads always get the floor so
        a TLS handshake doesn't trigger a spurious warning).  KILL
        fires ``kill_grace_s`` later, giving a marginally-slow-but-
        making-progress upload a final window to land before we
        treat it as hung.
        """
        scaled = total_bytes / self._min_throughput_b_per_s
        warn = max(self._warn_floor_s, scaled)
        kill = warn + self._kill_grace_s
        return warn, kill

    def _progress_tick(
        self,
        *,
        ops: list[tuple[str | Path, str]],
        direction: str,
        log_label: str,
        attempt: int,
        elapsed: float,
        first_tick: bool,
        current_total: int | None,
        current_warn: float,
        current_kill: float,
    ) -> tuple[int | None, float, float]:
        """One progress tick: lazy-resolve unknown size, then measure + log.

        On the *first* tick for a download with unknown total size,
        runs ``s5cmd du`` to learn the source size and rebases the
        warn/kill thresholds from it.  On every tick (first or
        subsequent), measures destination-side progress and emits a
        single log line.

        Returns the (possibly updated) ``(total, warn, kill)`` so the
        caller can keep tracking across ticks.
        """
        if first_tick and current_total is None and direction == "download":
            resolved = _s5cmd_du(
                _s3_paths(ops, side="src"), self._env, self._endpoint,
            )
            if resolved:
                current_total = resolved
                current_warn, current_kill = self._thresholds_for(resolved)
                logger.info(
                    "%s attempt %d resolved size: %.1f MiB; "
                    "rebased thresholds (warn=%.0fs kill=%.0fs)",
                    log_label, attempt, resolved / _MIB,
                    current_warn, current_kill,
                )
        # Local dir scan for downloads; ``s5cmd du`` against the
        # destination prefix(es) for uploads.  ``None`` from the
        # lookup degrades to a "total unknown" message instead of
        # crashing the watchdog.
        current = _measure_progress(
            ops, direction, self._env, self._endpoint,
        )
        if current is not None and current_total:
            pct = 100.0 * current / max(current_total, 1)
            logger.info(
                "%s attempt %d progress: %.1f / %.1f MiB (%.0f%%) "
                "at %.0fs elapsed",
                log_label, attempt,
                current / _MIB, current_total / _MIB, pct, elapsed,
            )
        elif current is not None:
            logger.info(
                "%s attempt %d progress: %.1f MiB at %.0fs elapsed "
                "(total unknown)",
                log_label, attempt, current / _MIB, elapsed,
            )
        return current_total, current_warn, current_kill

    @staticmethod
    def _log_kill(
        log_label: str, attempt: int, elapsed: float, kill_after: float,
        current_total: int | None, ops: list[tuple[str | Path, str]],
    ) -> None:
        logger.error(
            "%s attempt %d hung %.0fs (>= %.0fs limit, %.1f MiB) on "
            "%d file(s) [%s] — killing process group",
            log_label, attempt, elapsed, kill_after,
            (current_total or 0) / _MIB, len(ops), _sample_paths(ops),
        )

    def _log_warn(
        self,
        log_label: str, attempt: int, elapsed: float, warn_after: float,
        current_total: int | None, ops: list[tuple[str | Path, str]],
    ) -> None:
        logger.warning(
            "%s attempt %d still running after %.0fs (warn budget "
            "%.0fs for %.1f MiB) on %d file(s) [%s] — killing in "
            "%.0fs if no progress",
            log_label, attempt, elapsed, warn_after,
            (current_total or 0) / _MIB, len(ops), _sample_paths(ops),
            self._kill_grace_s,
        )

    def cp(
        self,
        ops: list[tuple[str | Path, str]],
        *,
        total_bytes: int | None = None,
        expected_count: int | None = None,
        log_label: str = "s5cmd",
    ) -> None:
        """Run ``s5cmd cp`` for each ``(src, dst)`` in *ops* via one
        ``s5cmd run`` invocation, with watchdog + retry + JSON
        validation.

        Either side of each pair may be local or ``s3://``.  Glob
        patterns (``"s3://bucket/dir/*"``) are supported on the source
        side; in that case the caller can't supply ``expected_count``
        and we fall back to validating only ``returncode`` + reported
        error records.

        Args:
            ops: ``(src, dst)`` pairs.  Empty list is a no-op.
            total_bytes: Total bytes for the batch, used to compute
                size-scaled watchdog thresholds.  ``None`` falls back
                to ``warn_floor_s``.  Uploads with local sources can
                pass the local-side ``stat().st_size`` sum;
                downloads where the size isn't known upfront should
                pass ``None``.
            expected_count: Number of success records to require.
                ``None`` skips the count check (use for glob downloads
                where the op count expands at the server).  When
                supplied, mismatched success counts raise even if
                ``returncode == 0``.
            log_label: Prefix included in log messages so concurrent
                runners are distinguishable in tail output.

        Raises:
            S5cmdError: on subprocess exit failure, watchdog kill,
                JSON parse error, any reported per-op error, or a
                success-count mismatch when ``expected_count`` was
                supplied.  Retries are wrapped; the final raised
                exception lands after all attempts are exhausted.
        """
        if not ops:
            return
        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._run_once(
                    ops,
                    attempt=attempt,
                    log_label=log_label,
                    total_bytes=total_bytes,
                    expected_count=expected_count,
                )
                return
            except S5cmdRetryableError as exc:
                last_error = exc
                if attempt >= self._max_attempts:
                    break
                backoff = self._retry_backoff_s * attempt
                logger.warning(
                    "%s attempt %d/%d failed, retrying in %.1fs: %s",
                    log_label, attempt, self._max_attempts, backoff, exc,
                )
                time.sleep(backoff)
        raise S5cmdError(
            f"{log_label} failed after {self._max_attempts} attempt(s) "
            f"on {len(ops)} op(s)"
        ) from last_error


    def _run_once(
        self,
        ops: list[tuple[str | Path, str]],
        *,
        attempt: int,
        log_label: str,
        total_bytes: int | None,
        expected_count: int | None,
    ) -> None:
        cmd_input = "\n".join(f"cp {src} {dst}" for src, dst in ops)
        argv = ["s5cmd", "--endpoint-url", self._endpoint, "--json", "run"]

        # Size-scaled watchdog thresholds.  When ``total_bytes`` is
        # not supplied (the typical case for downloads where the
        # server-side glob expands to an unknown size) we fall back
        # to ``warn_floor_s`` for the warn timer; ``kill_grace_s``
        # is added on top.  After ``progress_first_du_s`` the
        # progress monitor will resolve the actual size via
        # ``s5cmd du`` and rebase these thresholds.
        warn_after_s, kill_after_s = self._thresholds_for(total_bytes or 0)
        direction = _direction(ops)

        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env,
            # Own process group: SIGTERM/SIGKILL targets the whole
            # subtree (s5cmd plus any child workers it spawns) and
            # the child won't be orphaned if Python is killed.
            start_new_session=True,
        )

        # Watchdog runs in a sibling thread.  ``communicate()`` does
        # the heavy lifting (writing stdin, draining pipes) without
        # us re-implementing that.  Past the simple kill/warn loop,
        # this thread also drives a periodic progress monitor that
        # resolves unknown total sizes via ``s5cmd du`` and emits
        # one-line progress records every
        # ``progress_poll_interval_s``.
        watchdog_started_at = time.perf_counter()
        stop_watchdog = threading.Event()
        watchdog_killed = [False]

        def watchdog() -> None:
            warned = False
            current_total: int | None = total_bytes
            current_warn = warn_after_s
            current_kill = kill_after_s
            # First progress tick fires at progress_first_du_s; every
            # subsequent tick is progress_poll_interval_s later.
            next_progress_at: float | None = self._progress_first_du_s
            while not stop_watchdog.wait(1.0):
                elapsed = time.perf_counter() - watchdog_started_at
                if elapsed >= current_kill:
                    self._log_kill(
                        log_label, attempt, elapsed, current_kill,
                        current_total, ops,
                    )
                    watchdog_killed[0] = True
                    self._kill_pgroup(proc)
                    return
                if next_progress_at is not None and elapsed >= next_progress_at:
                    first_tick = next_progress_at == self._progress_first_du_s
                    current_total, current_warn, current_kill = self._progress_tick(
                        ops=ops, direction=direction, log_label=log_label,
                        attempt=attempt, elapsed=elapsed, first_tick=first_tick,
                        current_total=current_total,
                        current_warn=current_warn, current_kill=current_kill,
                    )
                    next_progress_at = elapsed + self._progress_poll_interval_s
                if elapsed >= current_warn and not warned:
                    self._log_warn(
                        log_label, attempt, elapsed, current_warn,
                        current_total, ops,
                    )
                    warned = True

        wd_thread = threading.Thread(target=watchdog, daemon=True)
        wd_thread.start()
        try:
            try:
                stdout, stderr = proc.communicate(input=cmd_input)
            except BaseException:
                self._kill_pgroup(proc)
                raise
        finally:
            stop_watchdog.set()
            wd_thread.join(timeout=2.0)

        if watchdog_killed[0]:
            raise S5cmdRetryableError(
                f"s5cmd hung past {kill_after_s:.0f}s "
                f"({(total_bytes or 0) / _MIB:.1f} MiB caller-supplied "
                f"size; progress monitor may have rebased thresholds) "
                f"— killed"
            )

        # Parse first — error counts inform whether to treat
        # ``returncode != 0`` as retryable or terminal.
        successes, errors = _parse_json_records(stdout)
        sample_errors = "; ".join(errors[:3])
        if len(errors) > 3:
            sample_errors += f" ... (+{len(errors) - 3} more)"

        if proc.returncode != 0:
            count_str = (
                f"{successes}/{expected_count}"
                if expected_count is not None else f"{successes}"
            )
            raise S5cmdRetryableError(
                f"s5cmd exited {proc.returncode} "
                f"(reported {count_str} successes, "
                f"{len(errors)} errors): {sample_errors or stderr.strip()[:500]}"
            )
        if errors:
            raise S5cmdRetryableError(
                f"s5cmd exited 0 but reported {len(errors)} error(s): "
                f"{sample_errors}"
            )
        # Critical case: ``returncode == 0`` AND no JSON error
        # records, but the success count doesn't match what the
        # caller asked for.  This is the SIGTERM-graceful-shutdown
        # failure mode: ``s5cmd`` cancelled in-flight ops and exited
        # cleanly without flagging the cancellation as an error.
        # Returning silently here would let the caller commit Lance
        # metadata for files that aren't on S3.
        #
        # Glob downloads pass ``expected_count=None`` because the
        # request expands server-side to an unknown number of ops;
        # for those we trust returncode + error records only.
        if expected_count is not None and successes != expected_count:
            raise S5cmdRetryableError(
                f"s5cmd reported {successes} success(es) but "
                f"{expected_count} were requested — partial transfer"
            )

    @staticmethod
    def _kill_pgroup(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error(
                "s5cmd process group %d did not die within 10s of SIGKILL",
                proc.pid,
            )


_MIB = 1024 * 1024


def _sample_paths(ops: list[tuple[str | Path, str]]) -> str:
    """Format a short sample of source paths for log lines."""
    sample = ", ".join(str(p) for (p, _) in ops[:3])
    if len(ops) > 3:
        sample += f", ... (+{len(ops) - 3} more)"
    return sample


def _direction(ops: list[tuple[str | Path, str]]) -> str:
    """Classify a batch as ``"upload"``, ``"download"``, or ``"unknown"``.

    Mixed-direction batches return ``"unknown"`` and skip
    progress monitoring — that's a caller bug, not something the
    runner should silently average over.
    """
    if not ops:
        return "unknown"
    src_remote = [str(s).startswith("s3://") for s, _ in ops]
    dst_remote = [str(d).startswith("s3://") for _, d in ops]
    if not any(src_remote) and all(dst_remote):
        return "upload"
    if all(src_remote) and not any(dst_remote):
        return "download"
    return "unknown"


def _s3_paths(
    ops: list[tuple[str | Path, str]],
    *,
    side: str,
) -> list[str]:
    """Return the ``s3://`` URI from each op's *side* (``"src"`` or ``"dst"``)."""
    if side not in ("src", "dst"):
        raise ValueError(f"side must be 'src' or 'dst', got {side!r}")
    out: list[str] = []
    for src, dst in ops:
        candidate = str(src if side == "src" else dst)
        if candidate.startswith("s3://"):
            out.append(candidate)
    return out


def _local_dir_size_bytes(paths: list[str | Path]) -> int:
    """Sum bytes of files at *paths* (recursively into directories).

    Returns the partial sum if any individual ``stat`` fails — the
    progress monitor degrades gracefully on a transient
    ``FileNotFoundError`` (e.g. ``s5cmd`` writing to a temp file
    we read mid-rename).
    """
    total = 0
    for p in paths:
        path = Path(str(p).rstrip("/"))
        try:
            if path.is_file():
                total += path.stat().st_size
                continue
            if not path.is_dir():
                continue
            for entry in path.rglob("*"):
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
    return total


def _s5cmd_du(
    s3_paths: list[str],
    env: dict[str, str],
    endpoint: str,
    *,
    timeout_s: float = 30.0,
) -> int | None:
    """Sum bytes reported by ``s5cmd --json du`` for each path.

    Returns ``None`` on any failure (timeout, non-zero exit,
    unparseable output).  The progress monitor treats ``None``
    as "unknown" rather than raising — a flaky ``du`` shouldn't
    abort an in-flight transfer.
    """
    if not s3_paths:
        return 0
    argv = [
        "s5cmd", "--endpoint-url", endpoint, "--json", "du", *s3_paths,
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("s5cmd du failed: %s", exc)
        return None
    if proc.returncode != 0:
        logger.debug(
            "s5cmd du exited %d: %s",
            proc.returncode, proc.stderr.strip()[:500],
        )
        return None
    total = 0
    saw_any = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        size = rec.get("size")
        if isinstance(size, int):
            total += size
            saw_any = True
    return total if saw_any else None


def _measure_progress(
    ops: list[tuple[str | Path, str]],
    direction: str,
    env: dict[str, str],
    endpoint: str,
) -> int | None:
    """Bytes that have already moved at this point in the transfer.

    For downloads, that's the destination's local on-disk size.
    For uploads, it's ``s5cmd du`` against the s3 destinations
    (each completed object becomes immediately visible; mid-flight
    multipart parts are not, which is the desired floor).  Returns
    ``None`` for unknown-direction batches or ``s5cmd du`` failures.
    """
    if direction == "download":
        return _local_dir_size_bytes([dst for _, dst in ops])
    if direction == "upload":
        return _s5cmd_du(_s3_paths(ops, side="dst"), env, endpoint)
    return None


def _parse_json_records(stdout: str) -> tuple[int, list[str]]:
    """Parse ``s5cmd --json`` output into success count + error list.

    s5cmd emits one JSON object per line.  Successful ops include
    ``"success": true`` (or at minimum a ``destination`` field);
    errors include an ``"error"`` field.  Lines that fail to parse
    or look ambiguous are counted as errors — we never silently
    treat them as successes.
    """
    successes = 0
    errors: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"unparseable JSON: {line[:200]}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"unexpected JSON shape: {line[:200]}")
            continue
        if obj.get("error"):
            errors.append(str(obj["error"]))
        elif obj.get("success") is True:
            successes += 1
        elif "destination" in obj or "object" in obj:
            # Some s5cmd versions omit the ``success`` field on
            # successful copies; presence of ``destination``/
            # ``object`` and absence of ``error`` is taken as
            # success.
            successes += 1
        else:
            errors.append(f"ambiguous record: {line[:200]}")
    return successes, errors
