# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logging for data-loading workers (process-pool and Ray) so output is visible in Ray logs.

Configures a logger that writes to sys.stdout (same stream Ray captures when inherited
by child processes) and optionally to a file when GROOT_DATA_DEBUG_LOG is set.
"""

import atexit
import logging
import os
import sys
from io import TextIOWrapper

import numpy as np
import structlog

UNIFIED_LOGGER_NAME = "groot.workers"
_configured = False

# Sentinel: distinguishes "not configured" from "configured with log_file=None".
_UNSET: object = object()
# Records the log_file that the *current process* has been configured with so a
# second call with a different path raises instead of silently leaking a new
# file handle. force=True bypasses this check.
_configured_log_file: str | None | object = _UNSET
# The single open file handle for the current process's structlog output; closed
# on process exit (and replaced on force-reconfigure to a different path).
_open_log_file: TextIOWrapper | None = None


def _close_open_log_file() -> None:
    global _open_log_file
    fh, _open_log_file = _open_log_file, None
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


atexit.register(_close_open_log_file)

# Silence structlog by default so debug/info calls are no-ops in processes that
# never call configure_logging(). configure_logging(force=True) replaces this
# with the JSON renderer + intended level when struct logs are enabled.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes the stream after each emit so logs appear in Ray/captured stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.stream:
            self.stream.flush()


def get_unified_logger() -> logging.Logger:
    """Return a unified logger for worker processes: stdout (Ray-captured) + optional file."""
    global _configured
    logger = logging.getLogger(UNIFIED_LOGGER_NAME)
    if _configured:
        return logger
    _configured = True
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers when multiple modules call get_unified_logger() in the same process.
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(message)s")
    # Stdout: same stream Ray captures when child processes inherit it.
    stdout_handler = FlushingStreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    # Optional file for tailing when stdout is not captured (e.g. some cluster setups).
    log_path: str | None = os.environ.get("GROOT_DATA_DEBUG_LOG")
    if log_path:
        try:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            pass  # Skip file logging if path is invalid or not writable
    return logger


def _coerce_numpy(logger, method, event_dict):
    """Recursively convert numpy scalars/arrays to native Python types for clean JSON output."""
    def _convert(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return [_convert(i) for i in v]
        if isinstance(v, dict):
            return {k: _convert(vv) for k, vv in v.items()}
        return v
    return {k: _convert(v) for k, v in event_dict.items()}


def _make_file_logger_factory(log_file: str):
    """Logger factory that writes JSON lines only to a file (not stdout).

    Replaces any previously opened handle in this process — callers must ensure
    only one log_file is configured per process (configure_logging enforces this).
    """
    global _open_log_file
    dirname = os.path.dirname(log_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    _close_open_log_file()
    f = open(log_file, "a", buffering=1, encoding="utf-8")  # line-buffered
    _open_log_file = f

    class _FileLogger:
        def msg(self, message: str) -> None:
            f.write(message + "\n")

        err = debug = info = warning = error = critical = msg

    class _Factory:
        def __call__(self, *args):
            return _FileLogger()

    return _Factory()


def configure_logging(
    worker_type: str = "ray_worker",
    level: int | str = logging.INFO,
    log_file: str | None = None,
    force: bool = False,
):
    """Configure structlog for this process.

    At most one `log_file` per process: a second call with a different `log_file`
    raises unless `force=True`. Reusing the same `log_file` is a no-op (only the
    `worker_type` contextvar is rebound). With `force=True`, the previous file
    handle is closed and structlog is fully reconfigured — useful for tests or
    when the import-time WARNING-level wrapper installed below needs to be
    replaced with the real INFO/DEBUG-level filter for an enabled struct-log run.

    Args:
        worker_type: Bound as a ``contextvar`` so every event records the role
            of the emitting process (``ray_worker``, ``training_actor``,
            ``pytorch_dataloader``, ...). Rebound on every call.
        level: Minimum level dispatched to processors; everything below is a
            no-op via ``make_filtering_bound_logger``.
        log_file: When set, structlog writes JSON lines to this file only
            (stdout is bypassed). When None, the JSONRenderer goes to stdout
            (Ray-captured).
        force: Bypass the "at most one log_file per process" assertion and
            re-run ``structlog.configure``. Use sparingly — primarily by the
            trainer / dataloader worker init to replace the import-time
            silent-default wrapper, and by tests to reset state.
    """
    global _configured_log_file
    if not force and _configured_log_file is not _UNSET:
        if _configured_log_file != log_file:
            raise RuntimeError(
                f"configure_logging already called in this process with "
                f"log_file={_configured_log_file!r}; refusing to reconfigure "
                f"to log_file={log_file!r}. Pass force=True to override."
            )
        # Same log_file as before: rebind worker_type and return.
        structlog.contextvars.bind_contextvars(worker_type=worker_type)
        return
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    factory = _make_file_logger_factory(log_file) if log_file else structlog.PrintLoggerFactory(file=sys.stdout)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _coerce_numpy,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=factory,
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
    _configured_log_file = log_file
    structlog.contextvars.bind_contextvars(worker_type=worker_type)
