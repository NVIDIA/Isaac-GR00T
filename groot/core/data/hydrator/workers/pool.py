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

"""Thin worker layer for storage I/O (download and upload) via xenna WorkerGroup.

IMPORTANT: This is an INTERNAL component. All operations are managed through
the service layer (Hydrator) and cache manager.

WorkerPool runs a long-lived WorkerGroup and a single consumer thread that
drains a task queue and dispatches to callbacks / upload waiters.
"""

from collections.abc import Callable, Iterator
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import ray

from groot.core.utils.distributed import worker_logging
from groot.core.data.hydrator.workers import worker
from groot.core.utils.distributed import xenna  # type: ignore[attr-defined]

worker_log = worker_logging.get_unified_logger()

# Task: (kind, path, content, profile)
# profile=None → use default credentials_profile from backend_config
# content: bytes (download payload / encoded upload), Path (upload from
# disk, streamed by backend), or None (download requests).
_Task = tuple[str, str, bytes | Path | None, str | None]

_SHUTDOWN = object()  # Sentinel for consumer thread


class WorkerPool:
    """Long-lived worker pool for CacheManager: task queue + consumer thread + callbacks.

    start() creates a WorkerGroup and a thread that feeds the queue to
    map_unordered("process", ...) and dispatches results to on_download_complete,
    on_download_error, or upload waiters. shutdown(wait=True) drains the queue
    and stops the group.
    """

    def __init__(
        self,
        backend: Any = None,  # noqa: ARG002
        backend_type: str = "boto3",
        backend_config: dict | None = None,
        cache_dir: str = "",
        max_workers: int = 4,
        download_timeout: float = 300.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        runtime_env: dict | None = None,
        download_index_db_dir: str | None = None,
    ) -> None:
        self.backend_type = backend_type
        self.backend_config = backend_config or {}
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.download_timeout = download_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._runtime_env = runtime_env
        self._download_index_db_dir = download_index_db_dir

        self.on_download_complete: Callable[[str, str], None] | None = None
        self.on_download_error: Callable[[str, Exception], None] | None = None
        # Upload-side callbacks — symmetric with the download pair so the
        # CacheManager can update its files dict (PENDING → READY/ERROR)
        # for async uploads without blocking the WorkerPool's consumer.
        self.on_upload_complete: Callable[[str], None] | None = None
        self.on_upload_error: Callable[[str, Exception], None] | None = None

        self._task_queue: queue.Queue[Any] = queue.Queue()
        self._upload_waiters: dict[str, tuple[threading.Event, bool | None, str | None]] = {}
        self._upload_lock = threading.Lock()
        self._worker_group: xenna.WorkerGroup | None = None
        self._consumer_thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        t0 = time.perf_counter()
        in_ray_cluster = ray.is_initialized()
        worker_log.info("[WorkerPool] start: in_ray_cluster=%s pid=%s", in_ray_cluster, os.getpid())
        if not in_ray_cluster:
            worker_log.info("[WorkerPool] not in cluster, initializing Ray ... pid=%s", os.getpid())
            from groot.core.utils.distributed import ray_init

            ray_init.ray_init_maybe_new_cluster()
            t_ray = time.perf_counter() - t0
            worker_log.info("[WorkerPool] ray.init() took %.2fs pid=%s", t_ray, os.getpid())
        else:
            worker_log.info(
                "[WorkerPool] already in Ray cluster, skipping ray.init() pid=%s", os.getpid()
            )
        t_before_workers = time.perf_counter()
        worker_log.info(
            "[WorkerPool] creating WorkerGroup (num_workers=%d) ... pid=%s",
            self.max_workers,
            os.getpid(),
        )
        # Use cpus=0 so Ray does not reserve CPU slots for these I/O actors. With 16 ranks
        # × 4 workers = 64 actors, requesting 1 CPU each would need 64 free CPUs and
        # they stay pending when training workers already hold cluster CPUs.
        # Propagate GEAR_DATA_CREDENTIALS into the worker runtime_env so actors can
        # resolve gear credential profiles. Ray actors don't inherit job-level env vars.
        # Do NOT build a custom runtime_env just to propagate GEAR_DATA_CREDENTIALS:
        # setting runtime_env on child actors *replaces* the job-level one (which
        # carries working_dir and pip).  GEAR_DATA_CREDENTIALS is already present
        # in the job-level env_vars when Gear provisions the cluster, so the child
        # actors inherit it automatically.  Only pass runtime_env when the caller
        # explicitly provides one via HydratorConfig.runtime_env.
        runtime_env = self._runtime_env

        self._worker_group = xenna.WorkerGroup(
            worker.StorageIOWorker,
            num_workers=self.max_workers,
            resources=xenna.Resources(cpus=0.0),
            worker_kwargs={
                "backend_type": self.backend_type,
                "backend_config": self.backend_config,
                "cache_dir": self.cache_dir,
                "retry_attempts": self.retry_attempts,
                "retry_delay": self.retry_delay,
                "download_index_db_dir": self._download_index_db_dir,
            },
            runtime_env=runtime_env,
        )
        worker_log.info("WorkerPool: calling WorkerGroup.start() (blocks until actors ready) ...")
        worker_log.info(
            "[WorkerPool] calling WorkerGroup.start() (blocks until actors ready) pid=%s",
            os.getpid(),
        )
        self._worker_group.start()
        worker_log.info("[WorkerPool] WorkerGroup.start() returned (pid=%s)", os.getpid())
        t_workers = time.perf_counter() - t_before_workers
        worker_log.info(
            "[WorkerPool] WorkerGroup(start) took %.2fs (num_workers=%d pid=%s)",
            t_workers,
            self.max_workers,
            os.getpid(),
        )
        self._consumer_thread = threading.Thread(target=self._consume_tasks, daemon=False)
        self._consumer_thread.start()
        self._started = True
        t_total = time.perf_counter() - t0
        worker_log.info("[WorkerPool] full start took %.2fs (pid=%s)", t_total, os.getpid())

    def get_pending_task_count(self) -> int:
        """Return number of tasks still in the queue (not yet handed to Ray)."""
        return self._task_queue.qsize()

    def _task_iter(self) -> Iterator[_Task | Any]:
        """Yield tasks from the queue until _SHUTDOWN.

        Yields NO_ITEM when empty (timeout) so WorkerGroup can keep draining
        in_flight without blocking.
        """
        no_item_count = 0
        while True:
            try:
                task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                no_item_count += 1
                if no_item_count == 1:
                    worker_log.info(
                        "[WorkerPool] consumer: queue empty on first pull (queue_size=%s pid=%s)",
                        self._task_queue.qsize(),
                        os.getpid(),
                    )
                elif no_item_count % 500 == 0:
                    worker_log.info(
                        "[WorkerPool] consumer: still pulling, queue empty %s times (queue_size=%s pid=%s)",
                        no_item_count,
                        self._task_queue.qsize(),
                        os.getpid(),
                    )
                yield xenna.NO_ITEM
                continue
            if task is _SHUTDOWN:
                return
            queue_remaining = self._task_queue.qsize()
            if isinstance(task, tuple) and len(task) >= 2:
                kind, path = task[0], task[1]
                if queue_remaining % 1000 == 0:
                    worker_log.info(
                        "[WorkerPool] task picked from queue kind=%s path=%r queue_remaining=%s pid=%s",
                        kind,
                        path,
                        queue_remaining,
                        os.getpid(),
                    )
            yield task

    def _consume_tasks(self) -> None:
        assert self._worker_group is not None
        worker_log.info("[WorkerPool] consumer thread entered _consume_tasks (pid=%s)", os.getpid())
        worker_log.info(
            "[WorkerPool] consumer: starting map_unordered (pulling from queue) (pid=%s)",
            os.getpid(),
        )
        try:
            for result_count, result in enumerate(
                self._worker_group.map_unordered("process", self._task_iter()), start=1
            ):
                kind, path, value, error_str = result
                if result_count % 1000 == 0:
                    worker_log.info(
                        "[WorkerPool] consumer: result #%s received kind=%s path=%r pid=%s",
                        result_count,
                        kind,
                        path,
                        os.getpid(),
                    )
                try:
                    if kind == "download":
                        if error_str is not None and self.on_download_error is not None:
                            self.on_download_error(path, Exception(error_str))
                        elif value is not None and self.on_download_complete is not None:
                            self.on_download_complete(path, value)
                    elif kind == "upload":
                        with self._upload_lock:
                            waiter = self._upload_waiters.get(path)
                            if waiter is not None:
                                event, _s, _e = waiter
                                self._upload_waiters[path] = (
                                    event,
                                    error_str is None,
                                    error_str,
                                )
                                event.set()
                        # Async-upload callbacks fire AFTER the waiter is
                        # updated so any caller that managed to race a
                        # wait_for_upload against the result still sees a
                        # consistent state. Sync-upload callers
                        # (submit_upload + wait_for_upload) ignore these —
                        # they pull from the waiter directly.
                        if error_str is not None:
                            if self.on_upload_error is not None:
                                self.on_upload_error(path, Exception(error_str))
                        elif self.on_upload_complete is not None:
                            self.on_upload_complete(path)
                    elif kind == "sync_dir" and error_str is not None:
                        worker_log.warning(
                            "[WorkerPool] sync_dir failed path=%r error=%s",
                            path,
                            error_str,
                        )
                except Exception:  # noqa: BLE001
                    worker_log.exception(
                        "[WorkerPool] callback raised for kind=%s path=%r pid=%s — continuing",
                        kind,
                        path,
                        os.getpid(),
                    )
        except Exception:  # noqa: BLE001
            worker_log.exception(
                "[WorkerPool] consumer thread DIED unexpectedly pid=%s — queue_size=%s tasks will stall",
                os.getpid(),
                self._task_queue.qsize(),
            )

    def signal_end_of_input(self) -> None:
        """Signal that no more tasks will be enqueued so the consumer can finish.

        Mostly for testing: call after enqueueing all tasks for a batch so the
        consumer can finish and yield all results. Otherwise the consumer
        blocks on the queue after each result and never yields the rest
        (deadlock). Production use (e.g. CacheManager) is streaming and does
        not need this.
        """
        if self._started:
            self._task_queue.put(_SHUTDOWN)

    def shutdown(self, wait: bool = True) -> None:
        if not self._started:
            return
        self._task_queue.put(_SHUTDOWN)
        if wait and self._consumer_thread is not None:
            self._consumer_thread.join(timeout=self.download_timeout + 10.0)
        if self._worker_group is not None:
            self._worker_group.stop()
        self._worker_group = None
        self._consumer_thread = None
        self._started = False

    def enqueue_download(self, remote_path: str, profile: str | None = None) -> None:
        self._task_queue.put(("download", remote_path, None, profile))

    def enqueue_upload(
        self,
        remote_path: str,
        content: bytes | Path,
        profile: str | None = None,
    ) -> None:
        with self._upload_lock:
            self._upload_waiters[remote_path] = (threading.Event(), None, None)
        self._task_queue.put(("upload", remote_path, content, profile))

    def wait_for_upload(self, remote_path: str) -> bool:
        with self._upload_lock:
            waiter = self._upload_waiters.get(remote_path)
        if waiter is None:
            return False
        event, success, _error = waiter
        if not event.wait(timeout=self.download_timeout):
            return False
        with self._upload_lock:
            entry = self._upload_waiters.get(remote_path)
            if entry is None:
                return False
            _, success, _ = entry
            del self._upload_waiters[remote_path]
        return success is True

    def enqueue_sync_dir(
        self, s3_path: str, local_dir: str, profile: str | None = None
    ) -> None:
        self._task_queue.put(("sync_dir", s3_path, local_dir.encode("utf-8"), profile))
