# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pipeline: streaming multi-stage data processing built on WorkerGroup.

All stages run concurrently, connected by bounded queues.  Backpressure
propagates naturally: when a downstream queue is full, the upstream stage
blocks on ``put()``.

Example::

    pipeline = Pipeline(stages=[
        StageSpec(Downloader, resources=Resources(cpus=2), num_workers=4),
        StageSpec(Processor, resources=Resources(gpus=1), num_workers=8),
    ])
    for result in pipeline.run(input_data=urls):
        print(result)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import logging
import queue
import threading
from typing import Any, Callable, Hashable

import ray

from groot.core.utils.distributed.xenna._resources import Resources
from groot.core.utils.distributed.xenna.worker_group import WorkerGroup

logger = logging.getLogger(__name__)

_DONE = object()


@dataclass
class _Error:
    """Wrapper to propagate exceptions through queues."""

    exception: BaseException


@dataclass
class StageSpec:
    """Specification for one stage of a pipeline.

    Args:
        worker_cls: Any Python class with a ``process(item) -> result`` method.
        resources: Resource requirements per worker.
        num_workers: Number of workers for this stage.
        batch_size: Number of items to group per process() call.
            If > 1, process() receives a list of items and must return a list.
        worker_args: Positional args for the worker class constructor.
        worker_kwargs: Keyword args for the worker class constructor.
    """

    worker_cls: type
    resources: Resources = field(default_factory=Resources)
    num_workers: int = 1
    batch_size: int = 1
    worker_args: tuple = ()
    worker_kwargs: dict[str, Any] = field(default_factory=dict)
    # Ray scheduling strategy for actors in this stage.
    # None = Ray default (PACK: fills nodes one at a time before spilling).
    # "SPREAD" forces even distribution across available nodes — useful when
    # a stage's per-actor work has a non-CPU bottleneck (network, /tmp I/O,
    # driver-pod memory) that would saturate the head node first under PACK.
    scheduling_strategy: str | None = None
    # Ray actor max_concurrency — number of method calls each actor can
    # process concurrently in threads. None = Ray default of 1 (strictly
    # serial). Set >1 only when the worker's state is thread-safe; useful
    # for single-actor coordinator stages (e.g. ShardAggregator) whose
    # per-message work is light + lockable so concurrent calls aren't
    # gated by the actor's mailbox.
    max_concurrency: int | None = None
    # xenna's per-actor in-flight slot count. Default None → use
    # WorkerGroup's default (2). When max_concurrency>1 on the actor,
    # also raise slots_per_worker to match — otherwise xenna only feeds
    # 2 concurrent process() calls and the actor's other threads sit
    # idle (the bug that hung mecka's ShardAggregator: 32 thread slots
    # available but only 2 ever in flight, so 2 finalize calls dammed
    # the entire pipeline).
    slots_per_worker: int | None = None
    # Optional partitioning: if set, items are deterministically routed
    # to actors by ``hash(partition_key(item)) % num_workers`` instead
    # of being load-balanced across actors. Each actor sees a stable
    # subset of the input stream — good for stages where state is keyed
    # by some identity (e.g. ShardAggregator partitioning by
    # (slice, shard) so each actor maintains its own per-shard state
    # without cross-actor locks). Items destined for a busy actor are
    # buffered locally in xenna; a saturated partition does NOT spill
    # to other actors (that would break the routing guarantee).
    partition_key: Callable[[Any], Hashable] | None = None


# ── Helpers ───────────────────────────────────────────────────────────


def _queue_iter(q: queue.Queue) -> Iterator[Any]:
    """Read items from *q* until the ``_DONE`` sentinel arrives."""
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, _Error):
            raise item.exception
        yield item


def _batch_iter(source: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    """Group items from *source* into lists of *batch_size*."""
    batch: list[Any] = []
    for item in source:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _put_result(
    result: Any,
    out_q: queue.Queue,
    emitted_ref: list[bool],
) -> None:
    """Apply fan-out / filter semantics and put into *out_q*.

    Side-effect: sets ``emitted_ref[0] = True`` if any item placed into
    ``out_q`` is an :class:`ray.ObjectRef`. The owning stage uses this
    to decide whether ``_run_stage`` must defer ``group.stop()`` until
    downstream has resolved every ref. See _run_stage for the full
    rationale.
    """
    if result is None:
        return
    items = result if isinstance(result, list) else [result]
    for item in items:
        if isinstance(item, ray.ObjectRef):
            emitted_ref[0] = True
        out_q.put(item)


def _run_stage(
    spec: StageSpec,
    in_q: queue.Queue,
    out_q: queue.Queue,
    stage_index: int,
    my_done: threading.Event,
    downstream_done: threading.Event,
) -> None:
    """Stage thread body.

    ``my_done`` is set when this stage has finished processing every item
    from upstream (success or failure). The upstream stage waits on this
    before tearing down its own actors.

    ``downstream_done`` is set by the *next* stage when it has finished
    processing all of *our* outputs (or by the final consumer for the
    last stage). We block on it ONLY when this stage emitted at least
    one :class:`ray.ObjectRef` — see "auto-detect" note below.

    Why the wait at all: ``call_method`` lazily resolves ObjectRef args
    at ``_worker_actor.py:84``, so a producer actor must outlive the
    consumer's actual ``ray.get`` call, not just the ref's enqueue. If
    we tear down right when our iterator drains, downstream hits
    OwnerDiedError on any of our refs still in its mailbox.

    --- Auto-detect: pros / cons / TODO ----------------------------------

    Pros:
      - Backwards-compat by default. Stages that emit only plain values
        (no ``ray.put`` inside ``process()``) tear down immediately —
        no idle actors holding CPU reservations after their work is
        done.
      - Stages that emit ObjectRefs get the safety wait automatically;
        no API change, no per-stage opt-in needed.

    Cons:
      - The wait is stage-level: once a stage emits ANY ObjectRef it
        holds ALL its actors alive until downstream is fully done. A
        stage with N actors that each finish quickly but feed a slow
        downstream will pin all N actors' CPU reservations on the
        cluster for the duration, blocking other work from claiming
        those slots.
      - One ref anywhere in the output stream triggers the wait, even
        if the rest of the items are plain values.
      - Detection uses ``isinstance(item, ray.ObjectRef)`` on the items
        passed to ``_put_result``. Refs nested inside containers
        (``{"data": ref}``, dataclass fields, etc.) are NOT detected
        and would silently regress to OwnerDiedError. Current xenna
        usage emits flat lists of refs, so this is safe today.

    TODO — finer-grained teardown for cluster-utilization wins:
      Track refs at producer-actor granularity: tag each emitted ref
      with its producer's actor handle, refcount how many of each
      producer's refs are still outstanding in downstream's mailbox /
      running tasks, and ``ray.kill`` each producer the moment its
      personal refcount hits zero AND the upstream source is exhausted
      for that actor. This keeps ObjectRef-pass throughput while
      releasing CPU reservations as each producer finishes. Costs
      ~80-100 LOC of careful protocol code in worker_group.py + this
      file; only worth implementing when a stage has actors whose work
      is materially shorter-lived than the downstream stage's.

      Separately: detect nested refs (scan containers / dataclass
      fields) to remove the silent-regression hazard noted above.
    """
    group_kwargs: dict[str, Any] = dict(
        num_workers=spec.num_workers,
        resources=spec.resources,
        worker_args=spec.worker_args,
        worker_kwargs=spec.worker_kwargs,
        scheduling_strategy=spec.scheduling_strategy,
        max_concurrency=spec.max_concurrency,
        partition_key=spec.partition_key,
    )
    if spec.slots_per_worker is not None:
        group_kwargs["slots_per_worker"] = spec.slots_per_worker
    group = WorkerGroup(spec.worker_cls, **group_kwargs)
    # Mutated by ``_put_result``: flips to True if this stage ever emits
    # a ray.ObjectRef, gating the downstream-done wait below. Single-
    # element list so _put_result can mutate it in-place without needing
    # a nonlocal/closure.
    emitted_ref = [False]
    try:
        group.start()
        logger.info(
            "Stage %d started: %s x%d", stage_index, spec.worker_cls.__name__, spec.num_workers
        )
        source: Iterable[Any] = _queue_iter(in_q)
        if spec.batch_size > 1:
            source = _batch_iter(source, spec.batch_size)
        for result in group.map_unordered("process", source):
            _put_result(result, out_q, emitted_ref)
    except Exception as e:
        logger.error("Stage %d failed: %s", stage_index, e)
        out_q.put(_Error(e))
    finally:
        # Set BEFORE waiting on downstream_done so the chain can unwind
        # even if our own teardown is blocked.
        my_done.set()
        out_q.put(_DONE)
        if emitted_ref[0]:
            # We emitted ObjectRefs — downstream may still be lazily
            # resolving them in its mailbox. Wait before tearing down.
            downstream_done.wait()
        # Else: every output was a plain value, already materialized in
        # driver RAM at the queue boundary. Producer death cannot orphan
        # any ref → safe to stop immediately.
        group.stop()


def _output_iter(
    out_q: queue.Queue,
    threads: list[threading.Thread],
    consumer_done: threading.Event,
) -> Iterator[Any]:
    """Yield results from the final output queue."""
    try:
        while True:
            item = out_q.get()
            if item is _DONE:
                return
            if isinstance(item, _Error):
                raise item.exception
            yield item
    finally:
        # Signal the last stage that the consumer has fully drained its
        # outputs and it's safe to tear down its actors.
        consumer_done.set()
        for t in threads:
            t.join(timeout=5.0)


# ── Pipeline ──────────────────────────────────────────────────────────


class Pipeline:
    """Streaming multi-stage data pipeline built on WorkerGroups.

    All stages run concurrently.  Data flows through bounded queues from
    stage to stage.  Results that are ``None`` are filtered out; results
    that are ``list`` are flattened (fan-out).
    """

    def __init__(self, stages: list[StageSpec], *, queue_size: int = 0) -> None:
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self._stages = stages
        self._queue_size = queue_size

    def run(self, input_data: Iterable[Any]) -> Iterator[Any]:
        """Run *input_data* through all stages.  Returns an iterator."""
        queues: list[queue.Queue] = []
        for spec in self._stages:
            size = self._queue_size or spec.num_workers * 4
            queues.append(queue.Queue(maxsize=size))
        # Output queue — also bounded for end-to-end backpressure
        last_size = self._queue_size or self._stages[-1].num_workers * 4
        queues.append(queue.Queue(maxsize=last_size))

        # Per-stage "I am done processing my upstream's outputs" events.
        # Stage i waits on done_events[i+1] (or consumer_done if last)
        # before tearing down its actors. See _run_stage.
        done_events: list[threading.Event] = [threading.Event() for _ in self._stages]
        consumer_done = threading.Event()

        threads: list[threading.Thread] = []

        # Feeder: pump input_data into the first queue
        def _feed() -> None:
            try:
                for item in input_data:
                    queues[0].put(item)
            except Exception as e:
                queues[0].put(_Error(e))
            finally:
                queues[0].put(_DONE)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        threads.append(feeder)

        # One thread per stage
        for i, spec in enumerate(self._stages):
            my_done = done_events[i]
            downstream_done = (
                done_events[i + 1] if i + 1 < len(self._stages) else consumer_done
            )
            t = threading.Thread(
                target=_run_stage,
                args=(spec, queues[i], queues[i + 1], i, my_done, downstream_done),
                daemon=True,
            )
            t.start()
            threads.append(t)

        return _output_iter(queues[-1], threads, consumer_done)

    def run_to_list(self, input_data: Iterable[Any]) -> list[Any]:
        """Like :meth:`run`, but materializes all results into a list."""
        return list(self.run(input_data))
