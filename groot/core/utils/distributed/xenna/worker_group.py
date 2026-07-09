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
"""WorkerGroup: a managed group of Ray actors that run the same user-defined class.

This is the core primitive of Xenna — analogous to Java's ExecutorService or
K8s Deployment. It handles actor lifecycle, work dispatch, and result collection.

Dispatch patterns:
    broadcast(method, ...) — call same method on ALL workers, return all results (sync)
    map_unordered(...)     — distribute items, yield in completion order
    map_batched(...)       — like map_unordered() but auto-batches items first

SPMD mode (``resources.spmd=True``):
    Each logical worker becomes a sub-group of ``gpus`` actors, one per GPU.
    All actors in a sub-group receive the same call; only rank 0's result is
    returned.  WorkerMetadata provides rank, world_size, master_addr,
    master_port for NCCL rendezvous.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
import logging
import time
from typing import Any, Callable, Hashable

import ray

from groot.core.utils.distributed.xenna import _resources, _worker_actor

logger = logging.getLogger(__name__)

_DEFAULT_SLOTS_PER_WORKER = 2
_SPMD_BASE_PORT = 29500
# Cap on how long ``map_unordered`` blocks in ``ray.wait`` before
# returning to top up the in-flight set. Bounds the latency between
# "an item arrives in a queue-backed iterator" and "it gets dispatched
# to an actor" — without this, a slow in-flight task would keep
# ``ray.wait`` blocked for the duration of that task and any items
# that arrive in the meantime would sit in the queue. 0.5s is small
# enough that downstream steady-state throughput isn't perturbed for
# fast tasks, large enough that idle polls don't burn CPU.
_RAY_WAIT_REFILL_INTERVAL = 0.5


class WorkerGroup:
    """A managed group of Ray actors running the same user-defined class.

    The worker class can be any Python class. It does NOT need to inherit from
    any base class. If the class has a ``setup()`` method, it will be called
    after the actor is created. ``setup()`` may optionally accept a
    ``worker_metadata`` argument for rank/node information.

    Pass ``runtime_env={"py_executable": sys.executable}`` to force workers to
    use the driver's Python (avoids deserialization errors when versions differ).

    Example::

        class Trainer:
            def setup(self):
                self.model = load_model()

            def train(self, step: int) -> float:
                return self.model.train(step)

        with WorkerGroup(Trainer, num_workers=8, resources=Resources(gpus=1)) as g:
            losses = g.broadcast("train", step=0)

    SPMD example::

        class TPWorker:
            def setup(self, worker_metadata):
                # worker_metadata.rank, .world_size, .master_addr, .master_port
                init_nccl(worker_metadata)

            def process(self, batch):
                return self.model.forward(batch)

        with WorkerGroup(TPWorker, num_workers=4, resources=Resources(gpus=2, spmd=True)) as g:
            # 4 logical workers, each a sub-group of 2 GPU actors
            results = list(g.map_unordered("process", batches))
    """

    def __init__(
        self,
        worker_cls: type,
        *,
        num_workers: int,
        resources: _resources.Resources = _resources.Resources(),
        worker_args: tuple = (),
        worker_kwargs: dict[str, Any] | None = None,
        slots_per_worker: int = _DEFAULT_SLOTS_PER_WORKER,
        runtime_env: dict[str, Any] | None = None,
        scheduling_strategy: str | None = None,
        max_concurrency: int | None = None,
        partition_key: Callable[[Any], Hashable] | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self._worker_cls = worker_cls
        self._num_workers = num_workers
        self._resources = resources
        self._worker_args = worker_args
        self._worker_kwargs = worker_kwargs or {}
        self._slots_per_worker = slots_per_worker
        self._runtime_env = runtime_env if runtime_env is not None else {}
        # Ray scheduling strategy ("SPREAD", "DEFAULT", or PlacementGroupSchedulingStrategy).
        # None = Ray default (PACK: fills nodes one at a time before spilling).
        self._scheduling_strategy = scheduling_strategy
        # Ray actor max_concurrency — number of method calls the actor
        # can process concurrently (in threads). Default None = Ray
        # default of 1 (serial). Set >1 for actors whose `process()` is
        # mostly I/O-bound or whose state is thread-safe (e.g.
        # ShardAggregator with locked _states).
        self._max_concurrency = max_concurrency
        # Optional partition key callable — when set, map_unordered
        # routes items deterministically by hash(partition_key(item))
        # instead of load-balancing. Each actor handles its own subset
        # of the input stream, so its state can be local and lock-free
        # (no cross-actor concurrency). Items for a busy partition are
        # buffered in xenna (cannot spill to other actors without
        # breaking the routing guarantee).
        self._partition_key = partition_key
        self._spmd = resources.spmd
        self._actors: list[ray.actor.ActorHandle] = []  # rank-0 actors (dispatch targets)
        self._sub_groups: list[list[ray.actor.ActorHandle]] = []  # SPMD only
        self._started = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Create all Ray actors and call setup() on each. Blocks until ready."""
        if self._started:
            raise RuntimeError("WorkerGroup already started")

        if self._spmd:
            self._start_spmd()
        else:
            self._start_normal()

        self._started = True
        logger.info(
            "WorkerGroup started: %d x %s running %s%s",
            self._num_workers,
            self._resources,
            self._worker_cls.__name__,
            f" (SPMD, {int(self._resources.gpus)} GPUs/worker)" if self._spmd else "",
        )

    def _start_normal(self) -> None:
        ray_options = self._build_ray_options()
        # Build (or reuse) a Ray actor class subclassed from
        # _WorkerActorBase whose name is the user's class name. This
        # makes the dashboard show real names (ChunkLoadWorker /
        # EpisodeProcessWorker / ShardAggregator) instead of one
        # generic "WorkerActor" for all stages.
        ray_actor_cls = _worker_actor.make_worker_actor_class(self._worker_cls)
        self._actors = [
            ray_actor_cls.options(
                **ray_options,
                # Per-actor unique name lets the dashboard distinguish
                # instances + makes `ray.get_actor(name)` work for ad-hoc
                # debugging. Names must be cluster-unique; include the
                # actor's worker_index for that.
                name=f"{self._worker_cls.__name__}-{i}-{id(self):x}",
            ).remote(
                self._worker_cls, self._worker_args, self._worker_kwargs
            )
            for i in range(self._num_workers)
        ]
        logger.info(
            "[WorkerGroup] start: actor refs created, submitting run_setup and waiting (ray.get) ..."
        )

        # Call setup() on all actors in parallel
        metadata_list = [
            _worker_actor.WorkerMetadata(
                worker_index=i,
                num_workers=self._num_workers,
                node_id="",
            )
            for i in range(self._num_workers)
        ]
        setup_refs = [
            actor.run_setup.remote(meta) for actor, meta in zip(self._actors, metadata_list)
        ]
        ray.get(setup_refs)

    def _start_spmd(self) -> None:
        world_size = int(self._resources.gpus)
        ray_options = self._build_spmd_ray_options()

        # Create all actors: num_workers sub-groups, each with world_size actors.
        # Use the user-class-named Ray actor class so the dashboard
        # distinguishes stages by class_name.
        ray_actor_cls = _worker_actor.make_worker_actor_class(self._worker_cls)
        all_actors: list[list[ray.actor.ActorHandle]] = []
        for sg_idx in range(self._num_workers):
            sub_group = [
                ray_actor_cls.options(
                    **ray_options,
                    name=f"{self._worker_cls.__name__}-{sg_idx}r{rank}-{id(self):x}",
                ).remote(
                    self._worker_cls, self._worker_args, self._worker_kwargs
                )
                for rank in range(world_size)
            ]
            all_actors.append(sub_group)

        # Discover rank-0 node IPs for NCCL master address
        rank0_ip_refs = [sg[0].get_node_ip.remote() for sg in all_actors]
        rank0_ips = ray.get(rank0_ip_refs)

        # Setup all actors with SPMD metadata
        setup_refs = []
        for worker_idx, sub_group in enumerate(all_actors):
            master_addr = rank0_ips[worker_idx]
            master_port = _SPMD_BASE_PORT + worker_idx
            for rank, actor in enumerate(sub_group):
                meta = _worker_actor.WorkerMetadata(
                    worker_index=worker_idx,
                    num_workers=self._num_workers,
                    node_id="",
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                )
                setup_refs.append(actor.run_setup.remote(meta))
        ray.get(setup_refs)

        self._sub_groups = all_actors
        self._actors = [sg[0] for sg in all_actors]  # rank-0 actors for dispatch

    def stop(self) -> None:
        """Terminate all actors and release resources."""
        if self._spmd:
            for sub_group in self._sub_groups:
                for actor in sub_group:
                    ray.kill(actor)
            self._sub_groups.clear()
            self._actors.clear()  # rank-0 refs already killed via sub_groups
        else:
            for actor in self._actors:
                ray.kill(actor)
            self._actors.clear()
        self._started = False

    def __enter__(self) -> WorkerGroup:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ── Dispatch: broadcast ────────────────────────────────────────────

    def broadcast(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call the same method with the same args on ALL workers.

        Returns a list of results, one per logical worker, in worker order.
        For SPMD, all ranks in each sub-group execute the call; only rank 0's
        result is returned.
        """
        self._check_started()
        if self._spmd:
            all_refs = []
            for sub_group in self._sub_groups:
                for actor in sub_group:
                    all_refs.append(actor.call_method.remote(method, args, kwargs))
            all_results = ray.get(all_refs)
            # Return rank-0 results only
            world_size = len(self._sub_groups[0])
            return [all_results[i * world_size] for i in range(len(self._sub_groups))]
        else:
            refs = [actor.call_method.remote(method, args, kwargs) for actor in self._actors]
            return ray.get(refs)

    # ── Dispatch: map_unordered ───────────────────────────────────────

    def map_unordered(self, method: str, items: Iterable[Any], **kwargs: Any) -> Iterator[Any]:
        """Distribute items across workers, yield results in completion order.

        ``items`` can be any iterable, including a generator or queue-backed
        iterator. If the iterator yields ``NO_ITEM``, it is skipped (not
        submitted) — this lets queue-backed iterators signal "temporarily empty"
        without blocking ``map_unordered`` from yielding completed results.

        For SPMD, each item is dispatched to all actors in a sub-group;
        only rank 0's result is yielded.

        If ``partition_key`` was set on the WorkerGroup, items are routed
        deterministically by ``hash(partition_key(item)) % num_workers``
        instead of being load-balanced. Each actor sees a stable subset
        of the input stream — useful for stages with per-key state that
        must remain process-local (no cross-actor synchronization).
        Items destined for a saturated partition are buffered locally
        (cannot spill to other actors without breaking routing).
        """
        self._check_started()
        items_iter = iter(items)
        in_flight: dict[ray.ObjectRef, int] = {}  # pending ray refs → actor index
        actor_load: list[int] = [0] * len(self._actors)  # tasks queued per actor
        spmd_peers: dict[ray.ObjectRef, list[ray.ObjectRef]] = {}  # rank-0 ref → peer refs
        # Per-actor pending buffer for partitioned routing — items pulled
        # from items_iter that couldn't be submitted because the target
        # actor was at capacity. Never used for load-balanced routing.
        pending_per_actor: list[deque[Any]] | None = (
            [deque() for _ in range(len(self._actors))]
            if self._partition_key is not None
            else None
        )

        def _dispatch_to(actor_idx: int, item: Any) -> None:
            """Issue the actor.call_method.remote and update bookkeeping."""
            if self._spmd:
                sub_group = self._sub_groups[actor_idx]
                refs = [a.call_method.remote(method, (item,), kwargs) for a in sub_group]
                rank0_ref = refs[0]
                in_flight[rank0_ref] = actor_idx
                spmd_peers[rank0_ref] = refs[1:]
            else:
                ref = self._actors[actor_idx].call_method.remote(method, (item,), kwargs)
                in_flight[ref] = actor_idx
            actor_load[actor_idx] += 1

        def _submit_next() -> bool | None:
            """Pull one item from iterator and submit (or buffer) it.

            Returns True if submitted/buffered, None if NO_ITEM (iterator
            temporarily empty), False if iterator exhausted.
            """
            item = next(items_iter, _SENTINEL)
            if item is _SENTINEL:
                return False
            if item is NO_ITEM:
                return None

            if self._partition_key is not None:
                target_idx = hash(self._partition_key(item)) % len(self._actors)
                if actor_load[target_idx] >= self._slots_per_worker:
                    # Target full — buffer locally; will dispatch when
                    # ray.wait pops a ref for this actor and frees a slot.
                    pending_per_actor[target_idx].append(item)
                    return True
                _dispatch_to(target_idx, item)
                return True

            # Load-balanced routing (the original behavior)
            actor_idx = self._pick_actor(actor_load)
            if actor_idx is None:
                actor_idx = min(range(len(actor_load)), key=lambda i: actor_load[i])
            _dispatch_to(actor_idx, item)
            return True

        def _drain_pending(actor_idx: int) -> None:
            """After actor_idx's slot frees up, submit buffered items for it."""
            if pending_per_actor is None:
                return
            buf = pending_per_actor[actor_idx]
            while buf and actor_load[actor_idx] < self._slots_per_worker:
                _dispatch_to(actor_idx, buf.popleft())

        def _total_in_flight() -> int:
            """Total work the group is tracking — submitted + buffered.

            For load-balanced mode, equals len(in_flight). For partitioned
            mode, includes items waiting in pending_per_actor so capacity
            backpressure stops items_iter pulling when the group is full.
            """
            if pending_per_actor is None:
                return len(in_flight)
            return len(in_flight) + sum(len(p) for p in pending_per_actor)

        def _fill_in_flight() -> bool:
            """Submit tasks until at capacity or iterator is empty/exhausted.

            Returns False if the iterator is exhausted, True otherwise.
            """
            capacity = len(self._actors) * self._slots_per_worker
            while _total_in_flight() < capacity:
                result = _submit_next()
                if result is False:
                    return False  # exhausted
                if result is None:
                    break  # iterator temporarily empty; stop filling
            return True  # not exhausted

        # Unified fill-and-drain loop. Works correctly for both finite iterables
        # and streaming queue-backed iterators (e.g. WorkerPool._task_iter) that
        # may be empty at start and receive items over time.
        exhausted = False
        while True:
            if not exhausted:
                exhausted = not _fill_in_flight()

            if not in_flight:
                # Pending without in_flight is anomalous (would only arise
                # if every actor was idle yet had buffered work — shouldn't
                # happen because _drain_pending fires immediately after a
                # slot frees). Sleep + retry to avoid a hot loop.
                if exhausted and (pending_per_actor is None or not any(pending_per_actor)):
                    break
                time.sleep(_ITER_SLEEP)
                continue

            # Cap the wait so the dispatcher gets a chance to refill from a
            # queue-backed iterator that may have received new items since
            # last fill. Without the timeout, a slow in-flight task keeps
            # this thread parked while items pile up in the queue —
            # producing the "1 download active, N waiting in Python queue"
            # symptom seen in Hydrator's WorkerPool under bursty submission.
            done, _ = ray.wait(
                list(in_flight.keys()),
                num_returns=1,
                timeout=_RAY_WAIT_REFILL_INTERVAL,
            )
            for ref in done:
                actor_idx = in_flight.pop(ref)
                actor_load[actor_idx] -= 1
                if self._spmd:
                    ray.get(spmd_peers.pop(ref))  # wait for peer ranks
                yield ray.get(ref)
                # Now that this actor has a free slot, drain any items
                # that were buffered for it.
                _drain_pending(actor_idx)

    def map_batched(
        self, method: str, items: list[Any], batch_size: int, **kwargs: Any
    ) -> Iterator[Any]:
        """Distribute items in batches via work-stealing. Yields batch results as they complete.

        Splits ``items`` into chunks of ``batch_size``, then dispatches each chunk
        through :meth:`map_unordered`. Useful when the worker method expects a list
        of items rather than a single item (e.g., batch inference).
        """
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        yield from self.map_unordered(method, batches, **kwargs)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def num_workers(self) -> int:
        """Number of logical workers (sub-groups in SPMD mode)."""
        return len(self._actors)

    # ── Internal ───────────────────────────────────────────────────────

    def _check_started(self) -> None:
        if not self._started:
            raise RuntimeError("WorkerGroup not started. Call start() first.")

    def _build_ray_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self._resources.cpus > 0:
            options["num_cpus"] = self._resources.cpus
        if self._resources.gpus > 0:
            options["num_gpus"] = self._resources.gpus
        if self._runtime_env is not None and len(self._runtime_env) > 0:
            options["runtime_env"] = self._runtime_env
        if self._scheduling_strategy is not None:
            options["scheduling_strategy"] = self._scheduling_strategy
        if self._max_concurrency is not None:
            options["max_concurrency"] = self._max_concurrency
        return options

    def _build_spmd_ray_options(self) -> dict[str, Any]:
        """Ray options for SPMD actors: 1 GPU each, inherit cpus and runtime_env."""
        options: dict[str, Any] = {"num_gpus": 1}
        if self._resources.cpus > 0:
            options["num_cpus"] = self._resources.cpus
        if self._runtime_env is not None and len(self._runtime_env) > 0:
            options["runtime_env"] = self._runtime_env
        if self._scheduling_strategy is not None:
            options["scheduling_strategy"] = self._scheduling_strategy
        if self._max_concurrency is not None:
            options["max_concurrency"] = self._max_concurrency
        return options

    def _pick_actor(self, actor_load: list[int]) -> int | None:
        """Pick the actor with the fewest in-flight tasks. Returns None if all at capacity."""
        best_idx = None
        best_load = self._slots_per_worker
        for i, load in enumerate(actor_load):
            if load < best_load:
                best_load = load
                best_idx = i
        return best_idx


# ── Utilities ──────────────────────────────────────────────────────────

_SENTINEL = object()
# Yielded by queue-backed iterators when queue is temporarily empty (no block).
NO_ITEM = object()
_ITER_SLEEP = 0.1
