# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Internal Ray actor that wraps any user-defined worker class."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from typing import Any

import ray

logger = logging.getLogger(__name__)


@dataclass
class WorkerMetadata:
    """Metadata about a worker, passed to setup() if the signature accepts it."""

    worker_index: int
    num_workers: int
    node_id: str
    # For gang-scheduled workers:
    rank: int = 0
    world_size: int = 1
    master_addr: str = ""
    master_port: int = 0


def _call_if_exists(instance: Any, method_name: str, **kwargs: Any) -> Any:
    """Call a method on instance if it exists, passing only kwargs the signature accepts."""
    method = getattr(instance, method_name, None)
    if method is None:
        return None
    sig = inspect.signature(method)
    # Filter kwargs to only those the method accepts
    accepted = {}
    for name, param in sig.parameters.items():
        if name in kwargs:
            accepted[name] = kwargs[name]
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # Method accepts **kwargs, pass everything
            accepted = kwargs
            break
    return method(**accepted)


class _WorkerActorBase:
    """A generic actor base — wraps any user-provided Python class.

    NOT decorated with ``@ray.remote`` here so subclasses (one per user
    class via ``make_worker_actor_class``) can apply the decorator with
    Ray's class_name set to the user's class name. This makes Ray's
    dashboard show e.g. ``ChunkLoadWorker`` / ``EpisodeProcessWorker``
    / ``ShardAggregator`` instead of the generic ``WorkerActor``.
    """

    def __init__(self, cls: type, args: tuple, kwargs: dict) -> None:
        logger.info("[WorkerActor] __init__ start: creating %s ...", cls.__name__)
        self._instance = cls(*args, **kwargs)
        logger.info("[WorkerActor] __init__ done: %s", cls.__name__)

    def run_setup(self, worker_metadata: WorkerMetadata) -> None:
        """Call setup() on the wrapped instance, passing worker_metadata if accepted."""
        logger.info("[WorkerActor] run_setup start")
        _call_if_exists(self._instance, "setup", worker_metadata=worker_metadata)
        logger.info("[WorkerActor] run_setup done")

    def call_method(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        """Dynamically dispatch to any method on the wrapped instance.

        Auto-dereferences any ``ray.ObjectRef`` instances inside ``args``
        / ``kwargs``. Ray's RPC layer auto-dereferences top-level
        ObjectRef args passed to ``.remote(...)``, but it does NOT
        recursively descend into tuples / lists / dicts. xenna packs
        the user's single argument as ``args = (item,)``, so when the
        upstream stage emits ObjectRefs (e.g. ChunkLoadWorker.process()
        returns ``list[ObjectRef]`` to avoid a driver round-trip for
        large items), the receiving actor would otherwise see a raw
        ObjectRef instead of the dereferenced value. Manual ray.get
        here closes that gap.
        """
        if any(isinstance(a, ray.ObjectRef) for a in args):
            args = tuple(
                ray.get(a) if isinstance(a, ray.ObjectRef) else a
                for a in args
            )
        if any(isinstance(v, ray.ObjectRef) for v in kwargs.values()):
            kwargs = {
                k: (ray.get(v) if isinstance(v, ray.ObjectRef) else v)
                for k, v in kwargs.items()
            }
        method = getattr(self._instance, method_name)
        return method(*args, **kwargs)

    def get_node_ip(self) -> str:
        """Return this actor's node IP (for SPMD master address discovery)."""
        return ray.util.get_node_ip_address()

    def ping(self) -> bool:
        """Health check."""
        return True


# Cache of dynamically-built Ray actor classes, keyed by user class.
# Each entry is a `ray.remote(...)`-decorated subclass of
# ``_WorkerActorBase`` whose Python class name == user_cls.__name__,
# so Ray's dashboard groups + filters on the user's class name.
_RAY_ACTOR_CLASS_CACHE: dict[type, Any] = {}


def make_worker_actor_class(worker_cls: type) -> Any:
    """Return a Ray-decorated actor class named after ``worker_cls``.

    Subclasses ``_WorkerActorBase`` and applies ``@ray.remote`` so that
    Ray's `class_name` field on each created actor reflects the user
    class (e.g. ``ChunkLoadWorker`` instead of the generic
    ``WorkerActor``). Cached per ``worker_cls`` so repeated calls reuse
    the same class — Ray would treat re-decorated classes as distinct,
    leading to dashboard duplication.
    """
    cached = _RAY_ACTOR_CLASS_CACHE.get(worker_cls)
    if cached is not None:
        return cached
    name = worker_cls.__name__
    # Build a fresh subclass with the user's class name, decorate as
    # a Ray actor. This is the supported way to give Ray actors a
    # custom class_name.
    Subclass = type(name, (_WorkerActorBase,), {"__module__": worker_cls.__module__})
    decorated = ray.remote(Subclass)
    _RAY_ACTOR_CLASS_CACHE[worker_cls] = decorated
    return decorated


# Backwards-compatible alias. Existing code that imports `WorkerActor`
# directly (e.g. `_worker_actor.WorkerActor.options(...).remote(...)`)
# should migrate to `make_worker_actor_class(user_cls)`. The alias is
# the generic decorated form (class_name = "WorkerActor" in the UI).
WorkerActor = ray.remote(type("WorkerActor", (_WorkerActorBase,), {}))
