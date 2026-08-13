# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Lightweight, env-gated host-memory monitor for diagnosing OOMs.

Enable by setting ``GROOT_MEMORY_MONITOR=1``. Starts a daemon thread that logs
node-level memory (``/proc/meminfo``), this process's RSS, and the top RSS
processes on the node, prefixed with ``[memmon]`` and flushed immediately so the
trace survives a SIGKILL (OOMKilled). Use ``phase()`` to label pipeline stages
(e.g. stats preflight vs training) so memory growth can be attributed.

No third-party deps (reads ``/proc`` directly); Linux only.
"""
# ruff: noqa: T201, PLW0603  -- intentional stdout logging + module-singleton state
from __future__ import annotations

from collections import defaultdict
import gc
import os
import threading
import time

_STARTED = False
_PHASE = "init"


def enabled() -> bool:
    return os.environ.get("GROOT_MEMORY_MONITOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def phase(label: str) -> None:
    """Mark the current pipeline phase (logged with each memory sample)."""
    global _PHASE
    _PHASE = label
    if enabled():
        print(f"[memmon-phase] -> {label}", flush=True)


def _meminfo_kb() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    info[key.strip()] = int(parts[0])
    except OSError:
        pass
    return info


def _cgroup_mem_bytes() -> tuple[int | None, int | None]:
    """(current, limit) bytes for this container's memory cgroup, or (None, None).

    This is the metric the OOM killer acts on -- ``/proc/meminfo`` reports the
    host, not the container's cgroup limit.
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            cur = int(fh.read().strip())
        with open("/sys/fs/cgroup/memory.max") as fh:
            raw = fh.read().strip()
        return cur, (None if raw == "max" else int(raw))
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as fh:
            cur = int(fh.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as fh:
            lim = int(fh.read().strip())
        return cur, (None if lim > (1 << 62) else lim)
    except (OSError, ValueError):
        pass
    return None, None


def _self_rss_kb() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _top_rss(n: int = 10) -> list[tuple[int, str, str]]:
    procs: list[tuple[int, str, str]] = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return procs
    for pid in pids:
        try:
            name = ""
            rss = 0
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("Name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
            if rss:
                procs.append((rss, pid, name))
        except (OSError, ValueError):
            continue
    procs.sort(reverse=True)
    return procs[:n]


def object_tally(top_n: int = 6) -> str:
    """One-line gc-based object tally.

    Reports the largest numpy arrays by dtype+shape and the heaviest Python
    types by count. Plain ndarrays are NOT gc-tracked, so they are reached
    through their tracked containers (dict values / list items -- e.g. the
    per-step ``decoded_video`` dicts), deduped by id. Answers *what object*
    holds the RAM.
    """
    import numpy as np

    seen: set[int] = set()
    np_bytes: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # key -> [bytes, count]
    type_count: dict[str, int] = defaultdict(int)
    totals = {"np": 0}

    def _consider(arr: object) -> None:
        aid = id(arr)
        if aid in seen:
            return
        try:
            nbytes = int(arr.nbytes)  # type: ignore[attr-defined]
            key = (str(arr.dtype), tuple(arr.shape))  # type: ignore[attr-defined]
        except AttributeError:
            return
        seen.add(aid)
        np_bytes[key][0] += nbytes
        np_bytes[key][1] += 1
        totals["np"] += nbytes

    for obj in gc.get_objects():
        # gc.get_objects() can return dead weakref proxies; touching them raises
        # ReferenceError, and container iteration can mutate mid-scan.
        try:
            type_count[type(obj).__name__] += 1
            if isinstance(obj, np.ndarray):
                _consider(obj)
            elif isinstance(obj, dict):
                for v in list(obj.values()):
                    if isinstance(v, np.ndarray):
                        _consider(v)
            elif isinstance(obj, list | tuple):
                for v in list(obj):
                    if isinstance(v, np.ndarray):
                        _consider(v)
        except (ReferenceError, RuntimeError):
            continue

    top_np = sorted(np_bytes.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
    np_s = ", ".join(
        f"{dtype}{shape}x{cnt}={tot / 1024**3:.2f}G" for (dtype, shape), (tot, cnt) in top_np
    )
    top_types = sorted(type_count.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    types_s = ", ".join(f"{name}:{cnt}" for name, cnt in top_types)
    return f"ndarray_total={totals['np'] / 1024**3:.2f}GiB top_np[{np_s}] top_types[{types_s}]"


def _g(kb: int) -> float:
    return kb / 1024.0 / 1024.0


def start(interval_s: float = 5.0) -> None:
    """Start the monitor daemon thread once (no-op unless GROOT_MEMORY_MONITOR)."""
    global _STARTED
    if _STARTED or not enabled():
        return
    _STARTED = True

    top_every = max(1, round(15.0 / max(interval_s, 0.1)))

    def _loop() -> None:
        host = os.environ.get("HOSTNAME", "")
        i = 0
        while True:
            mi = _meminfo_kb()
            total = mi.get("MemTotal", 0)
            avail = mi.get("MemAvailable", 0)
            used = total - avail
            cg_cur, cg_lim = _cgroup_mem_bytes()
            if cg_cur is not None and cg_lim:
                cg_s = (
                    f"cgroup={cg_cur / 1024**3:.1f}/{cg_lim / 1024**3:.1f}GiB"
                    f"({100.0 * cg_cur / cg_lim:.0f}%)"
                )
            elif cg_cur is not None:
                cg_s = f"cgroup={cg_cur / 1024**3:.1f}GiB(no-limit)"
            else:
                cg_s = "cgroup=n/a"
            print(
                f"[memmon] host={host} phase={_PHASE} {cg_s} "
                f"hostused={_g(used):.1f}GiB hostavail={_g(avail):.1f}GiB "
                f"self_rss={_g(_self_rss_kb()):.2f}GiB",
                flush=True,
            )
            if i % top_every == 0:
                tops = _top_rss()
                rendered = ", ".join(f"{name}[{pid}]:{_g(rss):.1f}G" for rss, pid, name in tops)
                print(
                    f"[memmon-top] host={host} phase={_PHASE} pid={os.getpid()} {rendered}",
                    flush=True,
                )
            if i % (2 * top_every) == 0:
                try:
                    print(
                        f"[memmon-objects] host={host} phase={_PHASE} pid={os.getpid()} {object_tally()}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[memmon-objects] tally-failed: {exc}", flush=True)
            i += 1
            time.sleep(interval_s)

    threading.Thread(target=_loop, name="groot-memmon", daemon=True).start()
    print(
        f"[memmon] started (interval={interval_s}s); set GROOT_MEMORY_MONITOR to control",
        flush=True,
    )
