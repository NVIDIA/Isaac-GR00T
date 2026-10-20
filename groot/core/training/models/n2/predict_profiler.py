# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Opt-in per-predict profiler for GrootN2 (GROOT_N2_PROFILE=1).

Collects per-stage timings (``profiler.section``) with CUDA events on GPU,
finalized once into the ``profile`` payload ``GrootN2.predict`` returns.
Disabled (the default), every hook is a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
import time
from typing import Any

import torch


def _profile_env_enabled() -> bool:
    raw = os.environ.get("GROOT_N2_PROFILE", "0").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


class _PredictProfiler:
    def __init__(self, *, enabled: bool, device: torch.device | str):
        self.enabled = enabled
        self.device = torch.device(device)
        self.timings_ms: dict[str, float] = defaultdict(float)
        self.metadata: dict[str, Any] = {}
        self._pending_timings: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._finalized = False

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                self._pending_timings.append((name, start_event, end_event))
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] += (time.perf_counter() - start) * 1000.0

    def _finalize(self) -> None:
        if self._finalized or not self.enabled:
            return
        if self.device.type == "cuda" and self._pending_timings:
            torch.cuda.synchronize(self.device)
        untimeable = 0
        for name, start_event, end_event in self._pending_timings:
            try:
                self.timings_ms[name] += float(start_event.elapsed_time(end_event))
            except RuntimeError:
                # events recorded while a CUDA graph was capturing are graph
                # nodes, not timestamps; their spans cannot be timed
                untimeable += 1
        if untimeable:
            self.metadata["untimeable_event_pairs"] = untimeable
        self._pending_timings.clear()
        self._finalized = True

    def to_dict(self) -> dict[str, Any]:
        self._finalize()
        return {
            "enabled": self.enabled,
            "timing_ms": dict(self.timings_ms),
            **self.metadata,
        }


def begin_predict_profile(device: torch.device | str, **metadata: Any) -> _PredictProfiler:
    """Profiler for one ``GrootN2.predict`` call, stamped with its arguments.

    Enabled by ``GROOT_N2_PROFILE`` (off by default = every hook is a no-op).
    """
    profiler = _PredictProfiler(enabled=_profile_env_enabled(), device=device)
    profiler.metadata.update(metadata)
    return profiler


def attach_profile(profiler: _PredictProfiler, payload: dict) -> dict:
    """Attach the finalized profile to a predict payload (no-op when disabled)."""
    if profiler.enabled:
        payload["profile"] = profiler.to_dict()
    return payload
