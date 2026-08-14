# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: ARG002, BLE001

"""Optional torch.compile diagnostics for GrootN2 training."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import os

from transformers import TrainerCallback


def _sum_counter(counter: Mapping | None) -> int:
    if counter is None:
        return 0
    return int(sum(value for value in counter.values() if isinstance(value, int)))


def read_torch_compile_counters() -> dict[str, int]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}

    stats = counters.get("stats", {})
    metrics: dict[str, int] = {}
    for key in ("unique_graphs", "calls_captured"):
        value = stats.get(key)
        if isinstance(value, int):
            metrics[key] = value

    metrics["graph_breaks"] = _sum_counter(counters.get("graph_break"))
    metrics["recompiles"] = _sum_counter(counters.get("recompiles"))
    return metrics


class N2CompileDiagnosticsCallback(TrainerCallback):
    """Log torch.compile counters for early N2 training steps when explicitly enabled."""

    def __init__(self, max_steps: int, interval: int = 1):
        self.max_steps = max_steps
        self.interval = max(interval, 1)
        self.enabled = max_steps > 0

    @classmethod
    def from_env(cls) -> N2CompileDiagnosticsCallback:
        max_steps = int(os.environ.get("GROOT_N2_COMPILE_DIAGNOSTICS_STEPS", "0"))
        interval = int(os.environ.get("GROOT_N2_COMPILE_DIAGNOSTICS_INTERVAL", "1"))
        return cls(max_steps=max_steps, interval=interval)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        logging.info(
            "[GrootN2 compile] diagnostics enabled for first %d steps; "
            "set TORCH_LOGS=recompiles for per-graph recompile reasons",
            self.max_steps,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        step = int(state.global_step)
        if step <= 0 or step > self.max_steps or step % self.interval != 0:
            return

        counters = read_torch_compile_counters()
        if not counters:
            return
        formatted = ", ".join(f"{key}={value}" for key, value in sorted(counters.items()))
        logging.info("[GrootN2 compile] step=%d %s", step, formatted)
