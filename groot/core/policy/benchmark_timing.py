# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Opt-in, server-side timing for the N2 policy and model prediction spans."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import torch

BENCHMARK_TIMING_INFO_KEY = "_benchmark_timing"
_INSTRUMENTED_ATTRIBUTE = "_benchmark_timing_instrumented"


def instrument_policy_timing(
    policy: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
    server_process: dict[str, Any] | None = None,
) -> None:
    """Instrument one sequential N2 policy server in place.

    The public policy call is the ``t3 -> t6`` span. The policy's admitted
    prediction execution seam is the ``t4 -> t5`` span. Accelerated policies
    expose that seam as ``policy._run_predict`` so eager, TensorRT, and CUDA
    graph execution share one measurement boundary. Policies without that seam
    fall back to ``policy.model.predict`` for compatibility. Both spans are
    synchronized at their start and end so queued CUDA work is neither omitted
    nor charged to the next request.

    Timing is returned in the policy ``info`` mapping. The function is opt-in
    because synchronization changes production scheduling and adds overhead;
    the benchmark server CLI enables it explicitly.
    """
    if getattr(policy, _INSTRUMENTED_ATTRIBUTE, False):
        raise RuntimeError("policy timing is already instrumented")
    if synchronize is None:
        if not isinstance(server_process, dict):
            raise ValueError(
                "default CUDA synchronization requires live server_process GPU identity"
            )
        gpu = server_process.get("gpu")
        if not isinstance(gpu, dict) or not isinstance(gpu.get("logical_device_index"), int):
            raise ValueError("default CUDA synchronization requires gpu.logical_device_index")
        selected_device = torch.device("cuda", gpu["logical_device_index"])

        def synchronize() -> None:
            torch.cuda.synchronize(selected_device)

    model = getattr(policy, "model", None)
    if callable(getattr(policy, "_run_predict", None)):
        predict_owner = policy
        predict_attribute = "_run_predict"
    elif model is not None and callable(getattr(model, "predict", None)):
        predict_owner = model
        predict_attribute = "predict"
    else:
        raise TypeError("policy timing requires policy._run_predict or policy.model.predict")

    original_get_action = policy.get_action
    original_predict = getattr(predict_owner, predict_attribute)
    active_timing: dict[str, float] | None = None
    predict_calls = 0

    def timed_predict(*args, **kwargs):
        nonlocal predict_calls
        if active_timing is None:
            return original_predict(*args, **kwargs)
        predict_calls += 1
        if predict_calls != 1:
            raise RuntimeError("benchmark timing requires exactly one prediction execution call")
        synchronize()
        start = clock()
        result = original_predict(*args, **kwargs)
        synchronize()
        active_timing["model_predict_ms"] = (clock() - start) * 1000.0
        return result

    def timed_get_action(*args, **kwargs):
        nonlocal active_timing, predict_calls
        if active_timing is not None:
            raise RuntimeError("benchmark policy timing does not support reentrant requests")
        active_timing = {}
        predict_calls = 0
        synchronize()
        start = clock()
        try:
            action, info = original_get_action(*args, **kwargs)
            synchronize()
            active_timing["policy_ms"] = (clock() - start) * 1000.0
            if predict_calls != 1 or "model_predict_ms" not in active_timing:
                raise RuntimeError(
                    "benchmark timing did not observe exactly one prediction execution call"
                )
            if not isinstance(info, dict):
                raise TypeError("policy info must be a dict to carry benchmark timing")
            timed_info = dict(info)
            timed_info[BENCHMARK_TIMING_INFO_KEY] = dict(active_timing)
            return action, timed_info
        finally:
            active_timing = None

    setattr(predict_owner, predict_attribute, timed_predict)
    policy.get_action = timed_get_action
    setattr(policy, _INSTRUMENTED_ATTRIBUTE, True)

    runtime_contract = getattr(policy, "benchmark_runtime_contract", None)
    if isinstance(runtime_contract, dict):
        runtime_contract = dict(runtime_contract)
        runtime_contract["benchmark_timing"] = True
        if server_process is not None:
            runtime_contract["server_process"] = dict(server_process)
        policy.benchmark_runtime_contract = runtime_contract
