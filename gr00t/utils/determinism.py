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

"""Determinism helpers for reproducible evaluation runs.

Seeding is opt-in. If no seed is supplied and the ``GR00T_EVAL_SEED``
environment variable is unset, these helpers are no-ops, so training and
production code paths are unaffected.

When enabled, :func:`seed_everything` seeds Python, NumPy, torch CPU and
torch CUDA RNGs, sets cuDNN to deterministic mode, and (optionally) asks
torch to use deterministic algorithm implementations. This is required
before any historical metric recording so that run-to-run variance on the
same checkpoint is driven by hardware/library noise only, not by unseeded
RNGs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


EVAL_SEED_ENV_VAR = "GR00T_EVAL_SEED"


logger = logging.getLogger(__name__)


def get_eval_seed(default: int | None = None) -> int | None:
    """Read the eval seed from the environment, or return ``default``.

    Returns ``None`` if the env var is unset / empty and no default is given.
    Raises ``ValueError`` if the env var is set but not a valid integer.
    """
    raw = os.environ.get(EVAL_SEED_ENV_VAR)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{EVAL_SEED_ENV_VAR}={raw!r} is not a valid integer") from exc


def seed_everything(seed: int | None = None, *, warn_only: bool = True) -> int | None:
    """Seed all standard RNGs and enable deterministic cuDNN / CUDA kernels.

    Args:
        seed: Seed to apply. If ``None``, reads ``GR00T_EVAL_SEED``; if that
            is also unset, the function returns ``None`` without changing
            any global state.
        warn_only: Forwarded to ``torch.use_deterministic_algorithms``. When
            ``True`` (default), ops without a deterministic implementation
            warn instead of raising.

    Returns:
        The effective seed that was applied, or ``None`` if no seeding was
        done (so callers can pass it downstream unconditionally).
    """
    if seed is None:
        seed = get_eval_seed()
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required by some CUDA kernels when deterministic algorithms are on.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    logger.info("Determinism enabled: seed=%d warn_only=%s", seed, warn_only)
    return seed
