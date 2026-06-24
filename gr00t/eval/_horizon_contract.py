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

"""Policy-resolved horizon / delta-indices contract for sim-eval and real-robot
consumers. See :class:`PolicyHorizonSpec`.

(The DROID real-robot client makes the same open-loop-within-chunk check inline:
it runs on a slim robot install without the ``gr00t`` package and cannot import
this module — see ``examples/DROID/main_gr00t.py``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _as_int_tuple(x: Any) -> tuple[int, ...]:
    """Collapse a list / ndarray / tuple of indices into a plain ``int`` tuple.

    A tuple keeps the contract hashable / picklable (it is passed to sim-env
    worker subprocesses) and makes ``==`` comparisons return a bool rather than
    an ndarray.
    """
    if isinstance(x, np.ndarray):
        return tuple(int(v) for v in x.tolist())
    return tuple(int(v) for v in x)


@dataclass(frozen=True)
class PolicyHorizonSpec:
    """Single source of truth for the policy-coupled horizons that the sim-eval
    wrapper and real-robot clients must agree on.

    Built from the policy's modality config (see :meth:`from_policy`), so the
    observation-interface fields cannot drift from the policy.

    Fields:
        n_action_steps: open-loop execution length — how many actions of the
            predicted chunk are stepped before re-planning. The only tunable
            knob; validated to lie in ``[1, action_horizon]``. Equals
            ``action_horizon`` for full-chunk execution; a smaller value is
            deliberate receding-horizon re-planning (e.g. LIBERO 8/16, GR1 8/40).
        action_horizon: length of the predicted action chunk
            (``len(action.delta_indices)``); the upper bound on
            ``n_action_steps``.
        video_delta_indices: per-step observation offsets for the video stream,
            copied verbatim from the policy (e.g. ``(0,)`` single-frame,
            ``(-15, 0)`` for DROID). Fed to ``MultiStepWrapper``.
        state_delta_indices: the same for the proprio/state stream, or ``None``
            for a vision-only policy that has no state stream.
    """

    n_action_steps: int
    action_horizon: int
    video_delta_indices: tuple[int, ...]
    state_delta_indices: tuple[int, ...] | None

    @classmethod
    def from_policy(
        cls,
        policy: Any,
        *,
        n_action_steps: int | None = None,
    ) -> "PolicyHorizonSpec":
        """Resolve the spec from a policy exposing ``get_modality_config``.

        Args:
            policy: any object with ``get_modality_config()`` returning a
                ``{modality: ModalityConfig}`` mapping (``Gr00tSimPolicyWrapper``,
                ``Gr00tPolicy``, ``PolicyClient``, ...).
            n_action_steps: see the :attr:`n_action_steps` field. ``None``
                executes the full chunk (``n_action_steps == action_horizon``).
        """
        return cls.from_modality_config(
            policy.get_modality_config(),
            n_action_steps=n_action_steps,
        )

    @classmethod
    def from_modality_config(
        cls,
        modality_config: dict[str, Any],
        *,
        n_action_steps: int | None = None,
    ) -> "PolicyHorizonSpec":
        """Resolve the spec from a raw ``{modality: ModalityConfig}`` dict.

        ``n_action_steps`` is the open-loop execution length (see the
        :attr:`n_action_steps` field); ``None`` defaults it to the full
        ``action_horizon``.
        """
        if "action" not in modality_config:
            raise ValueError(
                "policy modality config has no 'action' entry; cannot resolve "
                "the action horizon. Available keys: "
                f"{sorted(modality_config.keys())}."
            )

        action_delta = list(modality_config["action"].delta_indices)
        action_horizon = len(action_delta)
        if action_horizon < 1:
            raise ValueError("policy declared an empty action.delta_indices.")
        # MultiStepWrapper.step and the DROID client both index the predicted
        # chunk linearly (chunk[i] is the i-th executed action), so the policy
        # must predict a dense, contiguous window starting at 0. A sparse
        # window (e.g. [0, 4, 8]) would silently execute the wrong rows.
        if action_delta != list(range(action_horizon)):
            raise ValueError(
                f"action.delta_indices={action_delta} is not the contiguous "
                f"range(0, {action_horizon}). Consumers index the predicted "
                "action chunk linearly, so a sparse / shifted window would "
                "silently execute the wrong actions."
            )

        if n_action_steps is None:
            n_action_steps = action_horizon
        else:
            n_action_steps = int(n_action_steps)
        if not (1 <= n_action_steps <= action_horizon):
            raise ValueError(
                f"n_action_steps={n_action_steps} must satisfy "
                f"1 <= n_action_steps <= action_horizon={action_horizon}. "
                "n > action_horizon would IndexError in MultiStepWrapper.step; "
                "n < 1 would execute nothing."
            )

        video_delta_indices = _as_int_tuple(modality_config["video"].delta_indices)

        state_cfg = modality_config.get("state")
        if state_cfg is not None and len(state_cfg.delta_indices) > 0:
            state_delta_indices: tuple[int, ...] | None = _as_int_tuple(state_cfg.delta_indices)
        else:
            # Vision-only policy: no state stream.
            state_delta_indices = None

        return cls(
            n_action_steps=n_action_steps,
            action_horizon=action_horizon,
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
        )

    @property
    def video_delta_indices_array(self) -> np.ndarray:
        """``video_delta_indices`` as the ndarray ``MultiStepWrapper`` expects."""
        return np.array(self.video_delta_indices)

    @property
    def state_delta_indices_array(self) -> np.ndarray | None:
        """``state_delta_indices`` as an ndarray, or ``None`` if vision-only."""
        if self.state_delta_indices is None:
            return None
        return np.array(self.state_delta_indices)


__all__: Sequence[str] = [
    "PolicyHorizonSpec",
]
