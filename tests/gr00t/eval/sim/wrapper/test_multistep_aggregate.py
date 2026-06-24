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

"""Pin :class:`AggregateMethod` and the
``MultiStepWrapper(reward_agg_method=...)`` constructor's fail-fast
validation: every allowed method round-trips, and any unknown method
raises ``ValueError`` at the constructor boundary (not deep inside the
first ``step()``) with both the bad input and the allowed set named.
"""

from __future__ import annotations

from gr00t.eval._horizon_contract import PolicyHorizonSpec
import numpy as np
import pytest


def _import_module():
    """``multistep_wrapper`` imports ``gymnasium``; skip cleanly if it
    is not installed in this venv."""
    try:
        from gr00t.eval.sim.wrapper import multistep_wrapper
    except (ImportError, OSError) as e:
        pytest.skip(f"multistep_wrapper not importable in this env: {e}")
    return multistep_wrapper


# ---------------------------------------------------------------------------
# AggregateMethod Enum + aggregate() per-method correctness
# ---------------------------------------------------------------------------


def test_aggregate_method_enum_members_match_supported_values():
    """``AggregateMethod`` is a string Enum exposing exactly the
    implemented reductions, so the code references typed members rather
    than magic strings."""
    from enum import Enum

    mod = _import_module()
    assert issubclass(mod.AggregateMethod, Enum)
    assert {m.value for m in mod.AggregateMethod} == {"max", "min", "mean", "sum"}
    # str-backed members stay interchangeable with their raw strings.
    assert mod.AggregateMethod.MAX == "max"


@pytest.mark.parametrize(
    "method, data, expected",
    [
        ("max", [1.0, 2.0, 3.0], 3.0),
        ("min", [1.0, 2.0, 3.0], 1.0),
        ("mean", [1.0, 2.0, 3.0], 2.0),
        ("sum", [1.0, 2.0, 3.0], 6.0),
    ],
)
def test_aggregate_returns_expected_value_for_each_allowed_method(method, data, expected):
    """Each method in ``AggregateMethod`` produces the documented
    reduction, whether passed as the raw string or the Enum member."""
    mod = _import_module()
    result = mod.aggregate(np.asarray(data), method=method)
    assert result == pytest.approx(expected)
    # Passing the typed Enum member yields the same result.
    enum_result = mod.aggregate(np.asarray(data), method=mod.AggregateMethod(method))
    assert enum_result == pytest.approx(expected)


def test_aggregate_raises_value_error_with_helpful_message_on_unknown_method():
    """``aggregate``'s catch-all raises ``ValueError`` naming the bad
    method and the allowed set."""
    mod = _import_module()
    with pytest.raises(ValueError) as excinfo:
        mod.aggregate(np.asarray([1.0]), method="median")
    msg = str(excinfo.value)
    assert "median" in msg
    assert "max" in msg, "error message must enumerate the allowed set"


# ---------------------------------------------------------------------------
# MultiStepWrapper.__init__ fail-fast on reward_agg_method
# ---------------------------------------------------------------------------


def _build_dummy_env_kwargs(mod):
    """Minimum kwargs to construct ``MultiStepWrapper``.

    The rejection test raises before wrapper setup; the acceptance test
    uses the stub env's minimal action/observation spaces to complete
    construction.
    """
    import gymnasium as gym  # type: ignore[import-not-found]

    class _StubEnv(gym.Env):
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        observation_space = gym.spaces.Dict()

    env = _StubEnv()
    # Horizon parameters now come from a policy-resolved contract; a
    # vision-only, single-step contract is the minimal stand-in.
    contract = PolicyHorizonSpec(
        n_action_steps=1,
        action_horizon=1,
        video_delta_indices=(0,),
        state_delta_indices=None,
    )
    return dict(
        env=env,
        contract=contract,
        max_episode_steps=10,
    )


def test_multistep_wrapper_init_rejects_unknown_reward_agg_method():
    """Bad ``reward_agg_method`` raises at construction time, not on
    the first ``step()``, and the error names the allowed set."""
    mod = _import_module()
    try:
        kwargs = _build_dummy_env_kwargs(mod)
    except (ImportError, OSError) as e:
        pytest.skip(f"gymnasium not importable: {e}")

    with pytest.raises(ValueError) as excinfo:
        mod.MultiStepWrapper(reward_agg_method="median", **kwargs)
    msg = str(excinfo.value)
    assert "median" in msg
    assert "max" in msg, "error message must enumerate the allowed set"


@pytest.mark.parametrize("method", ["max", "min", "mean", "sum"])
def test_multistep_wrapper_init_accepts_each_allowed_method(method):
    """Every value in ``AggregateMethod`` constructs cleanly."""
    mod = _import_module()
    try:
        kwargs = _build_dummy_env_kwargs(mod)
    except (ImportError, OSError) as e:
        pytest.skip(f"gymnasium not importable: {e}")

    wrapper = mod.MultiStepWrapper(reward_agg_method=method, **kwargs)
    assert wrapper.reward_agg_method == method


# ---------------------------------------------------------------------------
# MultiStepWrapper.step() reports inner env-steps via info["n_env_steps"]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("done_after, expected", [(None, 5), (2, 2)])
def test_step_reports_inner_env_step_count(done_after, expected):
    """``step()`` reports the inner env-steps actually taken in
    ``info["n_env_steps"]``: the full chunk, or fewer when ``done`` fires
    mid-chunk."""
    mod = _import_module()
    try:
        import gymnasium as gym  # type: ignore[import-not-found]
    except Exception as e:
        pytest.skip(f"gymnasium not importable: {e}")

    class _StubEnv(gym.Env):
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        observation_space = gym.spaces.Dict()
        _t = 0

        def reset(self, *, seed=None, options=None):
            self._t = 0
            return {}, {}

        def step(self, action):
            self._t += 1
            return {}, 1.0, done_after is not None and self._t >= done_after, False, {}

    # Horizon parameters now come from a policy-resolved contract (post-#25/#26/#34);
    # a vision-only 5-step contract mirrors _build_dummy_env_kwargs above. n_env_steps
    # is independent of the horizon shape, so a single-frame video window suffices.
    contract = PolicyHorizonSpec(
        n_action_steps=5,
        action_horizon=5,
        video_delta_indices=(0,),
        state_delta_indices=None,
    )
    wrapper = mod.MultiStepWrapper(
        env=_StubEnv(),
        contract=contract,
        max_episode_steps=100,
    )
    wrapper.reset()
    _, _, _, _, info = wrapper.step({"action": np.zeros((5, 1), np.float32)})
    assert info["n_env_steps"] == expected
