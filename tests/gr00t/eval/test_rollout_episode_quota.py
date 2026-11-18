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

"""Regression tests for which episodes a parallel rollout counts.

With auto-resetting vector envs and ``terminate_on_success=True``, successes end
early and failures run to the step budget. Counting the first ``n_episodes`` to
finish across all envs therefore over-samples successes. Each env now counts a
fixed number of its own episodes, so the counted set does not depend on how
long episodes take.
"""

from gr00t.eval import rollout_policy
from gr00t.eval.rollout_policy import _collect_rollout_episodes
import numpy as np
import pytest


class ScriptedVectorEnv:
    """Auto-resetting vector env where each sub-env plays a fixed list of episodes.

    ``scripts[i]`` is a list of ``(length, success)`` or ``(length, success, valid)``
    tuples for env ``i``. A success ends with ``terminated``, a failure with
    ``truncated``, both after ``length`` macro-steps.

    With ``final_info=True`` a finished episode's info is moved into ``final_info``,
    as gymnasium 0.29.1's inline autoreset does in the sim venvs; otherwise it stays
    at the top level, as in gymnasium >= 1.0.
    """

    MAX_STEPS = 10_000  # fail fast instead of hanging if the loop never ends

    def __init__(self, scripts, final_info=False):
        self.scripts = [list(script) for script in scripts]
        self.n = len(scripts)
        self.with_valid = any(len(ep) == 3 for script in scripts for ep in script)
        self.final_info = final_info
        self.steps = 0

    def reset(self, seed=None):
        self.episode = [0] * self.n
        self.t = [0] * self.n
        return np.zeros((self.n, 1)), {}

    def step(self, actions):
        self.steps += 1
        assert self.steps <= self.MAX_STEPS, "rollout loop did not stop"
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        success = [False] * self.n
        valid = [[True] for _ in range(self.n)]
        for idx in range(self.n):
            script = self.scripts[idx]
            if self.episode[idx] < len(script):
                length, ok, *rest = script[self.episode[idx]]
            else:
                # Past the end of its script an env keeps running one long failure.
                length, ok, rest = 10**6, False, []
            self.t[idx] += 1
            if self.t[idx] >= length:
                terminated[idx], truncated[idx], success[idx] = ok, not ok, ok
                valid[idx] = [rest[0] if rest else True]
                self.episode[idx] += 1
                self.t[idx] = 0
        infos = {"success": success, "n_env_steps": [1] * self.n}
        if self.with_valid:
            infos["valid"] = valid
        if self.final_info:
            done = terminated | truncated
            # The top level now describes the freshly reset env for the envs that finished.
            infos = {
                "success": [False] * self.n,
                "n_env_steps": [1] * self.n,
                "_final_info": done,
                "final_info": np.array(
                    [
                        {"success": [success[i]], "n_env_steps": 1} if done[i] else None
                        for i in range(self.n)
                    ],
                    dtype=object,
                ),
            }
        return np.zeros((self.n, 1)), np.zeros(self.n), terminated, truncated, infos


class ZeroPolicy:
    def reset(self):
        pass

    def get_action(self, observations):
        return np.zeros((len(observations), 1)), {}


def collect(scripts, n_episodes, final_info=False):
    env = ScriptedVectorEnv(scripts, final_info=final_info)
    return _collect_rollout_episodes(env, ZeroPolicy(), n_episodes, len(scripts), seed=None)


@pytest.mark.parametrize("final_info", [False, True], ids=["gymnasium-1.x", "gymnasium-0.29"])
def test_short_successes_do_not_crowd_out_long_failures(final_info):
    """Env 0 keeps finishing fast successes while env 1's failures are still running.

    Counting the first 4 finishers would give 4 successes from env 0. Each env
    contributes 2, so env 1's two failures are counted.
    """
    success = (2, True)
    failure = (10, False)
    scripts = [[success] * 6, [failure] * 2]
    successes, lengths, _, _ = collect(scripts, n_episodes=4, final_info=final_info)
    assert sorted(successes) == [False, False, True, True]
    assert sorted(lengths) == [2, 2, 10, 10]


def test_uneven_quotas_follow_env_order():
    """5 episodes over 2 envs: env 0 counts 3, env 1 counts 2."""
    successes, lengths, _, _ = collect([[(1, True)] * 5, [(3, False)] * 5], n_episodes=5)
    assert len(successes) == 5
    assert sorted(lengths) == [1, 1, 1, 3, 3]
    assert successes.count(True) == 3


def test_exactly_n_episodes_when_envs_finish_on_the_same_step():
    """All envs finishing together on the last step must not add extra episodes."""
    successes, _, _, _ = collect([[(4, True)] * 3] * 3, n_episodes=4)
    assert len(successes) == 4


def test_invalid_episode_does_not_use_up_the_quota():
    """An invalid episode is recorded (and filtered later) but the env runs another."""
    scripts = [
        [(2, True, False), (2, False, True), (2, True, True)],
        [(3, True, True), (3, True, True)],
    ]
    successes, _, _, infos = collect(scripts, n_episodes=2)
    assert infos["valid"].count(True) == 2
    valid_successes = [s for s, v in zip(successes, infos["valid"]) if v]
    # Env 0's first valid episode is its failure, env 1's is a success.
    assert sorted(valid_successes) == [False, True]


def test_env_that_only_returns_invalid_episodes_raises(monkeypatch):
    """An env that never returns a valid episode would never meet its quota."""
    monkeypatch.setattr(rollout_policy, "_MAX_CONSECUTIVE_INVALID_EPISODES", 5)
    scripts = [[(2, True, True)] * 3, [(2, True, False)] * 50]
    with pytest.raises(RuntimeError, match="Env 1 returned 5 invalid episodes in a row"):
        collect(scripts, n_episodes=4)
