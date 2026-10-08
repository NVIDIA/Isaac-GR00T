from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Transition:
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool
    action_mask: BoolArray
    next_action_mask: BoolArray

    def __post_init__(self) -> None:
        state = np.asarray(self.state)
        next_state = np.asarray(self.next_state)
        action_mask = np.asarray(self.action_mask)
        next_action_mask = np.asarray(self.next_action_mask)
        if state.ndim != 1 or next_state.shape != state.shape:
            raise ValueError("state and next_state must be equal one-dimensional vectors")
        if np.any(~np.isfinite(state)) or np.any(~np.isfinite(next_state)):
            raise ValueError("transition states must be finite")
        if action_mask.ndim != 1 or next_action_mask.shape != action_mask.shape:
            raise ValueError("action masks must have equal one-dimensional shapes")
        if not 0 <= int(self.action) < len(action_mask) or not bool(action_mask[self.action]):
            raise ValueError("transition action must be allowed by action_mask")
        if not np.any(next_action_mask) and not self.done:
            raise ValueError("nonterminal transition requires a feasible next action")
        if not np.isfinite(self.reward):
            raise ValueError("transition reward must be finite")


@dataclass(frozen=True)
class NStepTransition:
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool
    action_mask: BoolArray
    next_action_mask: BoolArray
    discount: float
    steps: int


@dataclass(frozen=True)
class RainbowReplayTransition:
    """Aligned one-step and n-step views used by the combined Rainbow loss."""

    one_step: NStepTransition
    n_step: NStepTransition

    def __post_init__(self) -> None:
        if self.one_step.steps != 1:
            raise ValueError("one_step replay view must contain exactly one step")
        if self.one_step.action != self.n_step.action:
            raise ValueError("one-step and n-step replay actions lost alignment")
        if not np.array_equal(self.one_step.state, self.n_step.state):
            raise ValueError("one-step and n-step replay states lost alignment")

    @property
    def action(self) -> int:
        return self.one_step.action

    @property
    def action_mask(self) -> BoolArray:
        return self.one_step.action_mask


def build_n_step_transitions(
    episode: Sequence[Transition],
    *,
    n_step: int,
    gamma: float,
) -> list[NStepTransition]:
    if n_step < 1:
        raise ValueError("n_step must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    result: list[NStepTransition] = []
    for start, first in enumerate(episode):
        reward = 0.0
        final = first
        steps = 0
        for offset in range(n_step):
            index = start + offset
            if index >= len(episode):
                break
            current = episode[index]
            reward += gamma**offset * float(current.reward)
            final = current
            steps += 1
            if current.done:
                break
        result.append(
            NStepTransition(
                state=np.asarray(first.state, dtype=np.float32).copy(),
                action=int(first.action),
                reward=float(reward),
                next_state=np.asarray(final.next_state, dtype=np.float32).copy(),
                done=bool(final.done),
                action_mask=np.asarray(first.action_mask, dtype=np.bool_).copy(),
                next_action_mask=np.asarray(final.next_action_mask, dtype=np.bool_).copy(),
                discount=float(gamma**steps),
                steps=steps,
            )
        )
    return result


def build_rainbow_replay_transitions(
    episode: Sequence[Transition],
    *,
    n_step: int,
    gamma: float,
) -> list[RainbowReplayTransition]:
    one_step = build_n_step_transitions(episode, n_step=1, gamma=gamma)
    multi_step = build_n_step_transitions(episode, n_step=n_step, gamma=gamma)
    return [
        RainbowReplayTransition(one_step=one, n_step=multi)
        for one, multi in zip(one_step, multi_step, strict=True)
    ]


@dataclass(frozen=True)
class ReplaySample:
    transitions: tuple[RainbowReplayTransition, ...]
    indices: NDArray[np.int64]
    weights: FloatArray


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with a fixed-size circular store."""

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self._storage: list[RainbowReplayTransition] = []
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._next_index = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def action_counts(self) -> tuple[int, ...]:
        if not self._storage:
            return ()
        action_count = len(self._storage[0].action_mask)
        counts = np.bincount(
            [transition.action for transition in self._storage],
            minlength=action_count,
        )
        return tuple(int(count) for count in counts)

    def add(
        self,
        transition: RainbowReplayTransition,
        priority: float | None = None,
    ) -> None:
        if priority is None:
            priority = (
                float(np.max(self._priorities[: len(self._storage)])) if self._storage else 1.0
            )
        value = abs(float(priority)) + self.priority_epsilon
        if not np.isfinite(value):
            raise ValueError("priority must be finite")
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next_index] = transition
        self._priorities[self._next_index] = value
        self._next_index = (self._next_index + 1) % self.capacity

    def extend(self, transitions: Sequence[RainbowReplayTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, beta: float) -> ReplaySample:
        size = len(self)
        if batch_size < 1 or batch_size > size:
            raise ValueError(f"batch_size must be in [1, {size}]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        scaled = np.power(self._priorities[:size], self.alpha)
        total = float(np.sum(scaled))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Replay priorities have invalid total mass")
        probabilities = scaled / total
        indices = self._rng.choice(size, size=batch_size, replace=True, p=probabilities)
        weights = np.power(size * probabilities[indices], -beta)
        weights /= float(np.max(weights))
        return ReplaySample(
            transitions=tuple(self._storage[int(index)] for index in indices),
            indices=np.asarray(indices, dtype=np.int64),
            weights=np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(
        self,
        indices: NDArray[np.int64],
        priorities: NDArray[Any],
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if indices.shape != values.shape:
            raise ValueError("indices and priorities must have the same shape")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("replay priority index is out of range")
        if np.any(~np.isfinite(values)):
            raise ValueError("priorities must be finite")
        self._priorities[indices] = np.abs(values) + self.priority_epsilon

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "priority_epsilon": self.priority_epsilon,
            "storage": self._storage,
            "priorities": self._priorities.copy(),
            "next_index": self._next_index,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or float(state["alpha"]) != self.alpha:
            raise ValueError("Replay checkpoint configuration does not match")
        storage = list(state["storage"])
        if len(storage) > self.capacity:
            raise ValueError("Replay checkpoint exceeds configured capacity")
        priorities = np.asarray(state["priorities"], dtype=np.float64)
        if priorities.shape != (self.capacity,):
            raise ValueError("Replay checkpoint priorities have the wrong shape")
        self._storage = storage
        self._priorities = priorities.copy()
        self._next_index = int(state["next_index"])
        self._rng.bit_generator.state = state["rng_state"]
