from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


TOPPRA_SPEED_VALUES = (0.7, 1.0, 1.3, 1.6)
BASELINE_SPEED_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
# Backward-compatible name for callers that specifically mean the TOPPRA constraint scales.
SPEED_SCALES = TOPPRA_SPEED_VALUES


def speed_grid(minimum: float, maximum: float, step: float) -> tuple[float, ...]:
    """Build an inclusive discrete speed grid without accumulating float error."""

    minimum = float(minimum)
    maximum = float(maximum)
    step = float(step)
    if not all(math.isfinite(value) for value in (minimum, maximum, step)):
        raise ValueError("speed-grid parameters must be finite")
    if minimum <= 0.0 or maximum < minimum or step <= 0.0:
        raise ValueError("speed grid requires 0 < minimum <= maximum and step > 0")
    intervals = int(round((maximum - minimum) / step))
    if not math.isclose(minimum + intervals * step, maximum, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("speed range must be exactly divisible by the configured step")
    return tuple(round(minimum + index * step, 12) for index in range(intervals + 1))


def validate_speed_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("speed values must be finite and positive")
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("speed values must be strictly increasing")
    return result


def default_speed_values(execution_backend: str) -> tuple[float, ...]:
    if execution_backend == "toppra":
        return TOPPRA_SPEED_VALUES
    if execution_backend == "interpolation":
        return BASELINE_SPEED_VALUES
    raise ValueError(f"Unsupported Speed-RL execution backend {execution_backend!r}")


@dataclass(frozen=True)
class RainbowConfig:
    feature_dim: int
    hidden_dim: int = 256
    action_count: int = len(TOPPRA_SPEED_VALUES)
    atom_count: int = 121
    support_min: float = 0.0
    support_max: float = 180.0
    gamma: float = 0.99
    learning_rate: float = 1e-4
    tau: float = 0.5
    target_update_interval: int = 50
    max_grad_norm: float = 10.0
    batch_size: int = 128
    replay_capacity: int = 20_000
    learning_starts: int = 128
    per_alpha: float = 0.2
    per_beta_start: float = 0.6
    per_beta_steps: int = 100_000
    priority_epsilon: float = 1e-6
    n_step: int = 3
    n_step_loss_weight: float = 1.0
    noisy_std_init: float = 0.5

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim < 1 or self.action_count < 1 or self.atom_count < 2:
            raise ValueError("network dimensions must be positive")
        if not self.support_min < self.support_max:
            raise ValueError("support_min must be less than support_max")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if (
            self.learning_rate <= 0.0
            or self.tau <= 0.0
            or self.max_grad_norm <= 0.0
            or self.priority_epsilon <= 0.0
        ):
            raise ValueError("optimizer parameters must be positive")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_starts < 1:
            raise ValueError("learning_starts must be positive")
        if not 0.0 <= self.per_alpha <= 1.0:
            raise ValueError("per_alpha must be in [0, 1]")
        if not 0.0 <= self.per_beta_start <= 1.0 or self.per_beta_steps < 1:
            raise ValueError("invalid PER beta schedule")
        if self.n_step < 1 or self.n_step_loss_weight < 0.0:
            raise ValueError("n-step parameters are invalid")
        if self.noisy_std_init <= 0.0:
            raise ValueError("noisy_std_init must be positive")
        numeric = (
            self.support_min,
            self.support_max,
            self.gamma,
            self.learning_rate,
            self.tau,
            self.max_grad_norm,
            self.per_alpha,
            self.per_beta_start,
            self.priority_epsilon,
            self.n_step_loss_weight,
            self.noisy_std_init,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Rainbow parameters must be finite")

    def advance_beta(self, current_beta: float, decision_index: int) -> float:
        """Apply the retained SpeedTuning legacy beta update for one decision."""

        beta = float(current_beta)
        if not 0.0 <= beta <= 1.0:
            raise ValueError("current_beta must be in [0, 1]")
        fraction = min(max(int(decision_index), 0) / self.per_beta_steps, 1.0)
        return beta + fraction * (1.0 - beta)


def epsilon_for_decisions(_activated_decisions: int) -> float:
    """SpeedTuning uses NoisyNet exploration and therefore epsilon is always zero."""

    return 0.0
