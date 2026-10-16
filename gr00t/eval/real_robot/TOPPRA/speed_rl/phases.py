from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .config import TOPPRA_SPEED_VALUES, validate_speed_values


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    abort: bool = False
    speed_violation: bool = False
    control_fault: bool = False
    tracking_log: str | None = None
    duration_s: float | None = None

    @property
    def safe_success(self) -> bool:
        return (
            self.success and not self.abort and not self.speed_violation and not self.control_fault
        )

    @property
    def valid_for_calibration(self) -> bool:
        return not self.abort and not self.control_fault


@dataclass
class CalibrationState:
    speed_index: int = 0
    valid_episodes_at_speed: int = 0
    awaiting_approval: bool = False
    stopped: bool = False
    complete: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)


class CalibrationManager:
    """Persist an ordered, backend-specific, two-episode-per-speed calibration gate."""

    def __init__(
        self,
        state_path: str | Path,
        speed_values: tuple[float, ...] = TOPPRA_SPEED_VALUES,
    ) -> None:
        self.state_path = Path(state_path)
        self.speed_values = validate_speed_values(speed_values)
        self.state = CalibrationState()
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            stored_speed_values = validate_speed_values(raw.pop("speed_values", ()))
            if stored_speed_values != self.speed_values:
                raise ValueError(
                    "Calibration speed values do not match: "
                    f"checkpoint={stored_speed_values}, runtime={self.speed_values}"
                )
            self.state = CalibrationState(**raw)

    @property
    def fixed_action(self) -> int | None:
        if self.state.complete:
            return None
        if self.state.stopped:
            raise RuntimeError("Calibration is stopped after an abort or control fault")
        if self.state.awaiting_approval:
            raise RuntimeError("Current calibration speed requires explicit tracking approval")
        return self.state.speed_index

    @property
    def speed_scale(self) -> float | None:
        action = self.fixed_action
        return None if action is None else self.speed_values[action]

    def record_episode(self, outcome: EpisodeOutcome, transition_count: int) -> None:
        if self.state.complete:
            raise RuntimeError("Calibration is already complete")
        record = {
            **asdict(outcome),
            "speed_index": self.state.speed_index,
            "speed_scale": self.speed_values[self.state.speed_index],
            "transition_count": int(transition_count),
        }
        self.state.records.append(record)
        if outcome.abort or outcome.control_fault:
            self.state.stopped = True
            self._save()
            return
        if outcome.valid_for_calibration and transition_count > 0:
            self.state.valid_episodes_at_speed += 1
            if self.state.valid_episodes_at_speed >= 2:
                self.state.awaiting_approval = True
        self._save()

    def approve_current_speed(self, tracking_review: str) -> None:
        review = str(tracking_review).strip()
        if not self.state.awaiting_approval:
            raise RuntimeError("No calibration speed is awaiting approval")
        if not review:
            raise ValueError("Approval requires a tracking review note or report path")
        self.state.approvals.append(
            {
                "speed_index": self.state.speed_index,
                "speed_scale": self.speed_values[self.state.speed_index],
                "tracking_review": review,
            }
        )
        self.state.speed_index += 1
        self.state.valid_episodes_at_speed = 0
        self.state.awaiting_approval = False
        if self.state.speed_index >= len(self.speed_values):
            self.state.complete = True
            self.state.speed_index = len(self.speed_values) - 1
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {**asdict(self.state), "speed_values": list(self.speed_values)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)


class OnlineEpisodeBudget:
    def __init__(
        self,
        maximum_episodes: int = 100,
        state_path: str | Path | None = None,
    ) -> None:
        if maximum_episodes < 1:
            raise ValueError("maximum_episodes must be positive")
        self.maximum_episodes = int(maximum_episodes)
        self.completed_episodes = 0
        self.state_path = None if state_path is None else Path(state_path)
        if self.state_path is not None and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(state["maximum_episodes"]) != self.maximum_episodes:
                raise ValueError("Online episode budget checkpoint does not match")
            self.completed_episodes = int(state["completed_episodes"])
            if not 0 <= self.completed_episodes <= self.maximum_episodes:
                raise ValueError("Online episode budget checkpoint is invalid")

    def record(self) -> None:
        if self.completed_episodes >= self.maximum_episodes:
            raise RuntimeError("The configured online-training episode budget is exhausted")
        self.completed_episodes += 1
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "maximum_episodes": self.maximum_episodes,
                        "completed_episodes": self.completed_episodes,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.state_path)

    @property
    def remaining(self) -> int:
        return self.maximum_episodes - self.completed_episodes


@dataclass
class GreedyAcceptance:
    outcomes: list[EpisodeOutcome] = field(default_factory=list)
    state_path: Path | None = None

    def __post_init__(self) -> None:
        if self.state_path is None:
            return
        self.state_path = Path(self.state_path)
        if not self.state_path.exists():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        stored = raw.get("outcomes") if isinstance(raw, dict) else None
        if not isinstance(stored, list):
            raise ValueError("Greedy acceptance checkpoint is invalid")
        self.outcomes = [EpisodeOutcome(**item) for item in stored]
        if len(self.outcomes) > 5:
            raise ValueError("Greedy acceptance checkpoint contains more than five episodes")

    def add(self, outcome: EpisodeOutcome) -> None:
        if len(self.outcomes) >= 5:
            raise RuntimeError("Five greedy acceptance episodes are already recorded")
        self.outcomes.append(outcome)
        self._save()

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"outcomes": [asdict(outcome) for outcome in self.outcomes]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def result(self) -> dict[str, Any]:
        successes = [outcome for outcome in self.outcomes if outcome.safe_success]
        durations = sorted(
            float(outcome.duration_s) for outcome in successes if outcome.duration_s is not None
        )
        median = None
        if durations:
            middle = len(durations) // 2
            median = (
                durations[middle]
                if len(durations) % 2
                else 0.5 * (durations[middle - 1] + durations[middle])
            )
        passed = (
            len(self.outcomes) == 5
            and len(successes) >= 4
            and all(not item.abort and not item.speed_violation for item in self.outcomes)
            and median is not None
            and median < 50.0
        )
        return {
            "episodes": len(self.outcomes),
            "safe_successes": len(successes),
            "median_success_duration_s": median,
            "passed": passed,
        }
