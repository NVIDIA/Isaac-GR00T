from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import random
import threading
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor

from .config import RainbowConfig, default_speed_values, validate_speed_values
from .contract import FeatureContract
from .network import (
    DuelingC51Network,
    cpu_state_dict,
    load_cpu_state_dict,
    project_categorical_distribution,
    select_masked_action,
)
from .replay import (
    NStepTransition,
    PrioritizedReplayBuffer,
    Transition,
    build_rainbow_replay_transitions,
)


class SpeedActor:
    """CPU actor whose weights remain frozen for one real-robot episode."""

    def __init__(
        self,
        config: RainbowConfig,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.network = DuelingC51Network(config).cpu().eval()
        self.policy_version = 0
        self._rng = random.Random(seed)
        self._greedy = True

    def install_weights(self, state_dict: dict[str, Tensor], policy_version: int) -> None:
        load_cpu_state_dict(self.network, state_dict)
        self.policy_version = int(policy_version)
        self.set_greedy(self._greedy)

    def set_greedy(self, greedy: bool) -> None:
        self._greedy = bool(greedy)
        if self._greedy:
            self.network.eval()
        else:
            self.network.train()
            self.network.reset_noise()

    def select(
        self,
        feature: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float,
        fixed_action: int | None = None,
    ) -> int:
        if fixed_action is not None:
            action = int(fixed_action)
            if action < 0 or action >= self.config.action_count:
                raise ValueError("fixed_action is out of range")
            if not bool(np.asarray(action_mask)[action]):
                raise RuntimeError("The fixed calibration speed is currently infeasible")
            return action
        return select_masked_action(
            self.network,
            np.asarray(feature, dtype=np.float32),
            np.asarray(action_mask),
            epsilon=epsilon,
            rng=self._rng,
        )


class RainbowLearner:
    """Rainbow learner state owned by the spawned learner process."""

    def __init__(
        self,
        config: RainbowConfig,
        contract: FeatureContract,
        *,
        seed: int = 0,
        execution_backend: str = "toppra",
        speed_values: Sequence[float] | None = None,
    ) -> None:
        if execution_backend not in {"toppra", "interpolation"}:
            raise ValueError(f"Unsupported Speed-RL execution backend {execution_backend!r}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.config = config
        self.contract = contract
        self.execution_backend = execution_backend
        self.speed_values = validate_speed_values(
            default_speed_values(execution_backend) if speed_values is None else speed_values
        )
        if len(self.speed_values) != config.action_count:
            raise ValueError("Rainbow action_count does not match the configured speed values")
        self.online = DuelingC51Network(config).cpu()
        self.target = DuelingC51Network(config).cpu()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = PrioritizedReplayBuffer(
            config.replay_capacity,
            alpha=config.per_alpha,
            priority_epsilon=config.priority_epsilon,
            seed=seed,
        )
        self.policy_version = 0
        self.update_count = 0
        self.activated_decisions = 0
        self.per_beta = config.per_beta_start
        self._normalization_count = 0
        self._normalization_sum = np.zeros(config.feature_dim, dtype=np.float64)
        self._normalization_sum_squares = np.zeros(config.feature_dim, dtype=np.float64)

    @property
    def online_training_ready(self) -> bool:
        return len(self.replay) >= max(self.config.batch_size, self.config.learning_starts)

    def add_episode(self, transitions: Sequence[Transition]) -> int:
        if not transitions:
            return 0
        converted = build_rainbow_replay_transitions(
            transitions,
            n_step=self.config.n_step,
            gamma=self.config.gamma,
        )
        self.replay.extend(converted)
        previous_decisions = self.activated_decisions
        self.activated_decisions += len(transitions)
        for decision_index in range(previous_decisions + 1, self.activated_decisions + 1):
            self.per_beta = self.config.advance_beta(self.per_beta, decision_index)
        normalization_states = np.stack(
            [transitions[0].state, *(transition.next_state for transition in transitions)]
        ).astype(np.float64)
        self._normalization_count += len(normalization_states)
        self._normalization_sum += normalization_states.sum(axis=0)
        self._normalization_sum_squares += np.square(normalization_states).sum(axis=0)
        self._install_normalization_stats()
        return len(converted)

    def _install_normalization_stats(self) -> None:
        if self._normalization_count < 1:
            return
        mean = self._normalization_sum / self._normalization_count
        variance = self._normalization_sum_squares / self._normalization_count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1e-12))
        self.online.update_norm_stats(mean, std)
        self.target.update_norm_stats(mean, std)

    def _categorical_loss(self, transitions: Sequence[NStepTransition]) -> Tensor:
        states = torch.from_numpy(np.stack([item.state for item in transitions]))
        actions = torch.as_tensor([item.action for item in transitions], dtype=torch.long)
        rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([item.next_state for item in transitions]))
        dones = torch.as_tensor([item.done for item in transitions], dtype=torch.bool)
        next_masks = torch.from_numpy(
            np.stack([item.next_action_mask for item in transitions])
        ).bool()
        discounts = torch.as_tensor([item.discount for item in transitions], dtype=torch.float32)

        logits = self.online(states)
        chosen_logits = logits[torch.arange(len(transitions)), actions]
        log_probabilities = torch.log_softmax(chosen_logits, dim=-1)
        with torch.no_grad():
            online_next_q = self.online.q_values(next_states)
            online_next_q = online_next_q.masked_fill(~next_masks, -torch.inf)
            next_actions = torch.argmax(online_next_q, dim=1)
            target_next_probabilities = self.target.probabilities(next_states)[
                torch.arange(len(transitions)), next_actions
            ]
            target_distribution = project_categorical_distribution(
                target_next_probabilities,
                rewards,
                dones,
                discounts,
                self.online.support,
            )
        return -(target_distribution * log_probabilities).sum(dim=1)

    def train_updates(
        self,
        requested_updates: int,
        *,
        require_online_gate: bool,
    ) -> dict[str, Any]:
        updates = max(0, int(requested_updates))
        if require_online_gate and not self.online_training_ready:
            raise RuntimeError(
                "Online training gate is closed: replay has not reached learning_starts"
            )
        if len(self.replay) < self.config.batch_size:
            return {"updates": 0, "mean_loss": None}

        losses: list[float] = []
        for _ in range(updates):
            sample = self.replay.sample(self.config.batch_size, self.per_beta)
            batch = sample.transitions
            importance = torch.from_numpy(sample.weights)
            one_step_loss = self._categorical_loss([item.one_step for item in batch])
            n_step_loss = self._categorical_loss([item.n_step for item in batch])
            per_item_loss = one_step_loss + self.config.n_step_loss_weight * n_step_loss
            loss = torch.mean(importance * per_item_loss)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.replay.update_priorities(
                sample.indices,
                per_item_loss.detach().cpu().numpy(),
            )
            self.update_count += 1
            if self.update_count % self.config.target_update_interval == 0:
                with torch.no_grad():
                    for target_parameter, online_parameter in zip(
                        self.target.parameters(),
                        self.online.parameters(),
                        strict=True,
                    ):
                        target_parameter.lerp_(online_parameter, self.config.tau)
            self.online.reset_noise()
            self.target.reset_noise()
            self.policy_version += 1
            losses.append(float(loss.detach()))
        return {
            "updates": len(losses),
            "mean_loss": None if not losses else float(np.mean(losses)),
        }

    def actor_snapshot(self) -> dict[str, Any]:
        return {
            "state_dict": cpu_state_dict(self.online),
            "policy_version": self.policy_version,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "execution_backend": self.execution_backend,
            "speed_values": self.speed_values,
            "replay_size": len(self.replay),
            "action_counts": self.replay.action_counts,
            "policy_version": self.policy_version,
            "update_count": self.update_count,
            "activated_decisions": self.activated_decisions,
            "online_training_ready": self.online_training_ready,
            "beta": self.per_beta,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "execution_backend": self.execution_backend,
            "speed_values": self.speed_values,
            "config": asdict(self.config),
            "contract": self.contract.to_dict(),
            "online": cpu_state_dict(self.online),
            "target": cpu_state_dict(self.target),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "policy_version": self.policy_version,
            "update_count": self.update_count,
            "activated_decisions": self.activated_decisions,
            "per_beta": self.per_beta,
            "normalization_count": self._normalization_count,
            "normalization_sum": self._normalization_sum.copy(),
            "normalization_sum_squares": self._normalization_sum_squares.copy(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 2:
            raise ValueError("Unsupported Speed-RL checkpoint format")
        checkpoint_backend = str(state.get("execution_backend", "toppra"))
        if checkpoint_backend != self.execution_backend:
            raise ValueError(
                "Speed-RL checkpoint execution backend does not match: "
                f"checkpoint={checkpoint_backend!r}, runtime={self.execution_backend!r}"
            )
        checkpoint_speeds = validate_speed_values(state.get("speed_values", ()))
        if checkpoint_speeds != self.speed_values:
            raise ValueError(
                "Speed-RL checkpoint speed values do not match: "
                f"checkpoint={checkpoint_speeds}, runtime={self.speed_values}"
            )
        checkpoint_config = RainbowConfig(**state["config"])
        if checkpoint_config != self.config:
            raise ValueError("Speed-RL checkpoint Rainbow configuration does not match")
        self.contract.validate_compatible(FeatureContract.from_mapping(state["contract"]))
        load_cpu_state_dict(self.online, state["online"])
        load_cpu_state_dict(self.target, state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.policy_version = int(state["policy_version"])
        self.update_count = int(state["update_count"])
        self.activated_decisions = int(state["activated_decisions"])
        self.per_beta = float(state["per_beta"])
        if not 0.0 <= self.per_beta <= 1.0:
            raise ValueError("Speed-RL checkpoint PER beta is invalid")
        self._normalization_count = int(state["normalization_count"])
        self._normalization_sum = np.asarray(state["normalization_sum"], dtype=np.float64).copy()
        self._normalization_sum_squares = np.asarray(
            state["normalization_sum_squares"], dtype=np.float64
        ).copy()
        if self._normalization_sum.shape != (
            self.config.feature_dim,
        ) or self._normalization_sum_squares.shape != (self.config.feature_dim,):
            raise ValueError("Speed-RL checkpoint normalization statistics have wrong shape")

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        torch.save(self.checkpoint_state(), temporary)
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Speed-RL checkpoint root must be a mapping")
        self.load_checkpoint_state(state)


def _learner_worker(
    config: RainbowConfig,
    contract: FeatureContract,
    command_queue: Any,
    result_queue: Any,
    seed: int,
    execution_backend: str,
    speed_values: tuple[float, ...],
) -> None:
    torch.set_num_threads(1)
    learner = RainbowLearner(
        config,
        contract,
        seed=seed,
        execution_backend=execution_backend,
        speed_values=speed_values,
    )
    while True:
        command = command_queue.get()
        request_id = command["request_id"]
        operation = command["operation"]
        try:
            if operation == "stop":
                result_queue.put({"request_id": request_id, "ok": True, "result": None})
                return
            if operation == "add_episode":
                result = learner.add_episode(command["transitions"])
            elif operation == "train":
                result = learner.train_updates(
                    command["updates"],
                    require_online_gate=command["require_online_gate"],
                )
            elif operation == "snapshot":
                result = learner.actor_snapshot()
            elif operation == "stats":
                result = learner.stats()
            elif operation == "save":
                learner.save_checkpoint(command["path"])
                result = learner.stats()
            elif operation == "load":
                learner.load_checkpoint(command["path"])
                result = learner.stats()
            else:
                raise ValueError(f"Unknown learner operation {operation!r}")
            result_queue.put({"request_id": request_id, "ok": True, "result": result})
        except BaseException as exc:
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class LearnerProcess:
    """Request/reply interface to a dedicated multiprocessing-spawn learner."""

    def __init__(
        self,
        config: RainbowConfig,
        contract: FeatureContract,
        *,
        seed: int = 0,
        response_timeout_s: float = 120.0,
        execution_backend: str = "toppra",
        speed_values: Sequence[float] | None = None,
    ) -> None:
        resolved_speed_values = validate_speed_values(
            default_speed_values(execution_backend) if speed_values is None else speed_values
        )
        if len(resolved_speed_values) != config.action_count:
            raise ValueError("Rainbow action_count does not match the configured speed values")
        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._results = context.Queue()
        self._process = context.Process(
            target=_learner_worker,
            args=(
                config,
                contract,
                self._commands,
                self._results,
                int(seed),
                execution_backend,
                resolved_speed_values,
            ),
            name="SpeedRLRainbowLearner",
            daemon=True,
        )
        self._process.start()
        self._timeout_s = float(response_timeout_s)
        self._lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False

    def _call(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("Speed-RL learner process is closed")
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            self._commands.put({"request_id": request_id, "operation": operation, **payload})
            try:
                response = self._results.get(timeout=self._timeout_s)
            except Empty as exc:
                if not self._process.is_alive():
                    raise RuntimeError("Speed-RL learner process exited unexpectedly") from exc
                raise TimeoutError(
                    f"Timed out waiting for learner operation {operation!r}"
                ) from exc
            if response.get("request_id") != request_id:
                raise RuntimeError("Speed-RL learner response order was corrupted")
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Speed-RL learner operation failed"))
            return response.get("result")

    def add_episode(self, transitions: Sequence[Transition]) -> int:
        return int(self._call("add_episode", transitions=tuple(transitions)))

    def train(self, updates: int, *, require_online_gate: bool = True) -> dict[str, Any]:
        return dict(
            self._call(
                "train",
                updates=int(updates),
                require_online_gate=bool(require_online_gate),
            )
        )

    def install_actor_weights(self, actor: SpeedActor) -> int:
        snapshot = self._call("snapshot")
        actor.install_weights(snapshot["state_dict"], snapshot["policy_version"])
        return actor.policy_version

    def stats(self) -> dict[str, Any]:
        return dict(self._call("stats"))

    def save(self, path: str | Path) -> dict[str, Any]:
        return dict(self._call("save", path=str(path)))

    def load(self, path: str | Path) -> dict[str, Any]:
        return dict(self._call("load", path=str(path)))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._call("stop")
        finally:
            self._closed = True
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._commands.close()
            self._results.close()

    def __enter__(self) -> LearnerProcess:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
