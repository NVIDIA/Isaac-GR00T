"""ZeroMQ PUSH sender for environment transitions (observations, actions, rewards, etc.) from a rollout loop
to a DSRL implementation."""

from typing import Any

import numpy as np
import zmq

from gr00t.policy.server_client import MsgSerializer


class TransitionSender:
    """Sends environment transitions over a ZeroMQ PUSH socket.

    Designed to be used inside a rollout loop. Each call to `send_reset` or
    `send_step` pushes a serialized message to whatever PULL socket is listening
    on the other end (e.g. DSRL training process).

    Sends are non-blocking (DONTWAIT): if the receiver is not keeping up the
    message is dropped rather than stalling the rollout.

    Args:
        host: Hostname or IP of the receiver (default "localhost").
        port: TCP port the receiver is listening on (default 5556).
    """

    def __init__(self, host: str = "localhost", port: int = 5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{host}:{port}")

    def send_reset(self, obs: dict[str, Any]) -> None:
        """Send the initial observation produced by env.reset().

        Args:
            obs: Batched observation dict returned by the vectorised env.
        """
        msg = {"type": "reset", "obs": obs}
        self._send(msg)

    def send_step(
        self,
        obs: dict[str, Any],
        actions: dict[str, Any],
        next_obs: dict[str, Any],
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
    ) -> None:
        """Send a complete (s, a, r, s', done) transition from env.step().

        Args:
            obs: Observation before the step (batched).
            actions: Actions that were executed (batched).
            next_obs: Observation after the step (batched).
            rewards: Per-environment scalar rewards.
            terminations: Per-environment termination flags.
            truncations: Per-environment truncation flags.
        """
        msg = {
            "type": "step",
            "obs": obs,
            "actions": actions,
            "next_obs": next_obs,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
        }
        self._send(msg)

    def _send(self, msg: dict) -> None:
        try:
            self.socket.send(MsgSerializer.to_bytes(msg), flags=zmq.DONTWAIT)
        except zmq.Again:
            # Receiver not ready / buffer full — drop and continue.
            pass

    def close(self) -> None:
        self.socket.close()
        self.context.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
