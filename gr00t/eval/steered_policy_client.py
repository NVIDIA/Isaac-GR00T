"""PolicyClient wrapper that steers GR00T's denoiser via a local LatentActor.

Used in Window 3 (rollout).  On every ``get_action`` call it:

1. Encodes the proprioceptive state via a local StateEncoder copy.
2. Samples ``w ~ π^W(·|s)`` from a local LatentActor copy.
3. Projects ``w`` through a local noise_projector to the denoiser shape.
4. Sends the projected noise alongside the observation to the PolicyServer
   (Window 2), which injects it into the SteerableActionHead.

Weight updates from the RL trainer (Window 1) arrive asynchronously over a
ZeroMQ SUB socket and are hot-swapped into the local modules.

Usage
-----
    policy = SteeredPolicyClient(
        host="localhost",
        port=5555,
        weight_sub_host="<host-ip>",
        weight_sub_port=5557,
        latent_dim=64,
        action_horizon=16,
        action_dim=32,
    )
    # Use as a drop-in replacement for PolicyClient in rollout_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import zmq

# Allow importing from the rl/ package when it is on PYTHONPATH.
_RL_ROOT = str(Path(__file__).resolve().parents[3] / "rl")
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

from networks import LatentActor, StateEncoder  # noqa: E402  # type: ignore[import]

from gr00t.policy.server_client import MsgSerializer, PolicyClient  # noqa: E402


class SteeredPolicyClient(PolicyClient):
    """PolicyClient that injects actor-sampled noise into every get_action request.

    Inherits from PolicyClient so it can be used as a drop-in ``BasePolicy``
    in ``run_rollout_gymnasium_policy``.

    Args:
        host: PolicyServer hostname (Window 2).
        port: PolicyServer port (default 5555).
        weight_sub_host: Host running rl.py's ActorWeightPublisher (Window 1).
        weight_sub_port: PUB port for weight updates (default 5557).
        latent_dim: Dimensionality of the actor's latent vector w.
        action_horizon: GR00T action horizon (denoiser noise shape dim 1).
        action_dim: GR00T max action dim (denoiser noise shape dim 2).
        obs_keys: Sorted list of proprioceptive observation keys that the
            StateEncoder was trained on.  Must match the keys used by
            Window 1's DSRLAgent.  If None, will be inferred from the first
            weight update.
        hidden_dim: LatentActor / StateEncoder MLP width (must match Window 1).
        n_hidden_layers: Number of hidden layers (must match Window 1).
        device: Torch device for local inference (CPU is fine — these are tiny MLPs).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        weight_sub_host: str = "localhost",
        weight_sub_port: int = 5557,
        latent_dim: int = 64,
        action_horizon: int = 16,
        action_dim: int = 32,
        obs_keys: list[str] | None = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(host=host, port=port, strict=False, **kwargs)

        self._latent_dim = latent_dim
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._device = torch.device(device)
        self._obs_keys = obs_keys
        self._hidden_dim = hidden_dim
        self._n_hidden_layers = n_hidden_layers

        # Local modules — initialised lazily after the first weight update
        # because we need obs_space (inferred from state_encoder weights).
        self._actor: LatentActor | None = None
        self._state_encoder: StateEncoder | None = None
        self._noise_projector: nn.Linear | None = None

        # ZMQ SUB socket for weight updates from Window 1.
        self._sub_ctx = zmq.Context()
        self._sub_socket = self._sub_ctx.socket(zmq.SUB)
        self._sub_socket.connect(f"tcp://{weight_sub_host}:{weight_sub_port}")
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        self._sub_socket.setsockopt(zmq.CONFLATE, 1)  # keep only latest msg
        print(f"[SteeredPolicyClient] SUB tcp://{weight_sub_host}:{weight_sub_port}")

        self._weights_received = False

    # ------------------------------------------------------------------
    # Weight subscription
    # ------------------------------------------------------------------

    def _poll_weights(self) -> None:
        """Non-blocking check for a new weight update; apply if available."""
        while self._sub_socket.poll(timeout=0):
            raw = self._sub_socket.recv()
            state_dict = MsgSerializer.from_bytes(raw)
            self._apply_weights(state_dict)

    def _apply_weights(self, state_dict: dict[str, np.ndarray]) -> None:
        """Load weights into local actor, state_encoder, and noise_projector."""

        # Lazily build local modules on the very first weight update.
        if self._actor is None:
            self._init_local_modules(state_dict)

        # Split the flat dict back into per-module state_dicts.
        actor_sd, enc_sd, proj_sd = {}, {}, {}
        for key, arr in state_dict.items():
            tensor = torch.from_numpy(arr)
            if key.startswith("actor."):
                actor_sd[key[len("actor.") :]] = tensor
            elif key.startswith("state_encoder."):
                enc_sd[key[len("state_encoder.") :]] = tensor
            elif key.startswith("noise_projector."):
                proj_sd[key[len("noise_projector.") :]] = tensor

        self._actor.load_state_dict(actor_sd)
        self._state_encoder.load_state_dict(enc_sd)
        self._noise_projector.load_state_dict(proj_sd)
        self._weights_received = True
        print("[SteeredPolicyClient] Weights updated")

    def _init_local_modules(self, state_dict: dict[str, np.ndarray]) -> None:
        """Build LatentActor, StateEncoder, noise_projector from weight shapes.

        The obs_space is inferred from state_encoder weights if obs_keys was
        not provided.  StateEncoder only stores obs_keys and obs_dim — we
        reconstruct obs_dim from the first linear layer's input dimension
        (which *is* obs_dim since StateEncoder has no learned layers — but it
        won't have any weights if there are no learned params).  In practice we
        just need obs_keys and obs_dim to construct it, then weights are loaded.

        Since StateEncoder has no parameters (it's just a reshaper), we only
        need ``obs_keys`` and ``obs_dim``.  We infer ``obs_dim`` = state_dim
        from ``actor.trunk.0.weight`` (shape ``[hidden_dim, state_dim]``).
        """
        # Infer state_dim from actor trunk input layer.
        trunk_weight = state_dict["actor.trunk.0.weight"]
        state_dim = trunk_weight.shape[1]

        # Build LatentActor
        self._actor = LatentActor(
            state_dim=state_dim,
            latent_dim=self._latent_dim,
            hidden_dim=self._hidden_dim,
            n_hidden=self._n_hidden_layers,
        ).to(self._device)
        self._actor.eval()

        # Build StateEncoder as a thin shim — no learned params, just needs
        # obs_keys and obs_dim for the forward pass.
        if self._obs_keys is not None:
            obs_space = {k: (1,) for k in self._obs_keys}  # shapes don't matter for encoder
        else:
            raise ValueError(
                "obs_keys must be provided to SteeredPolicyClient so the "
                "StateEncoder can select the correct observation keys."
            )
        self._state_encoder = StateEncoder(obs_space, str(self._device)).to(self._device)

        # Build noise_projector
        noise_dim = self._action_horizon * self._action_dim
        self._noise_projector = nn.Linear(self._latent_dim, noise_dim, bias=False).to(self._device)

    # ------------------------------------------------------------------
    # Overridden get_action
    # ------------------------------------------------------------------

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample w from local actor, project to noise, send to server."""

        # Always check for fresh weights (non-blocking).
        self._poll_weights()

        data: dict[str, Any] = {"observation": observation, "options": options}

        # If we have actor weights, compute and inject noise.
        if self._weights_received and self._actor is not None:
            noise_np = self._compute_noise(observation)
            data["noise"] = noise_np

        response = self.call_endpoint("get_action", data)
        return tuple(response)

    @torch.no_grad()
    def _compute_noise(self, observation: dict[str, Any]) -> np.ndarray:
        """Encode state, sample w, project to denoiser noise shape."""

        # Build a batched tensor dict for the state encoder.
        obs_tensors = {}
        for k in self._state_encoder.obs_keys:
            arr = np.asarray(observation[k])
            obs_tensors[k] = torch.from_numpy(arr).float().to(self._device)

        state = self._state_encoder(obs_tensors)  # (B, state_dim)
        w, _, _ = self._actor.sample(state)  # (B, latent_dim)
        projected = self._noise_projector(w)  # (B, H * D)

        B = state.shape[0]
        noise = projected.reshape(B, self._action_horizon, self._action_dim)
        return noise.cpu().numpy()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        try:
            self._sub_socket.close()
            self._sub_ctx.term()
        except Exception:
            pass
        super().__del__()
