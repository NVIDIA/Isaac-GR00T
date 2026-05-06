from dataclasses import dataclass
import json
import os
from typing import Any

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    steerable: bool = False
    """Enable noise injection for DSRL online RL.  When True the server
    accepts an optional ``noise`` ndarray in get_action requests and uses
    it as the flow-matching starting point instead of N(0,I)."""


# ---------------------------------------------------------------------------
# Steerable action head (mirrors rl/diffusion_policy_wrapper.py)
# ---------------------------------------------------------------------------


class _SteerableActionHead(Gr00tN1d6ActionHead):
    """Gr00tN1d6ActionHead that uses externally injected noise.

    Applied via ``__class__`` swap — never instantiated directly.
    """

    _injected_noise: torch.Tensor | None = None

    def set_initial_noise(self, noise: torch.Tensor) -> None:
        self._injected_noise = noise

    @torch.no_grad()
    def get_action_with_features(
        self, backbone_features, state_features, embodiment_id, backbone_output
    ):
        from transformers.feature_extraction_utils import BatchFeature

        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        if self._injected_noise is not None:
            actions = self._injected_noise.to(dtype=vl_embeds.dtype, device=device)
            self._injected_noise = None
        else:
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

        dt = 1.0 / self.num_inference_timesteps

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )


def _apply_steerable_head(policy) -> _SteerableActionHead:
    """Swap the loaded model's action_head class in-place and return it."""
    # Unwrap Gr00tSimPolicyWrapper (and similar) which expose the underlying
    # Gr00tPolicy as ``.policy``.
    inner = policy
    while not hasattr(inner, "model") and hasattr(inner, "policy"):
        inner = inner.policy
    action_head = inner.model.action_head
    action_head.__class__ = _SteerableActionHead
    action_head._injected_noise = None
    print("[run_gr00t_server] SteerableActionHead applied — noise injection enabled")
    return action_head


def _make_noise_aware_handler(policy, action_head: _SteerableActionHead):
    """Return a get_action handler that accepts an optional *noise* field."""

    def _get_action(
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        noise: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if noise is not None:
            noise_t = torch.from_numpy(np.asarray(noise)).float()
            if noise_t.ndim == 2:
                noise_t = noise_t.unsqueeze(0)  # (H, D) -> (1, H, D)
            action_head.set_initial_noise(noise_t)
        return policy.get_action(observation=observation, options=options)

    return _get_action


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    # Optionally enable noise injection for DSRL online RL.
    if config.steerable:
        action_head = _apply_steerable_head(policy)
        noise_handler = _make_noise_aware_handler(policy, action_head)
        server.register_endpoint("get_action", noise_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
