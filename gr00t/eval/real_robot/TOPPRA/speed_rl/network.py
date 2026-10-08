from __future__ import annotations

from collections.abc import Mapping
import math
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .config import RainbowConfig


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer used by the SpeedTuning Rainbow head."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int, *, device: torch.device) -> Tensor:
        values = torch.randn(size, device=device)
        return values.sign() * values.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scaled_noise(self.in_features, device=self.weight_epsilon.device)
        epsilon_out = self._scaled_noise(self.out_features, device=self.weight_epsilon.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, values: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return functional.linear(values, weight, bias)


class DuelingC51Network(nn.Module):
    """SpeedTuning-compatible normalized dueling categorical Q-network."""

    def __init__(self, config: RainbowConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.advantage_hidden = NoisyLinear(
            config.hidden_dim,
            config.hidden_dim,
            config.noisy_std_init,
        )
        self.advantage = NoisyLinear(
            config.hidden_dim,
            config.action_count * config.atom_count,
            config.noisy_std_init,
        )
        self.value_hidden = NoisyLinear(
            config.hidden_dim,
            config.hidden_dim,
            config.noisy_std_init,
        )
        self.value = NoisyLinear(config.hidden_dim, config.atom_count, config.noisy_std_init)
        self.register_buffer(
            "support",
            torch.linspace(config.support_min, config.support_max, config.atom_count),
        )
        self.register_buffer("states_mean", torch.zeros(config.feature_dim))
        self.register_buffer("states_std", torch.ones(config.feature_dim))

    def update_norm_stats(self, mean: NDArray[Any], std: NDArray[Any]) -> None:
        mean_tensor = torch.as_tensor(
            mean, dtype=self.states_mean.dtype, device=self.states_mean.device
        )
        std_tensor = torch.as_tensor(
            std, dtype=self.states_std.dtype, device=self.states_std.device
        )
        if mean_tensor.shape != self.states_mean.shape or std_tensor.shape != self.states_std.shape:
            raise ValueError("normalization statistics do not match the feature dimension")
        if torch.any(~torch.isfinite(mean_tensor)) or torch.any(~torch.isfinite(std_tensor)):
            raise ValueError("normalization statistics must be finite")
        self.states_mean.copy_(mean_tensor)
        self.states_std.copy_(std_tensor.clamp_min(1e-6))

    def reset_noise(self) -> None:
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features must have shape (batch, {self.config.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        normalized = (features - self.states_mean) / self.states_std.clamp_min(1e-6)
        hidden = self.backbone(normalized)
        value_hidden = functional.relu(self.value_hidden(hidden))
        advantage_hidden = functional.relu(self.advantage_hidden(hidden))
        value = self.value(value_hidden).view(-1, 1, self.config.atom_count)
        advantage = self.advantage(advantage_hidden).view(
            -1,
            self.config.action_count,
            self.config.atom_count,
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def probabilities(self, features: Tensor) -> Tensor:
        probabilities = torch.softmax(self(features), dim=-1).clamp_min(1e-6)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    def q_values(self, features: Tensor) -> Tensor:
        return torch.sum(self.probabilities(features) * self.support, dim=-1)


def validate_action_mask(
    action_mask: NDArray[Any] | Tensor,
    action_count: int,
) -> BoolArray:
    if isinstance(action_mask, Tensor):
        mask = action_mask.detach().cpu().numpy()
    else:
        mask = np.asarray(action_mask)
    if mask.shape != (action_count,):
        raise ValueError(f"action mask must have shape ({action_count},), got {mask.shape}")
    if mask.dtype != np.bool_ and not np.all(np.isin(mask, (0, 1))):
        raise ValueError("action mask must contain only booleans")
    result = mask.astype(np.bool_, copy=True)
    if not np.any(result):
        raise ValueError("action mask must allow at least one action")
    return result


@torch.inference_mode()
def select_masked_action(
    network: DuelingC51Network,
    feature: FloatArray,
    action_mask: BoolArray,
    *,
    epsilon: float,
    rng: random.Random | None = None,
    device: torch.device | str = "cpu",
) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    mask = validate_action_mask(action_mask, network.config.action_count)
    generator = rng or random
    allowed = np.flatnonzero(mask)
    if generator.random() < epsilon:
        return int(generator.choice(allowed.tolist()))
    tensor = torch.as_tensor(feature, dtype=torch.float32, device=device).reshape(1, -1)
    q_values = network.q_values(tensor)[0]
    torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device)
    q_values = q_values.masked_fill(~torch_mask, -torch.inf)
    return int(torch.argmax(q_values).item())


def project_categorical_distribution(
    next_probabilities: Tensor,
    rewards: Tensor,
    dones: Tensor,
    discounts: Tensor,
    support: Tensor,
) -> Tensor:
    """Project a target distribution onto the fixed C51 support."""

    if next_probabilities.ndim != 2:
        raise ValueError("next_probabilities must have shape (batch, atoms)")
    batch_size, atom_count = next_probabilities.shape
    if support.shape != (atom_count,):
        raise ValueError("support length must match the probability atom count")
    for name, tensor in (
        ("rewards", rewards),
        ("dones", dones),
        ("discounts", discounts),
    ):
        if tensor.reshape(-1).shape != (batch_size,):
            raise ValueError(f"{name} must have one value per batch item")

    support_min = support[0]
    support_max = support[-1]
    delta = (support_max - support_min) / (atom_count - 1)
    rewards = rewards.reshape(-1, 1)
    dones = dones.reshape(-1, 1).to(next_probabilities.dtype)
    discounts = discounts.reshape(-1, 1)
    target_atoms = rewards + (1.0 - dones) * discounts * support.reshape(1, -1)
    target_atoms = target_atoms.clamp(float(support_min), float(support_max))
    locations = (target_atoms - support_min) / delta
    lower = locations.floor().long()
    upper = locations.ceil().long()

    projection = torch.zeros_like(next_probabilities)
    offsets = torch.arange(batch_size, device=next_probabilities.device).reshape(-1, 1) * atom_count
    flat = projection.reshape(-1)
    flat.index_add_(
        0,
        (lower + offsets).reshape(-1),
        (next_probabilities * (upper.to(locations.dtype) - locations)).reshape(-1),
    )
    flat.index_add_(
        0,
        (upper + offsets).reshape(-1),
        (next_probabilities * (locations - lower.to(locations.dtype))).reshape(-1),
    )
    equal = lower == upper
    if torch.any(equal):
        flat.index_add_(
            0,
            (lower + offsets)[equal],
            next_probabilities[equal],
        )
    return projection


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def load_cpu_state_dict(module: nn.Module, state_dict: Mapping[str, Any]) -> None:
    module.load_state_dict(
        {
            name: value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
            for name, value in state_dict.items()
        },
        strict=True,
    )
