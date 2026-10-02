"""Numeric SH5 tactile encoders and a lazy import boundary for research adapters.

Contract: normalized [..., 2 hands, 5 sensors, 9 taxels] -> [..., embed_dim].
Taxel order is the recorded pressure_names order; no physical 3x3 geometry is
assumed. These small trainable baselines are not pretrained SoTA checkpoints.
External wrappers (e.g. for Sparsh-Skin) must implement this numeric contract;
image-based Sparsh/DIGIT models cannot consume these pressures directly.
"""

from importlib import import_module
from math import prod

import torch
from torch import nn


class FingerMLPEncoder(nn.Module):
    """Shared per-finger features, then an ordered two-hand projection."""

    def __init__(self, embed_dim: int, input_shape=(2, 5, 9)):
        super().__init__()
        self.finger = nn.Sequential(nn.Linear(input_shape[-1], 32), nn.GELU(), nn.Linear(32, 16))
        self.projection = nn.Sequential(
            nn.Flatten(-3),
            nn.Linear(prod(input_shape[:-1]) * 16, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pressures):
        return self.projection(self.finger(pressures))


class FingerTransformerEncoder(nn.Module):
    """Ten ordered finger tokens with learned hand/finger identities."""

    def __init__(self, embed_dim: int, input_shape=(2, 5, 9)):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.num_fingers = prod(input_shape[:-1])
        self.patch = nn.Linear(input_shape[-1], 64)
        self.finger_id = nn.Embedding(self.num_fingers, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.projection = nn.Sequential(nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, pressures):
        leading = pressures.shape[:-3]
        tokens = self.patch(pressures.reshape(-1, self.num_fingers, self.input_shape[-1]))
        tokens = tokens + self.finger_id(torch.arange(self.num_fingers, device=pressures.device))
        features = self.transformer(tokens).mean(dim=1)
        return self.projection(features).reshape(*leading, -1)


def build_tactile_encoder(name: str, embed_dim: int, input_shape=(2, 5, 9)) -> nn.Module:
    """Load only the selected encoder; external factories take embed_dim=... and input_shape=... .

    An external name is an installed Python 'package.module:factory'. Its
    factory must construct the architecture without downloading weights; all
    weights are subsequently loaded from the GR00T checkpoint. Only use trusted
    module paths, just as with a Python modality config.
    """
    if embed_dim <= 0 or len(input_shape) != 3 or any(d <= 0 for d in input_shape):
        raise ValueError("Tactile latent width and all three input dimensions must be positive")
    builtins = {"finger_mlp": FingerMLPEncoder, "finger_transformer": FingerTransformerEncoder}
    if name in builtins:
        return builtins[name](embed_dim, input_shape)
    if ":" not in name:
        raise ValueError(
            f"Unknown tactile encoder {name!r}; choose {list(builtins)} or module:factory"
        )
    module, factory = name.rsplit(":", 1)
    encoder = getattr(import_module(module), factory)(
        embed_dim=embed_dim, input_shape=tuple(input_shape)
    )
    if not isinstance(encoder, nn.Module):
        raise TypeError("A tactile encoder factory must return torch.nn.Module")
    return encoder
