# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Frozen Tactile-VAE force encoder used by N2 tactile conditioning (inference only).

Runtime subset of the Tactile-VAE project (config, F6 normalization stats, the
finger-/hand-wise encoders and the raw-force inference wrapper), flattened into one
importable module so no ``sys.modules`` shim is needed. The decoder, the full VAE,
the datasets and the stats builders are training tooling and are not here.

Attribute paths are load-bearing: GrootN2 checkpoints store this module under the
``tactile_vae.`` prefix (``encoder.stem.0.weight`` ... ``tacf6_min/max/mask``).
Edits relative to the upstream sources are lint-only (same ops, same order, same
init RNG consumption), which ``tests/.../n2/tactile/test_encoders.py`` pins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_F6_DIM = 60
_HAND_DIM = 30


@dataclass
class TacF6Stats:
    """q01/q99 min-max F6 stats: shared 30D hand values or full bimanual 60D values."""

    tacf6_min: np.ndarray
    tacf6_max: np.ndarray
    tacf6_mask: np.ndarray
    stats_source: str = "manifest"

    @classmethod
    def _checked(
        cls,
        tacf6_min: np.ndarray,
        tacf6_max: np.ndarray,
        tacf6_mask: np.ndarray,
        stats_source: str,
    ) -> TacF6Stats:
        if (
            tacf6_min.shape[0] not in (_HAND_DIM, _F6_DIM)
            or tacf6_max.shape[0] != tacf6_min.shape[0]
        ):
            raise ValueError(
                f"Expected F6 stats dim {_HAND_DIM} or {_F6_DIM}, got "
                f"{tacf6_min.shape}/{tacf6_max.shape}"
            )
        if tacf6_mask.shape[0] != tacf6_min.shape[0]:
            raise ValueError(
                f"Expected F6 stats mask dim {tacf6_min.shape[0]}, got {tacf6_mask.shape}"
            )
        return cls(
            tacf6_min=tacf6_min.astype(np.float32, copy=False),
            tacf6_max=tacf6_max.astype(np.float32, copy=False),
            tacf6_mask=tacf6_mask.astype(bool, copy=False),
            stats_source=stats_source,
        )

    @property
    def is_shared_hand(self) -> bool:
        return int(self.tacf6_min.shape[0]) == _HAND_DIM

    @classmethod
    def from_dict(cls, d: dict) -> TacF6Stats:
        return cls._checked(
            np.array(d["tacf6_min"], dtype=np.float32),
            np.array(d["tacf6_max"], dtype=np.float32),
            np.array(d["tacf6_mask"], dtype=bool),
            d.get("stats_source", "unknown"),
        )


@dataclass
class TactileVAEConfig:
    # The full training config is kept so ``from_dict`` on a checkpoint's saved
    # config keeps exactly the same fields; only the architecture fields matter here.
    source_window: int = 64
    input_window: int = 16
    subsample_stride: int = 4
    per_finger_dim: int = 6
    n_fingers: int = 5
    hidden_channels: int = 128
    bottleneck_channels: int = 256
    bottleneck_T: int = 4
    latent_dim: int = 256
    latent_scope: str = "hand"
    n_strided_blocks: int = 2
    temporal_pool: str = "attn"
    use_finger_embed: bool = True
    use_time_embed: bool = True
    beta_kl: float = 1e-3
    use_magnitude_weight: bool = True
    weight_alpha: float = 2.0
    weight_tau: float = 4.0

    @classmethod
    def from_dict(cls, d: dict) -> TactileVAEConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T]
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)


class FingerAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, C]
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class HandWiseTactileEncoder(nn.Module):
    """Encode [B, T, 5, 6] into VAE posterior parameters [B, 1, latent_dim]."""

    def __init__(
        self,
        window: int = 16,
        per_finger_dim: int = 6,
        n_fingers: int = 5,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        latent_dim: int = 256,
        n_strided_blocks: int = 2,
        temporal_pool: str = "attn",
        use_finger_embed: bool = True,
    ):
        super().__init__()
        if temporal_pool not in ("attn", "flatten_mlp"):
            raise ValueError("temporal_pool must be 'attn' or 'flatten_mlp'")
        self.window = int(window)
        self.n_fingers = int(n_fingers)
        self.latent_dim = int(latent_dim)
        self.temporal_pool = temporal_pool
        self.use_finger_embed = bool(use_finger_embed)

        self.stem = _conv_block(per_finger_dim, hidden_channels, kernel=5, stride=1)
        self.finger_embed = (
            nn.Embedding(n_fingers, hidden_channels) if self.use_finger_embed else None
        )

        blocks: list[nn.Module] = []
        cur_T = self.window
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(_conv_block(cur_ch, out_ch, kernel=5, stride=stride))
            cur_ch = out_ch
            cur_T = cur_T // stride if stride > 1 else cur_T
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_T = cur_T
        self.proj = nn.Conv1d(cur_ch, bottleneck_channels, kernel_size=3, padding=1)

        if temporal_pool == "attn":
            self.temporal_pooler = TemporalAttentionPool(bottleneck_channels)
        else:
            self.temporal_pooler = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(bottleneck_channels * self._bottleneck_T, bottleneck_channels),
                nn.LayerNorm(bottleneck_channels),
                nn.GELU(),
            )

        self.finger_pooler = FingerAttentionPool(bottleneck_channels)
        self.mu_head = nn.Linear(bottleneck_channels, latent_dim)
        self.logvar_head = nn.Linear(bottleneck_channels, latent_dim)

    @property
    def bottleneck_T(self) -> int:
        return self._bottleneck_T

    def forward(self, f6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """f6: [B, T=16, 5, 6] -> (mu, logvar): [B, 1, latent_dim]."""
        B, T, F, D = f6.shape
        if self.window != T:
            raise ValueError(f"Encoder built for window={self.window}, got T={T}")
        if self.n_fingers != F:
            raise ValueError(f"Expected {self.n_fingers} fingers, got {F}")

        x = f6.permute(0, 2, 1, 3).contiguous()  # [B, 5, T, 6]
        x = x.reshape(B * F, T, D).transpose(1, 2)  # [B*5, 6, T]
        x = self.stem(x)  # [B*5, hidden, T]

        if self.finger_embed is not None:
            ids = torch.arange(F, device=f6.device).repeat(B)
            x = x + self.finger_embed(ids).unsqueeze(-1)

        x = self.strided(x)
        x = self.proj(x)  # [B*5, bottleneck, T_bn]
        finger_feat = self.temporal_pooler(x)  # [B*5, bottleneck]
        finger_feat = finger_feat.reshape(B, F, -1)  # [B, 5, bottleneck]
        hand_feat = self.finger_pooler(finger_feat)  # [B, bottleneck]
        mu = self.mu_head(hand_feat).unsqueeze(1)  # [B, 1, latent_dim]
        logvar = self.logvar_head(hand_feat).unsqueeze(1)
        return mu, logvar


class FingerWiseTactileEncoder(nn.Module):
    """Encode [B, T, 5, 6] into per-finger posterior parameters [B, 5, latent_dim]."""

    def __init__(
        self,
        window: int = 16,
        per_finger_dim: int = 6,
        n_fingers: int = 5,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        latent_dim: int = 256,
        n_strided_blocks: int = 2,
        temporal_pool: str = "attn",
        use_finger_embed: bool = True,
    ):
        super().__init__()
        if temporal_pool not in ("attn", "flatten_mlp"):
            raise ValueError("temporal_pool must be 'attn' or 'flatten_mlp'")
        self.window = int(window)
        self.n_fingers = int(n_fingers)
        self.latent_dim = int(latent_dim)
        self.temporal_pool = temporal_pool
        self.use_finger_embed = bool(use_finger_embed)

        self.stem = _conv_block(per_finger_dim, hidden_channels, kernel=5, stride=1)
        self.finger_embed = (
            nn.Embedding(n_fingers, hidden_channels) if self.use_finger_embed else None
        )

        blocks: list[nn.Module] = []
        cur_T = self.window
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(_conv_block(cur_ch, out_ch, kernel=5, stride=stride))
            cur_ch = out_ch
            cur_T = cur_T // stride if stride > 1 else cur_T
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_T = cur_T
        self.proj = nn.Conv1d(cur_ch, bottleneck_channels, kernel_size=3, padding=1)

        if temporal_pool == "attn":
            self.temporal_pooler = TemporalAttentionPool(bottleneck_channels)
        else:
            self.temporal_pooler = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(bottleneck_channels * self._bottleneck_T, bottleneck_channels),
                nn.LayerNorm(bottleneck_channels),
                nn.GELU(),
            )

        self.mu_head = nn.Linear(bottleneck_channels, latent_dim)
        self.logvar_head = nn.Linear(bottleneck_channels, latent_dim)

    @property
    def bottleneck_T(self) -> int:
        return self._bottleneck_T

    def forward(self, f6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """f6: [B, T=16, 5, 6] -> (mu, logvar): [B, 5, latent_dim]."""
        B, T, F, D = f6.shape
        if self.window != T:
            raise ValueError(f"Encoder built for window={self.window}, got T={T}")
        if self.n_fingers != F:
            raise ValueError(f"Expected {self.n_fingers} fingers, got {F}")

        x = f6.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B * F, T, D).transpose(1, 2)
        x = self.stem(x)

        if self.finger_embed is not None:
            ids = torch.arange(F, device=f6.device).repeat(B)
            x = x + self.finger_embed(ids).unsqueeze(-1)

        x = self.strided(x)
        x = self.proj(x)
        finger_feat = self.temporal_pooler(x)
        finger_feat = finger_feat.reshape(B, F, -1)
        return self.mu_head(finger_feat), self.logvar_head(finger_feat)


class TactileVAEEncoderInference(nn.Module):
    """Frozen encoder wrapper for raw one-hand tactile force inference.

    Public input is raw tactile force with shape [B, T, 5, 6]. The wrapper uses
    q01/q99 stats saved in the checkpoint, then returns deterministic VAE
    posterior means. New checkpoints may use shared 30D hand stats and do not
    require a hand id; old 60D checkpoints still require hand ids for raw input.
    """

    def __init__(
        self,
        encoder: HandWiseTactileEncoder | FingerWiseTactileEncoder,
        cfg: TactileVAEConfig,
        stats: TacF6Stats,
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.stats_source = stats.stats_source
        self.shared_hand_stats = stats.is_shared_hand

        if self.shared_hand_stats:
            stats_shape = (cfg.n_fingers, cfg.per_finger_dim)
        else:
            stats_shape = (2, cfg.n_fingers, cfg.per_finger_dim)
        self.register_buffer(
            "tacf6_min", torch.from_numpy(stats.tacf6_min.reshape(stats_shape)).float()
        )
        self.register_buffer(
            "tacf6_max", torch.from_numpy(stats.tacf6_max.reshape(stats_shape)).float()
        )
        self.register_buffer(
            "tacf6_mask", torch.from_numpy(stats.tacf6_mask.reshape(stats_shape)).bool()
        )

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        device: str | torch.device | None = None,
    ) -> TactileVAEEncoderInference:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = TactileVAEConfig.from_dict(state["config"])
        stats = TacF6Stats.from_dict(state["stats"])
        encoder_cls = (
            FingerWiseTactileEncoder if cfg.latent_scope == "finger" else HandWiseTactileEncoder
        )
        encoder = encoder_cls(
            window=cfg.input_window,
            per_finger_dim=cfg.per_finger_dim,
            n_fingers=cfg.n_fingers,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            latent_dim=cfg.latent_dim,
            n_strided_blocks=cfg.n_strided_blocks,
            temporal_pool=cfg.temporal_pool,
            use_finger_embed=cfg.use_finger_embed,
        )

        if "encoder_state" in state:
            encoder_state = state["encoder_state"]
        elif "model_state" in state:
            prefix = "encoder."
            encoder_state = {
                k[len(prefix) :]: v for k, v in state["model_state"].items() if k.startswith(prefix)
            }
        else:
            raise KeyError("Checkpoint must contain 'encoder_state' or 'model_state'.")
        encoder.load_state_dict(encoder_state)

        module = cls(encoder=encoder, cfg=cfg, stats=stats)
        # Under HF `from_pretrained`'s empty-weights/meta init (used when this
        # encoder is a submodule of a larger model loaded from a sharded
        # checkpoint), the freshly-built params/buffers live on the meta device
        # and `.to()` raises "Cannot copy out of meta tensor". In that case the
        # parent checkpoint already carries these frozen weights (tactile_vae.*),
        # so leave the module on meta and let `from_pretrained` materialize it.
        if any(p.is_meta for p in module.parameters()) or any(b.is_meta for b in module.buffers()):
            return module
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return module.to(device)

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def latent_shape(self) -> tuple[int, ...]:
        if self.cfg.latent_scope == "finger":
            return (self.cfg.n_fingers, self.cfg.latent_dim)
        return (1, self.cfg.latent_dim)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.cfg.input_window, self.cfg.n_fingers, self.cfg.per_finger_dim)

    @torch.no_grad()
    def encode_normalized(self, f6_normed: torch.Tensor | np.ndarray) -> torch.Tensor:
        f6_normed = self._as_input_tensor(f6_normed)
        self._validate_f6(f6_normed)
        mu, _ = self.encoder(f6_normed)
        return mu.detach()

    @torch.no_grad()
    def encode_raw(
        self,
        raw_f6: torch.Tensor | np.ndarray,
        hand: int | list[int] | np.ndarray | torch.Tensor | None = None,
        baseline_f6: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        raw = self._as_input_tensor(raw_f6)
        self._validate_f6(raw)
        raw = self._subtract_baseline(raw, baseline_f6)
        hand_idx = (
            None
            if self.shared_hand_stats
            else self._as_hand_tensor(hand, batch_size=raw.shape[0], device=raw.device)
        )
        f6_normed = self._normalize_raw(raw, hand_idx)
        return self.encode_normalized(f6_normed)

    def forward(
        self,
        raw_f6: torch.Tensor | np.ndarray,
        hand: int | list[int] | np.ndarray | torch.Tensor | None = None,
        baseline_f6: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        return self.encode_raw(raw_f6, hand, baseline_f6=baseline_f6)

    def _as_input_tensor(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        return x.to(device=self.tacf6_min.device, dtype=torch.float32)

    def _validate_f6(self, x: torch.Tensor) -> None:
        expected = (self.cfg.input_window, self.cfg.n_fingers, self.cfg.per_finger_dim)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected input shape [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(x.shape)}."
            )

    def _subtract_baseline(
        self,
        raw: torch.Tensor,
        baseline_f6: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        if baseline_f6 is None:
            return raw
        baseline = torch.as_tensor(
            baseline_f6,
            dtype=raw.dtype,
            device=raw.device,
        )
        if baseline.ndim == 1 and baseline.numel() == self.cfg.n_fingers * self.cfg.per_finger_dim:
            baseline = baseline.reshape(self.cfg.n_fingers, self.cfg.per_finger_dim)
        if baseline.ndim == 2 and tuple(baseline.shape) == (
            self.cfg.n_fingers,
            self.cfg.per_finger_dim,
        ):
            baseline = baseline.unsqueeze(0).unsqueeze(0)
        elif baseline.ndim == 3 and tuple(baseline.shape[1:]) == (
            self.cfg.n_fingers,
            self.cfg.per_finger_dim,
        ):
            baseline = baseline.unsqueeze(1)
        elif baseline.ndim == 4 and tuple(baseline.shape[1:]) == (
            1,
            self.cfg.n_fingers,
            self.cfg.per_finger_dim,
        ):
            pass
        else:
            raise ValueError(
                "baseline_f6 must have shape [5,6], [30], [B,5,6], or [B,1,5,6]; "
                f"got {tuple(baseline.shape)}."
            )
        if baseline.shape[0] not in (1, raw.shape[0]):
            raise ValueError(
                f"baseline_f6 batch dimension must be 1 or {raw.shape[0]}, "
                f"got {baseline.shape[0]}."
            )
        return raw - baseline

    @staticmethod
    def _as_hand_tensor(
        hand: int | list[int] | np.ndarray | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if hand is None:
            raise ValueError("hand is required for raw inference with 60D hand-specific stats.")
        if isinstance(hand, int):
            hand_idx = torch.full((batch_size,), hand, dtype=torch.long, device=device)
        else:
            hand_idx = torch.as_tensor(hand, dtype=torch.long, device=device)
            if hand_idx.ndim == 0:
                hand_idx = hand_idx.repeat(batch_size)
        if hand_idx.shape != (batch_size,):
            raise ValueError(
                f"hand must be an int or a length-{batch_size} tensor/list, "
                f"got shape {tuple(hand_idx.shape)}."
            )
        if not torch.all((hand_idx == 0) | (hand_idx == 1)):
            raise ValueError("hand values must be 0 or 1.")
        return hand_idx

    def _normalize_raw(self, raw: torch.Tensor, hand_idx: torch.Tensor | None) -> torch.Tensor:
        if self.shared_hand_stats:
            vmin = self.tacf6_min.unsqueeze(0).unsqueeze(0)
            vmax = self.tacf6_max.unsqueeze(0).unsqueeze(0)
            mask = self.tacf6_mask.unsqueeze(0).unsqueeze(0)
        else:
            if hand_idx is None:
                raise ValueError("hand is required for raw inference with 60D hand-specific stats.")
            vmin = self.tacf6_min[hand_idx].unsqueeze(1)
            vmax = self.tacf6_max[hand_idx].unsqueeze(1)
            mask = self.tacf6_mask[hand_idx].unsqueeze(1)
        normed = 2.0 * (raw - vmin) / ((vmax - vmin) + 1e-8) - 1.0
        normed = torch.clamp(normed, -1.0, 1.0)
        return torch.where(mask, normed, raw)
