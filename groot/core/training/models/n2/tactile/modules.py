# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tactile/deform modules on GrootN2: registration and the frozen encoders.

``attach`` registers every tactile submodule on the model, in the order and under the
names tactile checkpoints were trained with: parameter names are the state_dict keys,
registration order is the ZeRO-2 optimizer layout, and construction order fixes how much
init RNG each module consumes (and so the fresh init of every module registered after it).
"""

from dataclasses import dataclass, replace
import logging
import os

import numpy as np
import torch
from torch import nn

from groot.core.training.configs.model.n2_tactile import (
    DEFAULT_FUTURE_PAD_TOLERANCE,
    DEFAULT_PREDICT_FUTURE,
    DEFAULT_TACTILE_LOSS_SCALE,
    DEFAULT_TACTILE_LOSS_WARMUP_STEPS,
    DEFORM_FEATURE_DIM,
    DEFORM_INPUT_HW,
    DEFORM_N_MAPS,
    TACTILE_F6_DIM,
    TACTILE_INPUT_WINDOW,
    TACTILE_LATENT_DIM,
    TACTILE_N_FINGERS,
    TACTILE_N_HANDS,
    TACTILE_N_TOKENS,
)
from groot.core.training.models.n2.components.action_modules import ActionRoPE1D
from groot.core.training.models.n2.tactile import guards

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TactileSpec:
    """What a tactile GrootN2 carries, resolved once from its config at construction."""

    has_tactile: bool
    predict_future: bool
    has_deform: bool
    tactile_loss_scale: float
    tactile_loss_warmup_steps: int
    future_pad_tolerance: int
    # Cleared only after checkpoint loading supplies every required frozen-encoder key.
    tactile_vae_needs_checkpoint: bool = False
    deform_encoder_needs_checkpoint: bool = False
    pending_encoder_keys: frozenset[str] = frozenset()
    # Resolved here, not per step: the training forward may be compiled, and Dynamo cannot
    # introspect the HF config.
    config_fields: tuple[str, ...] = ()
    training_input_keys: tuple[str, ...] = ()


def attach(model: nn.Module, config, dit_config) -> TactileSpec | None:
    """Register the tactile/deform modules the config asks for; None for other models.

    Called by GrootN2.__init__ right after the frozen VAE is built.
    """
    fields = guards.tactile_config_fields(config)
    if not fields:
        return None
    guards.require_exact_grootn2(model, config)
    if getattr(config, "token_merge_kernel", None) is not None:
        raise guards.TactileUnsupportedError("tactile_cfg/deform_cfg do not support token merging")
    hidden_size = dit_config.num_attention_heads * dit_config.attention_head_dim
    head_dim = dit_config.attention_head_dim

    tactile_cfg = config.tactile_cfg
    predict_future = False
    vae_needs_checkpoint = False
    if tactile_cfg is not None:
        model.tactile_vae, vae_needs_checkpoint = _load_tactile_vae(
            tactile_cfg["tactile_vae_ckpt_path"]
        )
        model.tactile_vae.requires_grad_(False)
        model.tactile_norm = nn.LayerNorm(TACTILE_LATENT_DIM)
        model.tactile_token_embed = nn.Linear(TACTILE_LATENT_DIM, hidden_size)
        model.tactile_pos_emb = nn.Parameter(torch.zeros(1, TACTILE_N_TOKENS, hidden_size))
        predict_future = bool(tactile_cfg.get("predict_future", DEFAULT_PREDICT_FUTURE))
        # Predicting future force widens the RoPE to 20 positions: present (cond) tokens at
        # 0..9, future (denoise) tokens at 10..19. The first 10 are unchanged by the widening.
        rope_len = 2 * TACTILE_N_TOKENS if predict_future else TACTILE_N_TOKENS
        model.tactile_rope = ActionRoPE1D(head_dim, max_len=rope_len)
        if predict_future:
            # Velocity decoder: hidden -> tactile latent velocity.
            model.tactile_decoder_head = nn.Linear(hidden_size, TACTILE_LATENT_DIM)
            # Linear-warmup multiplier on the tactile loss, set per step by
            # TactileLossWarmupCallback. A schedule value, not a weight: not persisted.
            model.register_buffer("tactile_loss_warmup_factor", torch.tensor(1.0), persistent=False)

    deform_cfg = config.deform_cfg
    deform_needs_checkpoint = False
    if deform_cfg is not None:
        model.deform_encoder, deform_needs_checkpoint = _load_hb1_deform_encoder(
            deform_cfg["deform_encoder_ckpt_path"]
        )
        model.deform_encoder.requires_grad_(False)
        model.deform_norm = nn.LayerNorm(DEFORM_FEATURE_DIM)
        model.deform_token_embed = nn.Linear(DEFORM_FEATURE_DIM, hidden_size)
        model.deform_pos_emb = nn.Parameter(torch.zeros(1, DEFORM_N_MAPS, hidden_size))
        model.deform_rope = ActionRoPE1D(head_dim, max_len=DEFORM_N_MAPS)

    if predict_future:
        # Future-force flow matching samples noise levels like the action head.
        from groot.core.training.models.n2.components.scheduler import (
            BetaFlowMatchEulerScheduler,
            BetaSchedulerConfig,
        )

        model.tactile_scheduler = BetaFlowMatchEulerScheduler(
            BetaSchedulerConfig(beta_alpha=1.5, beta_beta=1.0, noise_s=0.999)
        )
        model.tactile_scheduler.setup_training()

    return TactileSpec(
        has_tactile=tactile_cfg is not None,
        predict_future=predict_future,
        has_deform=deform_cfg is not None,
        tactile_loss_scale=(tactile_cfg or {}).get(
            "tactile_loss_scale", DEFAULT_TACTILE_LOSS_SCALE
        ),
        tactile_loss_warmup_steps=(tactile_cfg or {}).get(
            "tactile_loss_warmup_steps", DEFAULT_TACTILE_LOSS_WARMUP_STEPS
        ),
        future_pad_tolerance=(tactile_cfg or {}).get(
            "future_pad_tolerance", DEFAULT_FUTURE_PAD_TOLERANCE
        ),
        tactile_vae_needs_checkpoint=vae_needs_checkpoint,
        deform_encoder_needs_checkpoint=deform_needs_checkpoint,
        pending_encoder_keys=frozenset(
            f"{name}.{key}"
            for name, pending in (
                ("tactile_vae", vae_needs_checkpoint),
                ("deform_encoder", deform_needs_checkpoint),
            )
            if pending
            for key in getattr(model, name).state_dict()
        ),
        config_fields=tuple(sorted(fields)),
        training_input_keys=guards.required_input_keys(config, training=True),
    )


def _load_tactile_vae(ckpt_path: str) -> tuple[nn.Module, bool]:
    """The frozen Tactile-VAE encoder, and whether its weights must come from the GrootN2 ckpt.

    Tactile checkpoints store the encoder weights and F6 stats (``tactile_vae.*``), so a
    model built to load one (serving off-cluster, resuming, warm-starting from a tactile
    checkpoint) does not need the external file: the architecture is built with NaN weights
    that the checkpoint then overwrites. Anything else must have the file.
    """
    from groot.core.training.models.n2.tactile.encoders import (
        FingerWiseTactileEncoder,
        TacF6Stats,
        TactileVAEConfig,
        TactileVAEEncoderInference,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        vae = TactileVAEEncoderInference.from_checkpoint(ckpt_path, device="cpu")
        if not vae.shared_hand_stats:
            raise ValueError(
                "GrootN2 requires shared 30D Tactile-VAE normalization stats; "
                "60D hand-specific stats are not supported."
            )
        return vae, False
    logger.warning(
        "Tactile-VAE checkpoint not found at %r; building the encoder with NaN weights, "
        "which the GrootN2 checkpoint must restore completely.",
        ckpt_path,
    )
    cfg = TactileVAEConfig(
        input_window=TACTILE_INPUT_WINDOW,
        per_finger_dim=TACTILE_F6_DIM,
        n_fingers=TACTILE_N_FINGERS,
        hidden_channels=128,
        bottleneck_channels=256,
        bottleneck_T=5,
        latent_dim=TACTILE_LATENT_DIM,
        latent_scope="finger",
        n_strided_blocks=2,
        temporal_pool="attn",
        use_finger_embed=True,
    )
    encoder = FingerWiseTactileEncoder(
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
    n = cfg.n_fingers * cfg.per_finger_dim  # shared (hand-agnostic) stats
    stats = TacF6Stats(
        tacf6_min=np.zeros(n, dtype=np.float32),
        tacf6_max=np.ones(n, dtype=np.float32),
        tacf6_mask=np.ones(n, dtype=bool),
        stats_source="deploy_placeholder",
    )
    vae = TactileVAEEncoderInference(encoder=encoder, cfg=cfg, stats=stats)
    with torch.no_grad():
        for tensor in [*vae.parameters(), vae.tacf6_min, vae.tacf6_max]:
            tensor.fill_(float("nan"))
    return vae, True


def _load_hb1_deform_encoder(ckpt_path: str) -> tuple[nn.Module, bool]:
    """The frozen HB1 deform encoder (``.deform_encoder`` of the full SharpaWave autoencoder).

    The full autoencoder is built because the released file is its state_dict and because
    building its decoder consumes init RNG that later modules' fresh init depends on.
    Without the file, the weights must come from the GrootN2 checkpoint (``deform_encoder.*``).
    """
    from groot.core.training.models.n2.tactile.providers import build_hb1_autoencoder

    model = build_hb1_autoencoder()
    needs_checkpoint = not (ckpt_path and os.path.exists(ckpt_path))
    if not needs_checkpoint:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # The released file is a full training checkpoint (teacher, detection/norm heads);
        # only the encoder is used, and it must load completely.
        prefix = "deform_encoder."
        model.deform_encoder.load_state_dict(
            {
                k.removeprefix(prefix): v
                for k, v in checkpoint.get("state_dict", checkpoint).items()
                if k.startswith(prefix)
            }
        )
    else:
        logger.warning(
            "Deform encoder checkpoint not found at %r; building it with NaN weights, which "
            "the GrootN2 checkpoint must restore completely.",
            ckpt_path,
        )
        with torch.no_grad():
            for param in model.deform_encoder.parameters():
                param.fill_(float("nan"))
    encoder = model.deform_encoder
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, needs_checkpoint


def record_encoder_load(model: nn.Module, loaded_keys) -> None:
    """Track explicit state-dict loads, including HF Trainer's shard-by-shard resume."""
    spec = model.tactile_spec
    if spec is None or not spec.pending_encoder_keys:
        return
    pending = spec.pending_encoder_keys.difference(loaded_keys)
    model.tactile_spec = replace(
        spec,
        pending_encoder_keys=pending,
        tactile_vae_needs_checkpoint=any(k.startswith("tactile_vae.") for k in pending),
        deform_encoder_needs_checkpoint=any(k.startswith("deform_encoder.") for k in pending),
    )


def complete_encoder_load(model: nn.Module, missing_keys, *, from_pretrained=False) -> None:
    """Validate frozen encoder restoration after all parent-checkpoint shards have loaded.

    HF can materialize missing parameters with finite, uninitialized storage, so NaN
    sentinels cannot establish that weights loaded. Under HF meta construction, external
    files loaded in __init__ can also be lost; reload only their missing keys here.
    """
    spec = model.tactile_spec
    if spec is None:
        return
    updated = {}
    for name, config_name, path_key, loader in (
        ("tactile_vae", "tactile_cfg", "tactile_vae_ckpt_path", _load_tactile_vae),
        ("deform_encoder", "deform_cfg", "deform_encoder_ckpt_path", _load_hb1_deform_encoder),
    ):
        encoder = getattr(model, name, None)
        if encoder is None:
            continue
        flag = f"{name}_needs_checkpoint"
        missing = sorted(k for k in missing_keys if k.startswith(f"{name}."))
        if missing:
            if getattr(spec, flag):
                raise RuntimeError(f"Frozen encoder weights missing from checkpoint: {missing}")
            if from_pretrained:
                # These tensors must come from the external file, not HF's uninitialized
                # storage. Preserve the caller's RNG and the parent checkpoint's other keys.
                path = getattr(model.config, config_name)[path_key]
                with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                    external, needs_checkpoint = loader(path)
                if needs_checkpoint:
                    raise RuntimeError(f"Frozen encoder weights missing from checkpoint: {missing}")
                state = external.state_dict()
                encoder.load_state_dict(
                    {
                        k.removeprefix(f"{name}."): state[k.removeprefix(f"{name}.")]
                        for k in missing
                    },
                    strict=False,
                )
        # HF may replace Parameter objects while materializing a meta model.
        encoder.requires_grad_(False)
        updated[flag] = False
    model.tactile_spec = replace(spec, pending_encoder_keys=frozenset(), **updated)


@torch.compiler.disable
def pin_tactile_vae(model: nn.Module, device: torch.device) -> None:
    """Pin the frozen Tactile-VAE to fp32 on ``device``, outside any compiled graph.

    DeepSpeed bf16 casts the whole model, this frozen encoder included. If ``encode_tactile``
    casts it back inside a compiled training forward, Dynamo recompiles encode_tactile one step
    later on the parameter-dtype guard. Under reduce-overhead, CUDA-graph trees then record that
    late graph mid-path by restoring a checkpointed allocator state, which asserts with
    expandable segments ("Expected curr_block->next == nullptr"). Pinning here first leaves
    encode_tactile nothing to convert, so it traces once. Same cast, one call earlier.
    """
    param = next(model.tactile_vae.parameters())
    if param.device != device or param.dtype != torch.float32:
        model.tactile_vae.to(device=device, dtype=torch.float32)


def encode_tactile(model: nn.Module, tactile_raw: torch.Tensor) -> torch.Tensor:
    """Encode a present/future F6 window into clean tactile latent tokens.

    Args:
        model: A GrootN2 with ``tactile_cfg``.
        tactile_raw: (B, 2 hands, 20 frames, 5 fingers, 6) raw wrench, hand-major.

    Returns:
        (B, 10, 256) posterior-mean latents in bf16 (2 hands x 5 fingers).
    """
    device = tactile_raw.device
    # Run the frozen encoder in fp32. This restores the compute dtype, not precision
    # already lost when DeepSpeed cast its weights/stat buffers or the raw inputs to bf16.
    param = next(model.tactile_vae.parameters())
    if param.device != device or param.dtype != torch.float32:
        model.tactile_vae.to(device=device, dtype=torch.float32)
    if model.tactile_spec.tactile_vae_needs_checkpoint:
        raise RuntimeError("Tactile-VAE weights were not loaded from the checkpoint")
    batch = tactile_raw.shape[0]
    x = tactile_raw.reshape(
        batch * TACTILE_N_HANDS, TACTILE_INPUT_WINDOW, TACTILE_N_FINGERS, TACTILE_F6_DIM
    )
    with torch.amp.autocast(device_type="cuda", enabled=False):
        mu = model.tactile_vae.encode_raw(x.float())  # (B*2, 5, 256)
    mu = mu.reshape(batch, TACTILE_N_TOKENS, TACTILE_LATENT_DIM)
    return mu.to(torch.bfloat16)


@torch.compiler.disable
@torch.no_grad()
def encode_deform(model: nn.Module, deform_raw: torch.Tensor) -> torch.Tensor:
    """Encode the 10 per-fingertip deform maps into clean deform conditioning tokens.

    Raw [0, 255] grayscale is fed without /255 (as the encoder was trained). Compiler-disabled:
    the frozen timm ConvNeXt V2 backbone trips Dynamo and runs once per step anyway.

    Args:
        model: A GrootN2 with ``deform_cfg``.
        deform_raw: (B, 10, 240, 240) float32 deform maps, hand-major.

    Returns:
        (B, 10, 512) per-fingertip features in bf16.
    """
    device = deform_raw.device
    param = next(model.deform_encoder.parameters())
    if param.device != device or param.dtype != torch.float32:
        model.deform_encoder.to(device=device, dtype=torch.float32)
    if model.tactile_spec.deform_encoder_needs_checkpoint:
        raise RuntimeError("Deform encoder weights were not loaded from the checkpoint")
    batch = deform_raw.shape[0]
    x = deform_raw.reshape(batch * DEFORM_N_MAPS, 1, DEFORM_INPUT_HW, DEFORM_INPUT_HW)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        feat = model.deform_encoder(deform_input=x.float())  # (B*10, 512, 1, 1)
    feat = nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
    feat = feat.reshape(batch, DEFORM_N_MAPS, DEFORM_FEATURE_DIM)
    return feat.to(torch.bfloat16)
