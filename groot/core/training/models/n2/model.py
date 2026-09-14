# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""GrootN2: Video world model with Cosmos DiT.

Clean implementation matching the official Cosmos Predict 2.5 conventions:
- Flow matching (sigma ∈ [0, 1], velocity prediction)
- Conditioning frames get cond_timestep=0.1 (not shared sigma)
- Mask-based I2V inference (diffusers pattern)

Uses the split architecture:
- VideoTokenizer: patchify, unpatchify, positional embeddings
- video_rope: RoPE (sibling to video_tokenizer, not inside it)
- CosmosTransformer3DModel: pure-sequence (B,L,D) → (B,L,D) transformer
- ConcatPacker: packs modality tokens into PackedSequence for the transformer
"""

import dataclasses
import glob
import logging

from peft import LoraConfig, get_peft_model
import torch
from torch.nn.attention.flex_attention import BlockMask
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from groot.core.training.configs.model.n2 import VAE_LATENT_CHANNELS, GrootN2Config
from groot.core.training.models.n2.components.kv_cache import CondKVCache
from groot.core.training.models.n2.components.scheduler import (
    BetaFlowMatchEulerScheduler,
    BetaSchedulerConfig,
    FlowMatchUniPCScheduler,
    SchedulerConfig,
)
from groot.core.training.models.n2.components.sequence_packing import (
    MODALITY_ACTION,
    MODALITY_STATE,
    MODALITY_VIDEO,
    MODALITY_VIDEO_COND,
    AttentionMaskPattern,
    ConcatPacker,
    PackedSequence,
    _build_attention_mask,
    unpack_contiguous,
)
from groot.core.training.models.n2.components.text_encoder import CosmosTextEncoder
from groot.core.training.models.n2.components.transformer_cosmos import (
    COSMOS_2B_CONFIG,
    COSMOS_14B_CONFIG,
    CosmosTransformer3DModel,
    MultiViewRoPE,
)
from groot.core.training.models.n2.components.video_tokenizer import VideoTokenizer
from groot.core.training.models.n2.components.wan22_video_vae import WanVideoVAE38
from groot.core.training.models.n2.components.wan_video_vae import WanVideoVAE

# Key prefixes for routing checkpoint weights to sub-modules.
VIDEO_TOKENIZER_PREFIXES = (
    "patch_embed.",
    "proj_out.",
    "learnable_pos_embed.",
    "view_embed.",
)
VIDEO_ROPE_PREFIXES = ("rope.",)
ACTION_ROPE_SEQ_KEYS = frozenset(("action_rope.seq", "state_rope.seq"))
ACTION_TEMPORAL_POS_EMB_KEY = "action_encoder.temporal_pos_emb"

VAE_TEMPORAL_FACTOR = 4  # Wan VAE temporal compression ratio


def apply_state_keep(state_tokens: torch.Tensor, data: dict) -> torch.Tensor:
    """Zero the state tokens of samples the processor marked ``state_keep == 0``.

    Mirrors ``GrootN2._forward_train_impl``: a state modality with
    ``exclude_state=True`` (or ``state_dropout_prob == 1.0``) makes the processor emit
    ``state_keep == 0`` at inference as well as training, so a policy trained
    proprio-free must not be fed proprio at deploy -- and an eval-time ``exclude_state``
    A/B is otherwise a silent no-op. Key absent => no-op, so other embodiments and
    existing checkpoints are unaffected.

    Args:
        state_tokens: (B, S, D) encoded state tokens.
        data: Inference batch; may carry a per-sample ``state_keep`` scalar.

    Returns:
        ``state_tokens``, with the tokens of dropped samples zeroed.
    """
    state_keep = data.get("state_keep")
    if state_keep is None:
        return state_tokens
    keep = torch.as_tensor(state_keep, device=state_tokens.device, dtype=state_tokens.dtype)
    return state_tokens * keep.reshape(-1, 1, 1)


@dataclasses.dataclass
class TokenMergePlan:
    """Video-specific geometry and routing for a mid-layer sequence reduction."""

    start_layer: int
    end_layer: int
    num_video_tokens: int
    num_frames: int
    reduced_height: int
    kernel_height: int
    reduced_width: int
    kernel_width: int
    reduced_token_indices: torch.Tensor
    reduced_attention_mask: BlockMask | None
    reduced_pad: int
    reduced_video_token_indices: torch.Tensor
    reduced_non_video_token_indices: torch.Tensor | None

    def is_reduced_layer(self, layer_idx: int) -> bool:
        return self.start_layer <= layer_idx < self.end_layer

    def keep_indices_per_segment(self, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fine and reduced row indices for a fine-length keep mask."""
        return (
            keep.nonzero(as_tuple=True)[0],
            keep[self.reduced_token_indices].nonzero(as_tuple=True)[0],
        )

    def reduce_sequence(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Mean-pool the video prefix and retain inputs needed to restore it."""
        batch_size, _, hidden_size = latents.shape
        video, rest = (
            latents[:, : self.num_video_tokens],
            latents[:, self.num_video_tokens :],
        )
        video_blocks = video.view(
            batch_size,
            self.num_frames,
            self.reduced_height,
            self.kernel_height,
            self.reduced_width,
            self.kernel_width,
            hidden_size,
        )
        pooled = video_blocks.mean(dim=(3, 5))
        return torch.cat([pooled.flatten(1, 3), rest], dim=1), (video, pooled)

    def restore_sequence(
        self,
        latents: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Restore fine video tokens with the reduced layer range's change."""
        # load-bearing: an identity kernel must leave a non-identity middle
        # range bitwise unchanged; subtracting and re-adding would round it.
        if self.kernel_height == 1 and self.kernel_width == 1:
            return latents

        batch_size, _, hidden_size = latents.shape
        video_input, pooled_input = state
        num_reduced_video_tokens = self.num_frames * self.reduced_height * self.reduced_width
        reduced_video, rest = (
            latents[:, :num_reduced_video_tokens],
            latents[:, num_reduced_video_tokens:],
        )
        reduced_delta = (
            reduced_video.view(
                batch_size,
                self.num_frames,
                self.reduced_height,
                self.reduced_width,
                hidden_size,
            )
            - pooled_input
        )
        video_delta = (
            reduced_delta[:, :, :, None, :, None, :]
            .expand(
                batch_size,
                self.num_frames,
                self.reduced_height,
                self.kernel_height,
                self.reduced_width,
                self.kernel_width,
                hidden_size,
            )
            .reshape(batch_size, self.num_video_tokens, hidden_size)
        )
        return torch.cat([video_input + video_delta, rest], dim=1)


class GrootN2(PreTrainedModel):
    """Video world model: Cosmos DiT with clean train/inference."""

    config_class = GrootN2Config
    supports_gradient_checkpointing = True

    def __init__(self, config: GrootN2Config):
        super().__init__(config)

        # 1. Build video tokenizer + video_rope + transformer (split architecture)
        dit_config = COSMOS_2B_CONFIG if config.model_size == "2B" else COSMOS_14B_CONFIG
        self.max_n_cameras = config.max_n_cameras
        latent_channels = VAE_LATENT_CHANNELS[config.vae_type]
        dit_config = dataclasses.replace(
            dit_config,
            max_n_cameras=self.max_n_cameras,
            view_embed_dim=config.view_embed_dim,
            in_channels=latent_channels + 1,  # +1 for the cond_mask channel
            out_channels=latent_channels,
        )
        self.video_tokenizer = VideoTokenizer(dit_config)
        # RoPE lives here (sibling to video_tokenizer), not inside it
        self.video_rope = MultiViewRoPE(
            dit_config.attention_head_dim,
            dit_config.max_size,
            dit_config.patch_size,
            rope_scale=dit_config.rope_scale,
        )
        transformer_config = dataclasses.replace(dit_config, max_n_cameras=self.max_n_cameras)
        token_merge_kernel = getattr(config, "token_merge_kernel", None)
        if token_merge_kernel is not None:
            _, tm_end = config.token_merge_layers
            if tm_end > transformer_config.num_layers:
                raise ValueError(
                    f"token_merge_layers end {tm_end} exceeds "
                    f"num_layers {transformer_config.num_layers}"
                )
        self.transformer = self._create_transformer(transformer_config)
        self.transformer.compile_flex_attention = config.compile_flex_attention
        self.transformer.requires_sequence_transform = token_merge_kernel is not None
        self.packer = ConcatPacker(flex_attention_block_size=config.flex_attention_block_size)

        # 1b. Action modules (conditional on action_cfg)
        self.has_action_head = config.action_cfg is not None
        if self.has_action_head:
            from groot.core.training.models.n2.components.action_modules import (
                ActionDecoder,
                ActionEncoder,
                ActionRoPE1D,
                StateEncoder,
            )

            ac = config.action_cfg
            hidden_size = dit_config.num_attention_heads * dit_config.attention_head_dim
            n_emb = ac.get("num_embodiments", 1)
            self.action_encoder = ActionEncoder(
                ac.get("max_action_dim", 32),
                hidden_size,
                n_emb,
                use_temporal_pos_emb=ac.get("use_temporal_pos_emb", False),
                action_horizon=ac.get("action_horizon", 48),
            )
            self.state_encoder = StateEncoder(ac.get("max_state_dim", 64), hidden_size, n_emb)
            self.state_dropout_prob = ac.get("state_dropout_prob", 0.0)
            self.action_decoder = ActionDecoder(hidden_size, ac.get("max_action_dim", 32), n_emb)
            self.action_rope = ActionRoPE1D(
                dit_config.attention_head_dim, max_len=ac.get("action_horizon", 48)
            )
            self.state_rope = ActionRoPE1D(
                dit_config.attention_head_dim, max_len=ac.get("action_horizon", 48)
            )
            self.attention_mask_pattern = ac.get("attention_mask_pattern", None)
        # 2. Frozen components
        self.text_encoder = CosmosTextEncoder(**config.text_encoder_cfg)
        self.vae = self.build_vae(config)

        # 3. Schedulers
        self.video_scheduler = self._make_video_scheduler(config)
        self.video_scheduler.setup_training(config.num_timestep_buckets)

        if self.has_action_head:
            ac = config.action_cfg
            self.action_scheduler = BetaFlowMatchEulerScheduler(
                BetaSchedulerConfig(
                    beta_alpha=ac.get("noise_beta_alpha", 1.5),
                    beta_beta=ac.get("noise_beta_beta", 1.0),
                    noise_s=ac.get("noise_s", 0.999),
                )
            )
            self.action_scheduler.setup_training()

        # 4. Freeze text encoder + VAE
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # 5. Optionally freeze transformer
        if not config.tune_dit:
            self.transformer.requires_grad_(False)

        # 6. LoRA (optional)
        if config.train_architecture == "lora" and config.tune_dit:
            self._apply_lora()
            # Keep action modules full-rank (not wrapped by LoRA)
            if self.has_action_head:
                self.action_encoder.requires_grad_(True)
                self.state_encoder.requires_grad_(True)
                self.action_decoder.requires_grad_(True)
        elif config.tune_dit and config.use_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # 7. Config values
        self.cond_timestep = config.cond_timestep
        self.tiled = config.tiled
        self.tile_size = (config.tile_size_height, config.tile_size_width)
        self.tile_stride = (config.tile_stride_height, config.tile_stride_width)

    @staticmethod
    def build_vae(config: GrootN2Config):
        """Build the frozen video VAE for `config.vae_type`.

        Both VAE classes already default z_dim/dim to their own geometry, so
        `vae_cfg` only ever carries the checkpoint path.
        """
        vae_cls = WanVideoVAE if config.vae_type == "wan21" else WanVideoVAE38
        vae = vae_cls(**config.vae_cfg)
        # A vae_cfg that overrides z_dim would give the DiT and the VAE different
        # latent widths -- which trains happily on garbage rather than crashing.
        expected = VAE_LATENT_CHANNELS[config.vae_type]
        if vae.z_dim != expected:
            raise ValueError(
                f"vae_type={config.vae_type!r} implies {expected} latent channels, "
                f"but vae_cfg built a VAE with z_dim={vae.z_dim}"
            )
        return vae

    @staticmethod
    def _make_video_scheduler(config):
        """Build the video flow-matching scheduler based on `config.video_scheduler_type`.

        "unipc" (default): shifted-sigma schedule + UniPC solver. Many-step inference.
        "beta":            Beta(alpha, beta) timestep sampling + Euler solver. Biases
                           toward the near-data regime for few/1-step inference.
        """
        scheduler_type = getattr(config, "video_scheduler_type", "unipc")
        if scheduler_type == "beta":
            return BetaFlowMatchEulerScheduler(
                BetaSchedulerConfig(
                    beta_alpha=getattr(config, "video_noise_beta_alpha", 2.0),
                    beta_beta=getattr(config, "video_noise_beta_beta", 1.0),
                    noise_s=getattr(config, "video_noise_s", 0.999),
                    num_timestep_buckets=config.num_timestep_buckets,
                    # Match UniPC's sign convention (v = noise − x_0) so a
                    # UniPC-pretrained video model can be posttrained with
                    # Beta sampling without a sign-flip in its learned output.
                    velocity_direction="noise_minus_x0",
                    # Beta(α, β=1) with α≥7 concentrates training at high σ →
                    # DreamZero σ-shift inference grid (linspace 1→ε then σ-shift,
                    # ending at σ=0, variable Euler dt) is required so inference
                    # queries land in the trained noise mass. Without this the
                    # video output collapses to confetti at every N>1.
                    use_shifted_inference_grid=True,
                )
            )
        return FlowMatchUniPCScheduler(
            SchedulerConfig(
                shift=config.train_video_sigma_shift,
                num_train_timesteps=config.num_timestep_buckets,
            )
        )

    @staticmethod
    def _pixel_to_latent_frames(num_pixel_frames: int) -> int:
        """Wan VAE: first pixel frame → latent 0, then every 4 pixel frames → 1 latent."""
        if num_pixel_frames <= 0:
            return 0
        return (num_pixel_frames - 1) // VAE_TEMPORAL_FACTOR + 1

    @staticmethod
    def _per_view_prefix_mask(
        T: int,
        T_per_view: int,
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame_ids = torch.arange(T, device=device) % T_per_view
        return frame_ids < prefix_len

    @staticmethod
    def _latent_padded_mask_from_video_padded_mask(
        video_padded_mask: torch.Tensor,
        n_cameras: int,
        T_lat_per_view: int,
    ) -> torch.Tensor:
        B = video_padded_mask.shape[0]
        T_pixel_per_view = video_padded_mask.shape[1] // n_cameras
        padded_pixel_per_view = (
            video_padded_mask.view(B, n_cameras, T_pixel_per_view).sum(dim=-1).to(torch.long)
        )
        padded_lat_per_view = torch.where(
            padded_pixel_per_view > 0,
            (padded_pixel_per_view - 1) // VAE_TEMPORAL_FACTOR + 1,
            padded_pixel_per_view,
        )
        latent_frame_ids = torch.arange(
            T_lat_per_view,
            device=video_padded_mask.device,
        ).view(1, 1, T_lat_per_view)
        return (latent_frame_ids >= T_lat_per_view - padded_lat_per_view.unsqueeze(-1)).view(
            B, n_cameras * T_lat_per_view
        )

    @staticmethod
    def _require_python_bool(data: dict, key: str, default: bool) -> bool:
        value = data.get(key, default)
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"GrootN2 training expects `{key}` as a Python bool before forward; "
                "normalize it in the collator or caller to avoid torch.compile graph breaks"
            )
        return bool(value)

    @staticmethod
    def _require_python_int(data: dict, key: str, default: int) -> int:
        value = data.get(key, default)
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"GrootN2 training expects `{key}` as a Python int before forward; "
                "normalize it in the collator or caller to avoid torch.compile graph breaks"
            )
        return int(value)

    @staticmethod
    def _expand_timesteps(
        timestep: torch.Tensor,
        B: int,
        T: int,
        shape_info: dict,
        L: int,
    ) -> torch.Tensor:
        """Convert (B, 1, T, 1, 1) per-frame sigmas to (B, L) per-token sigmas."""
        post_T = shape_info["post_T"]
        post_H = shape_info["post_H"]
        post_W = shape_info["post_W"]
        if timestep.ndim == 1:
            return timestep.unsqueeze(1).expand(B, L)
        elif timestep.ndim == 5:
            assert timestep.shape == (B, 1, T, 1, 1)
            per_frame = timestep[:, 0, :, 0, 0]  # (B, T)
            return per_frame[:, :, None, None].expand(B, post_T, post_H, post_W).reshape(B, L)
        else:
            raise ValueError(
                f"Expected timestep shape [B] or [B, 1, T, 1, 1], got {timestep.shape}"
            )

    def _apply_lora(self):
        modules = [m.strip() for m in self.config.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            r=self.config.lora_rank, lora_alpha=self.config.lora_alpha, target_modules=modules
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    def _preprocess_video(self, videos: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, C) uint8 → (B, C, T, H, W) float [-1, 1]."""
        x = videos.float().permute(0, 4, 1, 2, 3) / 255.0  # (B, C, T, H, W)
        # [0, 1] -> [-1, 1]. Scalar ops instead of v2.Normalize: the transform
        # re-uploads its CPU mean/std every call, which also breaks CUDA graph
        # capture. Values match Normalize bit-for-bit; .contiguous() also
        # matches the old reshape path's layout -- the elementwise op alone
        # keeps the permute's strides, and layout-sensitive downstream kernels
        # then accumulate in a different order (measured 9.3e-3 worst-key
        # action drift vs main without it).
        return ((x - 0.5) / 0.5).contiguous()

    # ─── Transformer construction hook ────────────────────────────────────────

    def _create_transformer(self, transformer_config) -> CosmosTransformer3DModel:
        """Construct the transformer module."""
        return CosmosTransformer3DModel(transformer_config)

    # ─── Transformer call hook ────────────────────────────────────────────────

    def _call_transformer(
        self,
        packed,
        prompt_embs: torch.Tensor,
        *,
        denoising_step: int = 0,
        is_cfg: bool = False,
        To: int = 1,
        total_seqlen=None,
        last_mini_batch_params_dict=None,
        state_collector=None,
        **transformer_kwargs,
    ) -> torch.Tensor:
        """Single hook every ``self.transformer(packed, ...)`` call routes through."""
        del denoising_step, is_cfg, To, total_seqlen
        del last_mini_batch_params_dict, state_collector
        return self.transformer._forward_with_routing_indices(  # noqa: SLF001
            packed.tokens,
            packed.timesteps,
            packed.rope_freqs,
            prompt_embs,
            packed.modality_ids,
            extra_pos_emb=packed.extra_pos_emb,
            attention_mask=packed.attention_mask,
            pad=packed.pad,
            video_token_indices=packed.video_token_indices,
            non_video_token_indices=packed.non_video_token_indices,
            **transformer_kwargs,
        )

    def _compute_additional_training_loss(
        self,
        *,
        data: dict,
        packed,
        prompt_embs: torch.Tensor,
        out_tokens: torch.Tensor,
        token_valid_mask: torch.Tensor | None,
        shape_info: dict,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Extension hook for project-local auxiliary training objectives.

        Base N2 has no auxiliary objective. Training extensions may consume the
        already-built packed inputs/outputs without copying the full N2 forward or
        rerunning the VAE/transformer. The returned scalar is added to the primary
        loss; metric tensors are detached before returning to the trainer.
        """
        del data, packed, prompt_embs, out_tokens, token_valid_mask, shape_info
        return None, {}

    def _training_packing_kwargs(
        self,
        *,
        data: dict,
        video_tokens: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_tokens: torch.Tensor | None,
    ) -> dict:
        """Supply optional token groups and attention policy before the single pack."""
        del data, video_tokens, video_timesteps, action_tokens
        return {}

    def _joint_inference_packing_kwargs(
        self,
        *,
        data: dict,
        video_tokens: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_tokens: torch.Tensor,
        denoising_step: int,
        total_denoising_steps: int,
    ) -> dict:
        """Supply project tokens at the selected joint solver step before packing."""
        del (
            data,
            video_tokens,
            video_timesteps,
            action_tokens,
            denoising_step,
            total_denoising_steps,
        )
        return {}

    def _build_token_merge_attention_mask(
        self,
        packed: PackedSequence,
        coarse_token_idx: torch.Tensor,
        token_valid_mask: torch.Tensor | None,
    ) -> tuple[BlockMask | None, int]:
        """Build the reduced mask used inside the token-merge layer range.

        Project extensions that add modalities with custom attention semantics
        may override this hook. Base N2 retains its existing four-modality mask
        construction exactly.
        """
        if packed.attention_mask is None:
            return None, 0
        return _build_attention_mask(
            packed.modality_ids[:, coarse_token_idx],
            packed.tokens.device,
            getattr(self, "attention_mask_pattern", None),
            token_valid_mask[:, coarse_token_idx] if token_valid_mask is not None else None,
            block_size=self.packer.flex_attention_block_size,
        )

    def _build_token_merge_plan(
        self,
        packed: PackedSequence,
        shape_info: dict,
        token_valid_mask: torch.Tensor | None,
    ) -> TokenMergePlan | None:
        """Coarse-sequence plan for the mid-layer video token merge, or None.

        Must stay traceable (NOT ``@torch.compiler.disable``): an eagerly built
        BlockMask entering a compiled region has its mask closure tensors lifted
        as CUDA-graph static inputs. Built in-graph, like the packer's fine mask,
        those tensors remain graph intermediates. The single-pattern config
        constraint also keeps the mask build off the RNG stream.

        Callers must thread the returned plan as a CONDITIONAL kwarg (never
        pass sequence_transform=None explicitly): external installers may
        monkeypatch `_call_transformer` with transform-unaware signatures.
        """
        kernel = getattr(self.config, "token_merge_kernel", None)
        if kernel is None:
            return None
        start, end = self.config.token_merge_layers
        _, kh, kw = kernel
        post_T, post_H, post_W = shape_info["post_T"], shape_info["post_H"], shape_info["post_W"]
        assert (
            post_H % kh == 0 and post_W % kw == 0
        ), f"token_merge kernel ({kh},{kw}) must divide (post_H, post_W)=({post_H},{post_W})"
        device = packed.tokens.device
        L_video = post_T * post_H * post_W
        L_total = packed.tokens.shape[1]
        video_corners = (
            torch.arange(L_video, device=device)
            .view(post_T, post_H, post_W)[:, ::kh, ::kw]
            .reshape(-1)
        )
        coarse_token_idx = torch.cat([video_corners, torch.arange(L_video, L_total, device=device)])
        mask_builder = getattr(self, "_build_token_merge_attention_mask", None)
        if mask_builder is None:
            attention_mask, pad = GrootN2._build_token_merge_attention_mask(
                self,
                packed,
                coarse_token_idx,
                token_valid_mask,
            )
        else:
            attention_mask, pad = mask_builder(
                packed,
                coarse_token_idx,
                token_valid_mask,
            )
        L_video_coarse = video_corners.numel()
        L_nonvideo = L_total - L_video
        return TokenMergePlan(
            start_layer=start,
            end_layer=end,
            num_video_tokens=L_video,
            num_frames=post_T,
            reduced_height=post_H // kh,
            kernel_height=kh,
            reduced_width=post_W // kw,
            kernel_width=kw,
            reduced_token_indices=coarse_token_idx,
            reduced_attention_mask=attention_mask,
            reduced_pad=pad,
            reduced_video_token_indices=torch.arange(L_video_coarse, device=device),
            reduced_non_video_token_indices=(
                torch.arange(L_video_coarse, L_video_coarse + L_nonvideo, device=device)
                if L_nonvideo > 0
                else None
            ),
        )

    @torch.compiler.disable()
    def _sample_training_noise_and_timesteps(
        self,
        latents: torch.Tensor,
        batch_size: int,
        device: torch.device,
        action_samples: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Sample training noise/timesteps in the legacy RNG order.

        Keep the random draws outside the compiled deterministic core without
        changing the sequence of RNG-consuming operations:
        video noise, video timestep, action noise, action timestep.
        """
        video_noise = torch.randn_like(latents)
        video_ts, video_sigmas = self.video_scheduler.sample_timesteps(batch_size, device)

        action_noise = action_ts = action_sigmas = None
        if action_samples is not None:
            action_noise = torch.randn_like(action_samples)
            action_ts, action_sigmas = self.action_scheduler.sample_timesteps(
                batch_size,
                device,
                action_samples.dtype,
            )

        return (
            video_noise,
            video_ts,
            video_sigmas,
            action_noise,
            action_ts,
            action_sigmas,
        )

    @torch.compiler.disable()
    def _video_loss_weight(
        self,
        video_ts_per_frame: torch.Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Keep UniPC CPU weight lookup and H2D copy out of CUDA graph capture."""
        return (
            self.video_scheduler.training_weight(video_ts_per_frame.flatten(0, 1))
            .unflatten(0, (batch_size, num_frames))
            .to(device=device, dtype=dtype)
        )

    # ─── Training Forward ─────────────────────────────────────────────────────

    def _forward_train_impl(
        self,
        inputs: dict,
        validate: bool = True,
        return_action_loss_details: bool = False,
        **transformer_kwargs,
    ) -> dict | BatchFeature:
        """Shared training forward pass.

        ``return_action_loss_details`` exposes the unreduced actor loss and masks
        for wrappers that need custom loss reductions. Normal N2 training keeps
        the legacy scalar-loss output.

        ``transformer_kwargs`` are threaded through to
        :py:meth:`_call_transformer` so subclasses can pass extra per-call state.
        """
        self.text_encoder.eval()
        self.vae.eval()

        data = inputs if not isinstance(inputs, BatchFeature) else inputs.data
        if "inputs" in data:
            data = data["inputs"]
        if isinstance(data, tuple):
            raise TypeError("GrootN2 training expects single_turn batches, not tuple batches")

        has_real_action = self._require_python_bool(data, "has_real_action", True)
        has_action = self.has_action_head and "action" in data and has_real_action

        if return_action_loss_details:
            required_keys = ("action", "action_mask", "state", "state_keep", "has_real_action")
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(
                    f"GrootN2 action loss details require {required_keys}; missing {missing_keys}"
                )
            assert (
                has_action is True
            ), "GrootN2 action loss details require an action head and real actions"
            if validate:
                assert (
                    has_real_action
                ), "GrootN2 action loss details require all samples to have real actions"

        videos = data["images"]  # (B, T, H, W, C) uint8
        text_ids = data["text"]

        # 1. Preprocess video
        videos = self._preprocess_video(videos)
        device = videos.device

        # 2. Encode text (frozen)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        with torch.no_grad():
            prompt_embs = self.text_encoder(text_ids.to(device)).to(
                device=device, dtype=torch.bfloat16
            )

        # 3. Encode video to latents (frozen), per-view for multiview
        n_cameras = self.max_n_cameras
        latents = self.encode_video(videos)

        B, C, T, H, W = latents.shape

        # 4. Determine conditioning frames
        # num_cond_frames = number of pixel frames with delta_indices <= 0
        # VAE temporal compression: (n_pixel - 1) // 4 + 1 latent frames
        num_cond_pixel = self._require_python_int(data, "num_cond_frames", 0)
        num_cond_latent_per_view = self._pixel_to_latent_frames(num_cond_pixel)

        assert T % n_cameras == 0, f"Latent T={T} not divisible by n_cameras={n_cameras}"
        T_lat_per_view = T // n_cameras
        total_cond_latent = num_cond_latent_per_view * n_cameras
        num_denoise_latent = T - total_cond_latent
        assert (
            num_denoise_latent > 0
        ), f"No frames to denoise: T={T}, total_cond_latent={total_cond_latent}"
        cond_frame_mask_1d = self._per_view_prefix_mask(
            T,
            T_lat_per_view,
            num_cond_latent_per_view,
            device,
        )

        action_samples = data["action"].to(device) if has_action else None

        # 5. Sample noise and timesteps. This stays outside the compiled
        # deterministic body but preserves the old RNG call order exactly.
        (
            video_noise,
            video_ts,
            video_sigmas,
            action_noise,
            action_ts,
            action_sigmas,
        ) = self._sample_training_noise_and_timesteps(
            latents,
            B,
            device,
            action_samples,
        )

        video_ts_per_frame = video_ts.expand(B, T).clone()
        video_sigmas_per_frame = video_sigmas.expand(B, T).clone()

        # 6. Add noise to all frames, then replace conditioning frames with clean latents
        latents_flat = latents.transpose(1, 2)
        noise_flat = video_noise.transpose(1, 2)
        noisy_video = (
            self.video_scheduler.add_noise(
                latents_flat.flatten(0, 1),
                noise_flat.flatten(0, 1),
                video_ts_per_frame.flatten(0, 1),
            )
            .unflatten(0, (B, T))
            .transpose(1, 2)
        )

        cond_frame_mask_5d = cond_frame_mask_1d.view(1, 1, T, 1, 1)
        noisy_video = torch.where(cond_frame_mask_5d, latents, noisy_video)
        video_sigmas_per_frame = torch.where(
            cond_frame_mask_1d.view(1, T),
            torch.full_like(video_sigmas_per_frame, self.cond_timestep),
            video_sigmas_per_frame,
        )

        video_target = self.video_scheduler.training_target(latents_flat, noise_flat).transpose(
            1, 2
        )

        # 7. Conditioning mask
        cond_mask = cond_frame_mask_5d.to(latents.dtype).expand(B, 1, T, H, W)

        # 8. Patchify video tokens
        video_sigmas_5d = video_sigmas_per_frame.view(B, 1, T, 1, 1)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            video_tokens, video_extra_pos_emb, shape_info = self.video_tokenizer.patchify(
                noisy_video, cond_mask, None, n_cameras
            )
            L_video = video_tokens.shape[1]
            video_token_sigmas = self._expand_timesteps(video_sigmas_5d, B, T, shape_info, L_video)

            # Video RoPE
            video_rope_freqs = self.video_rope(noisy_video, n_cameras=n_cameras)

            action_tokens = action_token_sigmas = action_rope_freqs = None
            state_tokens = state_token_sigmas = state_rope_freqs = None

            # 8b. Action/state encoding (if present)
            if has_action:
                actions = action_samples  # (B, T_action, max_action_dim)
                assert actions is not None
                assert action_noise is not None
                assert action_ts is not None
                assert action_sigmas is not None
                action_mask = data["action_mask"].to(device)  # (B, T_action, max_action_dim)
                states = data["state"].to(device)  # (B, T_state, max_state_dim)
                embodiment_id = data.get("embodiment_id", torch.zeros(B, dtype=torch.long)).to(
                    device
                )

                T_action = actions.shape[1]
                action_ts_per_step = action_ts.expand(B, T_action)
                action_sigmas_per_step = action_sigmas.expand(B, T_action)
                noisy_actions = self.action_scheduler.add_noise(
                    actions.flatten(0, 1),
                    action_noise.flatten(0, 1),
                    action_ts_per_step.flatten(0, 1),
                ).unflatten(0, (B, T_action))
                action_target = self.action_scheduler.training_target(actions, action_noise)

                # ActionEncoder uses t (N1.7 convention); transformer AdaLN uses sigma
                action_tokens = self.action_encoder(
                    noisy_actions, action_ts_per_step, embodiment_id
                )
                state_tokens = self.state_encoder(states, embodiment_id)

                # State dropout: zero out the encoded state tokens for samples
                # where the processor sampled keep=0. The mask is precomputed on
                # CPU and passed in as `state_keep` (shape [B]), so the compiled
                # forward has no torch.rand, no branch, no .to() — keeping
                # reduce-overhead CUDA graphs valid.
                state_keep = data["state_keep"].to(device=device, dtype=state_tokens.dtype)
                state_tokens = state_tokens * state_keep[:, None, None]

                action_rope_freqs = self.action_rope(action_tokens)
                state_rope_freqs = self.state_rope(state_tokens)
                action_token_sigmas = action_sigmas_per_step  # sigma convention for AdaLN
                state_token_sigmas = torch.full(
                    (B, state_tokens.shape[1]), self.cond_timestep, device=device
                )

            # 8c. Expand precomputed frame-level validity mask to token-level.
            # The processor builds frame_valid_mask as [video_frames, state, action].
            # We expand video frames to tokens with repeat_interleave.
            # All heavy computation is done in the processor (numpy, CPU) — the forward
            # only does .to(), slice, repeat_interleave, cat. Minimal new GPU ops.
            token_valid_mask = None
            if "frame_valid_mask" in data:
                frame_valid = data["frame_valid_mask"].to(device=device, dtype=torch.bool)
                tokens_per_frame = shape_info["post_H"] * shape_info["post_W"]
                n_video_frames = T  # n_cameras * T_lat_per_view
                video_frame_valid = frame_valid[:, :n_video_frames]
                rest_valid = frame_valid[:, n_video_frames:]
                video_token_valid = video_frame_valid.repeat_interleave(tokens_per_frame, dim=1)
                token_valid_mask = torch.cat([video_token_valid, rest_valid], dim=1)

            # 8d. Pack → transformer → unpack
            video_cond_token_mask = (
                (video_token_sigmas == self.cond_timestep) if has_action else None
            )
            packed = self.packer.pack(
                video_tokens=video_tokens,
                video_timesteps=video_token_sigmas,
                video_rope=video_rope_freqs,
                video_extra_pos_emb=video_extra_pos_emb,
                video_cond_token_mask=video_cond_token_mask,
                action_tokens=action_tokens,
                action_timesteps=action_token_sigmas,
                action_rope=action_rope_freqs,
                state_tokens=state_tokens,
                state_timesteps=state_token_sigmas,
                state_rope=state_rope_freqs,
                attention_mask_pattern=getattr(self, "attention_mask_pattern", None),
                token_valid_mask=token_valid_mask,
                **self._training_packing_kwargs(
                    data=data,
                    video_tokens=video_tokens,
                    video_timesteps=video_token_sigmas,
                    action_tokens=action_tokens,
                ),
            )
            token_merge = self._build_token_merge_plan(packed, shape_info, token_valid_mask)
            if token_merge is not None:
                transformer_kwargs = {**transformer_kwargs, "sequence_transform": token_merge}
            out_tokens = self._call_transformer(packed, prompt_embs, **transformer_kwargs)
            additional_loss, additional_metrics = self._compute_additional_training_loss(
                data=data,
                packed=packed,
                prompt_embs=prompt_embs,
                out_tokens=out_tokens,
                token_valid_mask=token_valid_mask,
                shape_info=shape_info,
            )
            video_out = packed.unpack(out_tokens, (MODALITY_VIDEO, MODALITY_VIDEO_COND))
            pred_video = self.video_tokenizer.unpatchify(video_out, shape_info)

        # 9. Video loss on denoising frames only. Keep the full static timeline
        # and zero conditioning frames through weights to avoid dynamic indexing.
        loss_per_frame = F.mse_loss(
            pred_video.float(), video_target.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        denoise_weight = (~cond_frame_mask_1d).to(dtype=loss_per_frame.dtype).view(1, T)
        weight = self._video_loss_weight(
            video_ts_per_frame,
            B,
            T,
            device,
            loss_per_frame.dtype,
        )
        weight = weight * denoise_weight

        # 10. Zero out loss on padded frames (clamped past episode end)
        # video_padded_mask: (B, V*T_pixel), 1=padded (past end), 0=real
        # Padded frames are at the tail of EACH VIEW, not just the global tail
        # Each batch element can have a different number of padded frames.
        video_padded_mask = data.get("video_padded_mask", None)
        # Project-local training objectives may make a target ineligible while
        # retaining it as a fixed input/key. Keep this loss-only mask separate
        # from frame_valid_mask, which defines attention validity.
        video_loss_frame_mask = data.get("video_loss_frame_mask", None)
        if video_loss_frame_mask is not None:
            frame_valid = video_loss_frame_mask.to(device=device, dtype=torch.bool)
            if frame_valid.ndim == 1:
                frame_valid = frame_valid.unsqueeze(0).expand(B, -1)
            assert frame_valid.shape == (
                B,
                T,
            ), f"video_loss_frame_mask shape {frame_valid.shape} != expected {(B, T)}"
            weight = weight * frame_valid.to(dtype=weight.dtype)
        elif "frame_valid_mask" in data:
            frame_valid = data["frame_valid_mask"].to(device=device, dtype=torch.bool)
            if frame_valid.ndim == 1:
                frame_valid = frame_valid.unsqueeze(0).expand(B, -1)
            assert (
                frame_valid.shape[0] == B
            ), f"frame_valid_mask batch={frame_valid.shape[0]} != B={B}"
            assert (
                frame_valid.shape[1] >= T
            ), f"frame_valid_mask length {frame_valid.shape[1]} shorter than video frames T={T}"
            weight = weight * frame_valid[:, :T].to(dtype=weight.dtype)
        elif video_padded_mask is not None:
            if isinstance(video_padded_mask, torch.Tensor):
                video_padded_mask = video_padded_mask.to(device)
            else:
                video_padded_mask = torch.tensor(video_padded_mask, device=device)
            if video_padded_mask.ndim == 1:
                video_padded_mask = video_padded_mask.unsqueeze(0).expand(B, -1)
            assert (
                video_padded_mask.shape[0] == B
            ), f"video_padded_mask batch={video_padded_mask.shape[0]} != B={B}"
            assert (
                video_padded_mask.shape[1] % max(n_cameras, 1) == 0
            ), f"video_padded_mask length {video_padded_mask.shape[1]} not divisible by n_cameras={n_cameras}"
            latent_padded_mask = self._latent_padded_mask_from_video_padded_mask(
                video_padded_mask,
                n_cameras,
                T_lat_per_view,
            )
            # 1=real frame (compute loss), 0=padded (skip loss)
            weight = weight * (~latent_padded_mask).to(dtype=weight.dtype)

        video_loss = (loss_per_frame * weight).sum() / weight.sum().clamp(min=1.0)

        if not torch.compiler.is_compiling():
            weight_sum = weight.sum()
            if video_loss.item() == 0 and weight_sum.item() > 0:
                logging.warning(
                    "video_loss=0 but weight.sum=%.4f, loss_per_frame range=[%.6f, %.6f]",
                    weight_sum.item(),
                    loss_per_frame.min().item(),
                    loss_per_frame.max().item(),
                )
            elif weight_sum.item() == 0:
                padded_sum = (
                    video_padded_mask.sum().item() if video_padded_mask is not None else "N/A"
                )
                logging.warning(
                    "All weights are zero — all prediction frames padded. "
                    "video_padded_mask sum=%s",
                    padded_sum,
                )

        # 11. Action loss
        # action_cfg is None for video-only recipes, which still scale the video loss.
        action_cfg = self.config.action_cfg or {}
        video_loss_scaled = action_cfg.get("video_loss_scale", 1.0) * video_loss
        if has_action:
            action_out = packed.unpack(out_tokens, MODALITY_ACTION)
            pred_actions = self.action_decoder(action_out, embodiment_id)
            action_loss_raw = F.mse_loss(
                pred_actions.float(), action_target.float(), reduction="none"
            )
            if return_action_loss_details:
                return BatchFeature(
                    data={
                        "video_loss": video_loss,
                        "video_loss_scaled": video_loss_scaled,
                        "dynamics_loss": video_loss.detach(),
                        "action_loss_per_elem": action_loss_raw,
                        "action_mask": action_mask,
                        "action_target": action_target,
                        "action_velocity_pred": pred_actions,
                    }
                )
            action_loss_per_elem = action_loss_raw * action_mask.float()
            action_loss = action_loss_per_elem.sum() / (action_mask.sum() + 1e-6)
            loss = (
                video_loss_scaled
                + self.config.action_cfg.get("action_loss_scale", 1.0) * action_loss
            )
            if additional_loss is not None:
                loss = loss + additional_loss
            result = {
                "loss": loss,
                "dynamics_loss": video_loss.detach(),
                "action_loss": action_loss.detach(),
            }
            result.update({name: value.detach() for name, value in additional_metrics.items()})
            return result

        loss = video_loss_scaled
        if additional_loss is not None:
            loss = loss + additional_loss
        result = {
            "loss": loss,
            "dynamics_loss": video_loss.detach(),
        }
        result.update({name: value.detach() for name, value in additional_metrics.items()})
        return result

    def forward(self, inputs: dict, validate: bool = True, **kwargs) -> dict:
        """Training forward pass. Returns dict with 'loss' for HF Trainer.

        ``kwargs`` are threaded through to :py:meth:`_forward_train_impl` and
        :py:meth:`_call_transformer` so subclasses can pass extra per-call
        state via ``super().forward(...)``.
        """
        return self._forward_train_impl(
            inputs,
            validate=validate,
            return_action_loss_details=False,
            **kwargs,
        )

    def forward_action_loss_details(
        self,
        inputs: dict,
        *,
        validate: bool = True,
    ) -> BatchFeature:
        """Return video loss plus unreduced action flow-matching loss details."""
        return self._forward_train_impl(
            inputs,
            validate=validate,
            return_action_loss_details=True,
        )

    # ─── Encoding helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to latents, handling multiview per-view VAE.

        Args:
            videos: (B, C, T, H, W) float [-1, 1] preprocessed video.
                For multiview: T = V * T_per_view.

        Returns:
            (B, C_lat, T_lat, H_lat, W_lat) latents.
            For multiview: T_lat = V * T_lat_per_view.
        """
        device = videos.device
        vae_dtype = next(self.vae.parameters()).dtype
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)
        n_cameras = self.max_n_cameras

        B_vid = videos.shape[0]
        T_total = videos.shape[2]
        assert T_total % n_cameras == 0, f"Video T={T_total} not divisible by n_cameras={n_cameras}"
        T_per_view = T_total // n_cameras
        videos_per_view = videos.reshape(
            B_vid, videos.shape[1], n_cameras, T_per_view, *videos.shape[3:]
        )
        videos_per_view = videos_per_view.permute(0, 2, 1, 3, 4, 5).reshape(
            B_vid * n_cameras, videos.shape[1], T_per_view, *videos.shape[3:]
        )
        latents_per_view = self.vae.encode(
            videos_per_view.to(vae_dtype),
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        C_lat = latents_per_view.shape[1]
        T_lat = latents_per_view.shape[2]
        H_lat, W_lat = latents_per_view.shape[3], latents_per_view.shape[4]
        latents = latents_per_view.reshape(B_vid, n_cameras, C_lat, T_lat, H_lat, W_lat)
        latents = latents.permute(0, 2, 1, 3, 4, 5).reshape(
            B_vid, C_lat, n_cameras * T_lat, H_lat, W_lat
        )
        return latents.to(device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text to embeddings.

        Args:
            text_ids: (B, N) token ids from the processor.

        Returns:
            (B, N, D) text embeddings in bfloat16.
        """
        device = text_ids.device
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        return self.text_encoder(text_ids).to(device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> list[torch.Tensor]:
        """Decode latents to pixel video, per-view for multiview.

        Args:
            latents: (B, C, T_lat, H_lat, W_lat). For multiview: T_lat = V * T_lat_per_view.

        Returns:
            List of (B, T_pixel, H, W, C) uint8 tensors, one per view.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        n_cameras = self.max_n_cameras
        assert (
            latents.shape[2] % n_cameras == 0
        ), f"Latent T={latents.shape[2]} not divisible by n_cameras={n_cameras}"
        T_per_view_lat = latents.shape[2] // n_cameras

        results = []
        for v in range(n_cameras):
            start = v * T_per_view_lat
            view_latents = latents[:, :, start : start + T_per_view_lat]
            decoded = self.vae.decode(
                view_latents.to(vae_dtype),
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            )
            # (B, C, T, H, W) -> (B, T, H, W, C) uint8
            frames = decoded.permute(0, 2, 3, 4, 1)
            frames = ((frames.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            results.append(frames)
        return results

    def _extract_cond_latents(
        self,
        latents: torch.Tensor,
        num_cond_latent_per_view: int,
    ) -> torch.Tensor:
        """Extract conditioning latents from full encoded video.

        Args:
            latents: (B, C, T_lat, H_lat, W_lat). For multiview: T_lat = V * T_lat_per_view.
            num_cond_latent_per_view: number of conditioning latent frames per view.

        Returns:
            (B, C, T_cond, H_lat, W_lat). For multiview: T_cond = V * num_cond_latent_per_view.
        """
        n_cameras = self.max_n_cameras
        assert (
            latents.shape[2] % n_cameras == 0
        ), f"Latent T={latents.shape[2]} not divisible by n_cameras={n_cameras}"
        T_lat_per_view = latents.shape[2] // n_cameras
        assert (
            num_cond_latent_per_view <= T_lat_per_view
        ), f"num_cond_latent_per_view={num_cond_latent_per_view} > T_lat_per_view={T_lat_per_view}"
        cond_slices = []
        for v in range(n_cameras):
            start = v * T_lat_per_view
            cond_slices.append(latents[:, :, start : start + num_cond_latent_per_view])
        return torch.cat(cond_slices, dim=2)

    # ─── Inference ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_inference_token_valid_mask(
        video_padded_mask,
        *,
        B: int,
        n_cameras: int,
        T_per_view: int,
        tokens_per_frame: int,
        n_extra_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Mirror the training-time ``token_valid_mask`` at inference.

        This prevents real-view query tokens from attending to padded-view KVs.

        Training (`forward`) builds this mask from `frame_valid_mask` and
        passes it to `SequencePacker.pack(...)` — see line 487 of this file.
        The inference paths (`_denoise_video`, `_denoise_action`,
        `_denoise_joint`, `_precompute_cond_kv_cache`) historically did not,
        so for any ckpt where `max_n_cameras > actual_n_cameras` the padded
        view KVs (VAE(zeros)+noise) leaked into real-view predictions
        through bidirectional attention. The symptom is dark/noisy output
        whose severity scales with the padded fraction (≈37% brightness
        drop at 3/4 padded, marginal at 1/4 padded).

        Source of truth for per-view real/padded: `data["video_padded_mask"]`
        from the processor (pixel-level, shape (B, n_cameras * T_pixel_per_view)).
        A view is "real" iff none of its pixel frames are padded. We expand
        the per-view flag across ALL T_per_view latent frames (cond + gen)
        and then to tokens via `repeat_interleave(tokens_per_frame)`, then
        concatenate `n_extra_tokens` of all-valid for any non-video tokens
        (state, action) that the caller is packing.

        Returns None when `video_padded_mask` is unavailable — preserves
        legacy unmasked behavior so this is safe to call from every site.
        """
        if video_padded_mask is None:
            return None
        if not isinstance(video_padded_mask, torch.Tensor):
            video_padded_mask = torch.tensor(video_padded_mask, device=device)
        else:
            video_padded_mask = video_padded_mask.to(device)
        if video_padded_mask.ndim == 1:
            video_padded_mask = video_padded_mask.unsqueeze(0).expand(B, -1)
        assert video_padded_mask.shape[1] % n_cameras == 0, (
            f"video_padded_mask length {video_padded_mask.shape[1]} not divisible by "
            f"n_cameras={n_cameras}"
        )
        T_pixel_per_view = video_padded_mask.shape[1] // n_cameras
        vpm_per_view = video_padded_mask.reshape(B, n_cameras, T_pixel_per_view)
        view_is_real = vpm_per_view.sum(dim=-1) == 0  # (B, n_cameras)
        video_frame_valid = view_is_real.repeat_interleave(T_per_view, dim=1)  # (B, T_full)
        video_token_valid = video_frame_valid.repeat_interleave(tokens_per_frame, dim=1)
        if n_extra_tokens > 0:
            rest_valid = torch.ones(B, n_extra_tokens, dtype=torch.bool, device=device)
            return torch.cat([video_token_valid, rest_valid], dim=1)
        return video_token_valid

    def _denoise_action(
        self,
        data: dict,
        cond_latent: torch.Tensor,
        prompt_embs: torch.Tensor,
        neg_prompt_embs: torch.Tensor | None,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Denoise action latents using conditioning frames + state.

        Runs an independent action diffusion loop. Video denoising tokens are
        not needed — the attention mask ensures action only sees video_cond,
        state, and other action tokens.

        When ``use_kv_cache`` is True, the static (video_cond + state) token
        prefix is run once through the transformer to precompute per-layer
        self-attention K/V, then the inner loop processes only action tokens,
        splicing the cached prefix back in. This is a pure speed optimization
        that relies on the disjoint attention pattern: the static prefix never
        attends to action, so its hidden states — and their K/V — are
        identical across all denoising steps. With CFG, positive and negative
        prompts need separate caches (cross-attention to text changes the
        static-token trajectory through the stack).

        When ``use_kv_cache`` is False, each step retains the original full
        ``[video_cond | state | action]`` packed forward.

        Returns:
            (B, T_action, max_action_dim) denoised action latents.
        """
        device = self.device
        dtype = torch.bfloat16
        n_cameras = self.max_n_cameras
        B = cond_latent.shape[0]

        # Scheduler
        act_sched = BetaFlowMatchEulerScheduler(self.action_scheduler.config)
        act_sched.set_timesteps(num_steps, device=device)

        # Initialize action noise
        ac = self.config.action_cfg
        T_action = ac.get("action_horizon", 48)
        max_action_dim = ac.get("max_action_dim", 32)
        # Local Generator, not torch.manual_seed(): this runs on every rank during
        # in-process open-loop eval and must not reset the global RNG, which would
        # undo the per-rank offset in experiment.run. Identical draws either way.
        noise_gen = torch.Generator(device=device).manual_seed(seed)
        action_latents = torch.randn(
            B, T_action, max_action_dim, device=device, dtype=dtype, generator=noise_gen
        )

        # Encode state once (constant across denoising steps)
        states = data["state"].to(device=device, dtype=dtype)
        embodiment_id = data.get("embodiment_id", torch.zeros(B, dtype=torch.long)).to(device)
        state_tokens = self.state_encoder(states, embodiment_id)
        state_tokens = apply_state_keep(state_tokens, data)
        state_rope_freqs = self.state_rope(state_tokens)
        state_token_sigmas = torch.full(
            (B, state_tokens.shape[1]),
            self.cond_timestep,
            device=device,
            dtype=dtype,
        )

        # Patchify conditioning frames once (constant across denoising steps)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            cond_mask = torch.ones(B, 1, *cond_latent.shape[2:], device=device, dtype=dtype)
            video_tokens, video_extra_pos_emb, cond_shape_info = self.video_tokenizer.patchify(
                cond_latent, cond_mask, None, n_cameras
            )
            L_video = video_tokens.shape[1]
            video_token_sigmas = torch.full(
                (B, L_video),
                self.cond_timestep,
                device=device,
                dtype=dtype,
            )
            video_rope_freqs = self.video_rope(cond_latent, n_cameras=n_cameras)
            video_cond_token_mask = torch.ones(B, L_video, dtype=torch.bool, device=device)

        # token_valid_mask matches the training-time mask so real-view query
        # tokens cannot attend to padded-view KVs. See
        # `_build_inference_token_valid_mask` for the why.
        T_cond_per_view = cond_latent.shape[2] // n_cameras
        token_valid_mask = self._build_inference_token_valid_mask(
            data.get("video_padded_mask"),
            B=B,
            n_cameras=n_cameras,
            T_per_view=T_cond_per_view,
            tokens_per_frame=cond_shape_info["post_H"] * cond_shape_info["post_W"],
            n_extra_tokens=state_tokens.shape[1] + T_action,
            device=device,
        )

        static_inputs = {
            "shape_info": cond_shape_info,
            "video_tokens": video_tokens,
            "video_token_sigmas": video_token_sigmas,
            "video_rope_freqs": video_rope_freqs,
            "video_extra_pos_emb": video_extra_pos_emb,
            "video_cond_token_mask": video_cond_token_mask,
            "state_tokens": state_tokens,
            "state_token_sigmas": state_token_sigmas,
            "state_rope_freqs": state_rope_freqs,
            "token_valid_mask": token_valid_mask,
        }

        needs_neg_cache = neg_prompt_embs is not None and guidance_scale != 1.0
        pos_cache = neg_cache = None
        if use_kv_cache:
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                pos_cache = self._precompute_cond_kv_cache(static_inputs, prompt_embs)
                if needs_neg_cache:
                    neg_cache = self._precompute_cond_kv_cache(static_inputs, neg_prompt_embs)

        # Denoising loop
        for i in range(len(act_sched.timesteps)):
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                action_t = act_sched.timesteps[i].to(dtype)
                action_sigma = act_sched.sigmas[i].to(dtype)
                # expand+clone instead of torch.full(fill_value=tensor): the
                # latter syncs via .item(), which is illegal under CUDA graph
                # capture.
                action_t_per_step = action_t.expand(B, T_action).clone()
                action_sigma_per_step = action_sigma.expand(B, T_action).clone()
                action_tokens = self.action_encoder(
                    action_latents, action_t_per_step, embodiment_id
                )
                action_rope_freqs = self.action_rope(action_tokens)

                action_noise_pred = self._run_action_step(
                    static_inputs=static_inputs,
                    action_tokens=action_tokens,
                    action_sigmas=action_sigma_per_step,
                    action_rope_freqs=action_rope_freqs,
                    prompt_embs=prompt_embs,
                    embodiment_id=embodiment_id,
                    kv_cache=pos_cache,
                    denoising_step=i,
                )

            # CFG for action
            if needs_neg_cache:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    action_noise_pred_neg = self._run_action_step(
                        static_inputs=static_inputs,
                        action_tokens=action_tokens,
                        action_sigmas=action_sigma_per_step,
                        action_rope_freqs=action_rope_freqs,
                        prompt_embs=neg_prompt_embs,
                        embodiment_id=embodiment_id,
                        kv_cache=neg_cache,
                        denoising_step=i,
                        is_cfg=True,
                    )
                action_noise_pred = action_noise_pred_neg + guidance_scale * (
                    action_noise_pred - action_noise_pred_neg
                )

            action_latents = act_sched.step(
                action_noise_pred, act_sched.timesteps[i], action_latents, step_index=i
            )[0]

        return action_latents

    def _resolve_attention_mask_pattern(self) -> AttentionMaskPattern:
        """Return the currently configured single attention mask pattern.

        The training config allows a stochastic mix of patterns (list of
        (name, prob) pairs). Inference uses a single deterministic pattern —
        pick the one with probability 1, or fall back to the default used by
        ``_build_attention_mask`` when none is configured.
        """
        raw = getattr(self, "attention_mask_pattern", None)
        if raw is None:
            return AttentionMaskPattern.BIDIRECTIONAL_VIDEO_ACTION_COND
        if isinstance(raw, str):
            return AttentionMaskPattern(raw)
        if isinstance(raw, AttentionMaskPattern):
            return raw
        # list of (name, prob) — pick the highest-probability entry
        name, _ = max(raw, key=lambda pair: pair[1])
        return AttentionMaskPattern(name)

    def _precompute_cond_kv_cache(
        self, static_inputs: dict, prompt_embs: torch.Tensor
    ) -> CondKVCache:
        """Capture per-layer self-attention K/V for a static token prefix.

        The cache is used by both code paths that splice a static prefix into
        action-only denoising steps:

        - Disjoint action denoising: ``static_inputs`` carries
          ``[video_cond, state]`` (video_cond_token_mask is all-True).
        - Joint→cached hybrid: ``static_inputs`` carries the post-video-phase
          blended video (cond + denoised denoise) and state. The caller only
          includes v_denoise when the configured attention pattern lets action
          attend to it — otherwise the v_denoise K/V would be wrongly attended
          to by the unmasked action step.

        The configured ``attention_mask_pattern`` must keep video+state
        self-contained (every pattern except ``FULL``); otherwise captured K/V
        depend on tokens not present in ``static_inputs`` and the cache is
        invalid.

        When the transformer is wrapped in ``torch.compile`` with CUDA Graphs
        (``mode="reduce-overhead"``), the tensors captured inside the compiled
        region alias static CUDA-Graph output buffers that get overwritten on
        the next invocation. We clone them here — outside the compiled region
        — so the cache holds ordinary tensors safe to reuse.
        """
        # token_valid_mask: if the caller pre-computed one in static_inputs
        # (e.g. via `_build_inference_token_valid_mask`), slice it to drop the
        # action portion since this prefix never includes action tokens.
        tvm = static_inputs.get("token_valid_mask")
        if tvm is not None:
            L_static = (
                static_inputs["video_tokens"].shape[1] + static_inputs["state_tokens"].shape[1]
            )
            tvm = tvm[:, :L_static]
        packed = self.packer.pack(
            video_tokens=static_inputs["video_tokens"],
            video_timesteps=static_inputs["video_token_sigmas"],
            video_rope=static_inputs["video_rope_freqs"],
            video_extra_pos_emb=static_inputs["video_extra_pos_emb"],
            video_cond_token_mask=static_inputs["video_cond_token_mask"],
            state_tokens=static_inputs["state_tokens"],
            state_timesteps=static_inputs["state_token_sigmas"],
            state_rope=static_inputs["state_rope_freqs"],
            attention_mask_pattern=getattr(self, "attention_mask_pattern", None),
            token_valid_mask=tvm,
        )
        token_merge = self._build_token_merge_plan(packed, static_inputs.get("shape_info"), tvm)
        cache = CondKVCache()
        tm_kwargs = {"sequence_transform": token_merge} if token_merge is not None else {}
        self._call_transformer(packed, prompt_embs, kv_cache=cache, **tm_kwargs)

        # Drop padded-view K/V from the cache. The cached fast path in
        # `_run_action_step` consumes this cache with `attention_mask=None`
        # + flash_attn, so unmasked action queries would otherwise attend to
        # padded-view K/V. The training-time mask only gates attention
        # outputs; the underlying K/V projections (and therefore the cache)
        # are populated regardless of validity. Filter them out here so the
        # downstream consumer sees only real positions.
        if tvm is not None and not bool(tvm.all()):
            assert torch.all(
                tvm == tvm[:1]
            ), "kv_cache_cond assumes per-view padding is uniform across the batch"
            keep_fine = tvm[0].nonzero(as_tuple=True)[0]
            keep_coarse = None
            if token_merge is not None:
                keep_fine, keep_coarse = token_merge.keep_indices_per_segment(tvm[0])
            for layer_idx in range(len(cache.key_cache)):
                # Mid layers captured the coarse prefix; keep indices follow.
                keep_idx = (
                    keep_coarse
                    if token_merge is not None and token_merge.is_reduced_layer(layer_idx)
                    else keep_fine
                )
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx].index_select(2, keep_idx)
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx].index_select(
                    2, keep_idx
                )

        cache.key_cache = [k.clone() for k in cache.key_cache]
        cache.value_cache = [v.clone() for v in cache.value_cache]
        return cache

    def _run_action_step(
        self,
        *,
        static_inputs: dict,
        action_tokens: torch.Tensor,
        action_sigmas: torch.Tensor,
        action_rope_freqs: tuple[torch.Tensor, torch.Tensor],
        prompt_embs: torch.Tensor,
        embodiment_id: torch.Tensor,
        kv_cache: CondKVCache | None,
        denoising_step: int = 0,
        is_cfg: bool = False,
    ) -> torch.Tensor:
        """One transformer forward producing the action noise prediction.

        When ``kv_cache`` is provided, skips the static prefix entirely and
        feeds only action tokens through the transformer, splicing the cached
        static K/V into each block's self-attention. Otherwise runs the full
        packed sequence as before.
        """
        B = action_tokens.shape[0]
        device = action_tokens.device
        if kv_cache is None:
            packed = self.packer.pack(
                video_tokens=static_inputs["video_tokens"],
                video_timesteps=static_inputs["video_token_sigmas"],
                video_rope=static_inputs["video_rope_freqs"],
                video_extra_pos_emb=static_inputs["video_extra_pos_emb"],
                video_cond_token_mask=static_inputs["video_cond_token_mask"],
                action_tokens=action_tokens,
                action_timesteps=action_sigmas,
                action_rope=action_rope_freqs,
                state_tokens=static_inputs["state_tokens"],
                state_timesteps=static_inputs["state_token_sigmas"],
                state_rope=static_inputs["state_rope_freqs"],
                attention_mask_pattern=getattr(self, "attention_mask_pattern", None),
                token_valid_mask=static_inputs.get("token_valid_mask"),
            )
            tm_kwargs = {}
            token_merge = self._build_token_merge_plan(
                packed, static_inputs.get("shape_info"), static_inputs.get("token_valid_mask")
            )
            if token_merge is not None:
                tm_kwargs["sequence_transform"] = token_merge
            out_tokens = self._call_transformer(
                packed,
                prompt_embs,
                denoising_step=denoising_step,
                is_cfg=is_cfg,
                **tm_kwargs,
            )
            action_out = unpack_contiguous(out_tokens, packed, MODALITY_ACTION)
        else:
            # Cached fast path: action-only forward, no BlockMask (every action
            # query attends to all static + action keys, and flash_attn handles
            # Lq < Lk natively).
            L_action = action_tokens.shape[1]
            action_modality_ids = torch.full(
                (B, L_action), MODALITY_ACTION, dtype=torch.long, device=device
            )
            packed_action = PackedSequence(
                tokens=action_tokens,
                timesteps=action_sigmas,
                rope_freqs=action_rope_freqs,
                modality_ids=action_modality_ids,
                attention_mask=None,
                extra_pos_emb=None,
                pad=0,
                video_token_indices=torch.arange(0, device=device),
                state_token_indices=torch.arange(0, device=device),
                action_token_indices=torch.arange(L_action, device=device),
                non_video_token_indices=torch.arange(L_action, device=device),
            )
            action_out = self._call_transformer(
                packed_action,
                prompt_embs,
                kv_cache=kv_cache,
                store_kv=False,
                denoising_step=denoising_step,
                is_cfg=is_cfg,
            )
        return self.action_decoder(action_out, embodiment_id)

    def _denoise_video(
        self,
        cond_latent: torch.Tensor,
        prompt_embs: torch.Tensor,
        neg_prompt_embs: torch.Tensor | None,
        num_gen_latent: int,
        num_steps: int,
        guidance_scale: float,
        shift: float,
        seed: int,
        video_padded_mask=None,
    ) -> torch.Tensor:
        """Denoise video latents using conditioning frames.

        Runs an independent video diffusion loop with optional CFG. No
        action/state tokens are packed — video denoising is self-contained.

        Returns:
            (B, C, T_full, H, W) denoised video latents.
        """
        device = self.device
        dtype = torch.bfloat16
        n_cameras = self.max_n_cameras

        B, C, T_cond_total, H, W = cond_latent.shape
        assert T_cond_total % n_cameras == 0
        T_cond_per_view = T_cond_total // n_cameras
        T_per_view = T_cond_per_view + num_gen_latent
        T_full = n_cameras * T_per_view

        # Scheduler
        vid_sched = self._make_video_scheduler(self.config)
        vid_sched.set_timesteps(num_steps, device=device, shift=shift)

        # Initialize video noise
        # Local Generator, not torch.manual_seed(): this runs on every rank during
        # in-process open-loop eval and must not reset the global RNG, which would
        # undo the per-rank offset in experiment.run. Identical draws either way.
        noise_gen = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(B, C, T_full, H, W, device=device, dtype=dtype, generator=noise_gen)

        # Build conditioning masks
        cond_mask = torch.zeros(B, 1, T_full, H, W, device=device, dtype=dtype)
        cond_indicator = torch.zeros(B, 1, T_full, 1, 1, device=device, dtype=dtype)
        cond_latent_full = torch.zeros_like(latents)

        for v in range(n_cameras):
            src_start = v * T_cond_per_view
            dst_start = v * T_per_view
            cond_mask[:, :, dst_start : dst_start + T_cond_per_view] = 1.0
            cond_indicator[:, :, dst_start : dst_start + T_cond_per_view] = 1.0
            cond_latent_full[:, :, dst_start : dst_start + T_cond_per_view] = cond_latent[
                :, :, src_start : src_start + T_cond_per_view
            ]

        gt_velocity = (latents - cond_latent_full) * cond_mask
        # torch.full instead of torch.tensor: graph-safe (no CPU->GPU scalar copy).
        cond_timestep_val = torch.full((), self.cond_timestep, device=device, dtype=dtype)

        # Denoising loop
        for i, t in enumerate(vid_sched.timesteps):
            video_sigma = vid_sched.sigmas[i].to(dtype)

            in_latents = cond_mask * cond_latent_full + (1 - cond_mask) * latents
            in_timestep = cond_indicator * cond_timestep_val + (1 - cond_indicator) * video_sigma

            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                video_tokens, video_extra_pos_emb, shape_info = self.video_tokenizer.patchify(
                    in_latents, cond_mask, None, n_cameras
                )
                L = video_tokens.shape[1]
                video_token_sigmas = self._expand_timesteps(in_timestep, B, T_full, shape_info, L)
                video_rope_freqs = self.video_rope(in_latents, n_cameras=n_cameras)

                # Video-only packing (no action/state → no BlockMask).
                # token_valid_mask still matters: real-view query tokens must
                # not attend to padded-view KVs at inference.
                token_valid_mask = self._build_inference_token_valid_mask(
                    video_padded_mask,
                    B=B,
                    n_cameras=n_cameras,
                    T_per_view=T_per_view,
                    tokens_per_frame=shape_info["post_H"] * shape_info["post_W"],
                    n_extra_tokens=0,
                    device=device,
                )
                packed = self.packer.pack(
                    video_tokens=video_tokens,
                    video_timesteps=video_token_sigmas,
                    video_rope=video_rope_freqs,
                    video_extra_pos_emb=video_extra_pos_emb,
                    token_valid_mask=token_valid_mask,
                )
                tm_kwargs = {}
                token_merge = self._build_token_merge_plan(packed, shape_info, token_valid_mask)
                if token_merge is not None:
                    tm_kwargs["sequence_transform"] = token_merge
                out_tokens = self._call_transformer(
                    packed,
                    prompt_embs,
                    denoising_step=i,
                    **tm_kwargs,
                )
                video_out = unpack_contiguous(
                    out_tokens, packed, (MODALITY_VIDEO, MODALITY_VIDEO_COND)
                )
                video_noise_pred = self.video_tokenizer.unpatchify(video_out, shape_info)

            video_noise_pred = gt_velocity + video_noise_pred * (1 - cond_mask)

            # CFG
            if neg_prompt_embs is not None and guidance_scale != 1.0:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    out_tokens_neg = self._call_transformer(
                        packed,
                        neg_prompt_embs,
                        denoising_step=i,
                        is_cfg=True,
                        **tm_kwargs,
                    )
                    video_out_neg = unpack_contiguous(
                        out_tokens_neg, packed, (MODALITY_VIDEO, MODALITY_VIDEO_COND)
                    )
                    video_noise_pred_neg = self.video_tokenizer.unpatchify(
                        video_out_neg, shape_info
                    )
                video_noise_pred_neg = gt_velocity + video_noise_pred_neg * (1 - cond_mask)
                video_noise_pred = video_noise_pred_neg + guidance_scale * (
                    video_noise_pred - video_noise_pred_neg
                )

            latents = vid_sched.step(video_noise_pred, t, latents, step_index=i)[0]

        return latents

    def _denoise_joint(
        self,
        data: dict,
        cond_latent: torch.Tensor,
        prompt_embs: torch.Tensor,
        neg_prompt_embs: torch.Tensor | None,
        num_gen_latent: int,
        num_steps: int,
        video_guidance_scale: float,
        action_guidance_scale: float,
        shift: float,
        seed: int,
        num_action_steps: int | None = None,
        use_kv_cache: bool = False,
        step_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint video-action denoising.

        Symmetric legacy mode (``num_action_steps is None`` or equal to
        ``num_steps``, or ``use_kv_cache=False``): one joint forward per step,
        video + action advance together for ``num_steps`` iterations.

        Interleaved cached mode (``use_kv_cache=True`` with
        ``num_action_steps > num_steps``): ``num_action_steps`` must be an
        integer multiple of ``num_steps``. Let ``k = num_action_steps //
        num_steps``. The loop runs ``num_steps`` outer iterations; each
        iteration does one joint forward (capturing per-layer self-attention
        K/V), steps video once and action once using that forward's outputs,
        then runs ``k - 1`` cached action-only forwards that reuse the
        captured K/V for the static (video + state) prefix. Total transformer
        forwards: ``num_steps`` full joint + ``num_action_steps - num_steps``
        action-only, saving one full video-processing forward vs a separate
        precompute pass.

        Returns:
            (video_latents, action_pred): video latents (B, C, T_full, H, W)
            and denoised actions (B, T_action, max_action_dim).
        """
        device = self.device
        dtype = torch.bfloat16
        n_cameras = self.max_n_cameras
        ac = self.config.action_cfg

        num_video_steps = num_steps
        if num_action_steps is None or not use_kv_cache:
            num_action_steps = num_video_steps
        if use_kv_cache and num_action_steps % num_video_steps != 0:
            raise ValueError(
                f"num_action_steps ({num_action_steps}) must be an integer "
                f"multiple of num_video_steps ({num_video_steps}) when "
                f"use_kv_cache is set"
            )
        k = num_action_steps // num_video_steps

        B, C, T_cond_total, H, W = cond_latent.shape
        assert T_cond_total % n_cameras == 0
        T_cond_per_view = T_cond_total // n_cameras
        T_per_view = T_cond_per_view + num_gen_latent
        T_full = n_cameras * T_per_view

        # ── Setup schedulers ───────────────────────────────────────────
        vid_sched = self._make_video_scheduler(self.config)
        vid_sched.set_timesteps(num_video_steps, device=device, shift=shift)

        act_sched = BetaFlowMatchEulerScheduler(self.action_scheduler.config)
        act_sched.set_timesteps(num_action_steps, device=device)

        # ── Initialize noise ──────────────────────────────────────────
        # Local Generator, not torch.manual_seed(): this runs on every rank during
        # in-process open-loop eval and must not reset the global RNG, which would
        # undo the per-rank offset in experiment.run. Identical draws either way.
        noise_gen = torch.Generator(device=device).manual_seed(seed)
        video_latents = torch.randn(
            B, C, T_full, H, W, device=device, dtype=dtype, generator=noise_gen
        )

        T_action = ac.get("action_horizon", 48)
        max_action_dim = ac.get("max_action_dim", 32)
        action_latents = torch.randn(
            B, T_action, max_action_dim, device=device, dtype=dtype, generator=noise_gen
        )

        # ── Encode state (constant across steps) ─────────────────────
        states = data["state"].to(device=device, dtype=dtype)
        embodiment_id = data.get("embodiment_id", torch.zeros(B, dtype=torch.long)).to(device)
        state_tokens = self.state_encoder(states, embodiment_id)
        state_tokens = apply_state_keep(state_tokens, data)
        state_rope_freqs = self.state_rope(state_tokens)
        state_token_sigmas = torch.full(
            (B, state_tokens.shape[1]),
            self.cond_timestep,
            device=device,
            dtype=dtype,
        )

        # ── Video conditioning setup ─────────────────────────────────
        cond_mask = torch.zeros(B, 1, T_full, H, W, device=device, dtype=dtype)
        cond_indicator = torch.zeros(B, 1, T_full, 1, 1, device=device, dtype=dtype)
        cond_latent_full = torch.zeros_like(video_latents)

        for v in range(n_cameras):
            src_start = v * T_cond_per_view
            dst_start = v * T_per_view
            cond_mask[:, :, dst_start : dst_start + T_cond_per_view] = 1.0
            cond_indicator[:, :, dst_start : dst_start + T_cond_per_view] = 1.0
            cond_latent_full[:, :, dst_start : dst_start + T_cond_per_view] = cond_latent[
                :, :, src_start : src_start + T_cond_per_view
            ]

        gt_velocity = (video_latents - cond_latent_full) * cond_mask
        # torch.full instead of torch.tensor: graph-safe (no CPU->GPU scalar copy).
        cond_timestep_val = torch.full((), self.cond_timestep, device=device, dtype=dtype)

        # ── Interleaved joint denoising loop ─────────────────────────
        # Each outer iteration: one joint forward (capturing K/V when caching
        # is on), step video + action once, then k-1 cached action sub-steps.
        act_step = 0
        action_cfg_on = neg_prompt_embs is not None and action_guidance_scale != 1.0
        video_cfg_on = neg_prompt_embs is not None and video_guidance_scale != 1.0
        cfg_on = action_cfg_on or video_cfg_on

        for i in range(num_video_steps):
            video_sigma = vid_sched.sigmas[i].to(dtype)
            video_t = vid_sched.timesteps[i]

            action_t = act_sched.timesteps[act_step].to(dtype)
            action_sigma = act_sched.sigmas[act_step].to(dtype)

            # Prepare video input
            in_latents = cond_mask * cond_latent_full + (1 - cond_mask) * video_latents
            in_timestep = cond_indicator * cond_timestep_val + (1 - cond_indicator) * video_sigma

            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                # Patchify video
                video_tokens, video_extra_pos_emb, shape_info = self.video_tokenizer.patchify(
                    in_latents, cond_mask, None, n_cameras
                )
                L = video_tokens.shape[1]
                video_token_sigmas = self._expand_timesteps(in_timestep, B, T_full, shape_info, L)
                video_rope_freqs = self.video_rope(in_latents, n_cameras=n_cameras)
                video_cond_token_mask = video_token_sigmas == self.cond_timestep

                # Encode action at current noise level
                # expand+clone instead of torch.full(fill_value=tensor): the
                # latter syncs via .item(), which is illegal under CUDA graph
                # capture.
                action_t_per_step = action_t.expand(B, T_action).clone()
                action_sigma_per_step = action_sigma.expand(B, T_action).clone()
                action_tokens = self.action_encoder(
                    action_latents, action_t_per_step, embodiment_id
                )
                action_rope_freqs = self.action_rope(action_tokens)

                # token_valid_mask: see `_build_inference_token_valid_mask`.
                # Built every iter because shape_info/T_per_view are loop-local;
                # the underlying view_is_real flag is data-dependent but constant
                # across iterations, so this is cheap.
                token_valid_mask = self._build_inference_token_valid_mask(
                    data.get("video_padded_mask"),
                    B=B,
                    n_cameras=n_cameras,
                    T_per_view=T_per_view,
                    tokens_per_frame=shape_info["post_H"] * shape_info["post_W"],
                    n_extra_tokens=state_tokens.shape[1] + T_action,
                    device=device,
                )

                # Pack all modalities together
                packed = self.packer.pack(
                    video_tokens=video_tokens,
                    video_timesteps=video_token_sigmas,
                    video_rope=video_rope_freqs,
                    video_extra_pos_emb=video_extra_pos_emb,
                    video_cond_token_mask=video_cond_token_mask,
                    action_tokens=action_tokens,
                    action_timesteps=action_sigma_per_step,
                    action_rope=action_rope_freqs,
                    state_tokens=state_tokens,
                    state_timesteps=state_token_sigmas,
                    state_rope=state_rope_freqs,
                    attention_mask_pattern=getattr(self, "attention_mask_pattern", None),
                    token_valid_mask=token_valid_mask,
                    **self._joint_inference_packing_kwargs(
                        data=data,
                        video_tokens=video_tokens,
                        video_timesteps=video_token_sigmas,
                        action_tokens=action_tokens,
                        denoising_step=i,
                        total_denoising_steps=num_video_steps,
                    ),
                )

                tm_kwargs = {}
                token_merge = None
                if getattr(self.config, "token_merge_kernel", None) is not None:
                    token_merge = self._build_token_merge_plan(packed, shape_info, token_valid_mask)
                    tm_kwargs["sequence_transform"] = token_merge

                # Joint forward (captures K/V when we have cached sub-steps to
                # run this iteration).
                pos_cache = CondKVCache() if (use_kv_cache and k > 1) else None
                out_tokens = self._call_transformer(
                    packed,
                    prompt_embs,
                    denoising_step=i,
                    kv_cache=pos_cache,
                    **tm_kwargs,
                )

                # Unpack predictions
                video_out = unpack_contiguous(
                    out_tokens, packed, (MODALITY_VIDEO, MODALITY_VIDEO_COND)
                )
                video_noise_pred = self.video_tokenizer.unpatchify(video_out, shape_info)

                action_out = unpack_contiguous(out_tokens, packed, MODALITY_ACTION)
                action_noise_pred = self.action_decoder(action_out, embodiment_id)

            # Apply conditioning mask to video prediction
            video_noise_pred = gt_velocity + video_noise_pred * (1 - cond_mask)

            # CFG for both video and action
            neg_cache = None
            if cfg_on:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    neg_cache = CondKVCache() if (use_kv_cache and k > 1) else None
                    out_tokens_neg = self._call_transformer(
                        packed,
                        neg_prompt_embs,
                        denoising_step=i,
                        is_cfg=True,
                        kv_cache=neg_cache,
                        **tm_kwargs,
                    )
                    video_out_neg = unpack_contiguous(
                        out_tokens_neg,
                        packed,
                        (MODALITY_VIDEO, MODALITY_VIDEO_COND),
                    )
                    video_noise_pred_neg = self.video_tokenizer.unpatchify(
                        video_out_neg, shape_info
                    )
                    action_out_neg = unpack_contiguous(out_tokens_neg, packed, MODALITY_ACTION)
                    action_noise_pred_neg = self.action_decoder(action_out_neg, embodiment_id)

                video_noise_pred_neg = gt_velocity + video_noise_pred_neg * (1 - cond_mask)
                video_noise_pred = video_noise_pred_neg + video_guidance_scale * (
                    video_noise_pred - video_noise_pred_neg
                )
                action_noise_pred = action_noise_pred_neg + action_guidance_scale * (
                    action_noise_pred - action_noise_pred_neg
                )

            # Step both schedulers (one video step, one action step)
            video_latents = vid_sched.step(video_noise_pred, video_t, video_latents, step_index=i)[
                0
            ]
            action_latents = act_sched.step(
                action_noise_pred,
                act_sched.timesteps[act_step],
                action_latents,
                step_index=act_step,
            )[0]
            act_step += 1

            # k-1 cached action sub-steps — action-only forwards against the
            # captured static-prefix K/V. Drops every key action doesn't
            # attend to per the configured attention pattern (always action
            # itself; often v_denoise too).
            if pos_cache is None:
                continue
            # Conditional kwarg — see _build_token_merge_plan's docstring.
            tm_slice_kwargs = {"token_merge": token_merge} if token_merge is not None else {}
            pos_cache = self._slice_cache_to_static_prefix(
                pos_cache, packed.modality_ids, token_valid_mask=token_valid_mask, **tm_slice_kwargs
            )
            if neg_cache is not None:
                neg_cache = self._slice_cache_to_static_prefix(
                    neg_cache,
                    packed.modality_ids,
                    token_valid_mask=token_valid_mask,
                    **tm_slice_kwargs,
                )

            for _ in range(1, k):
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    action_t = act_sched.timesteps[act_step].to(dtype)
                    action_sigma = act_sched.sigmas[act_step].to(dtype)
                    # expand+clone instead of torch.full(fill_value=tensor):
                    # the latter syncs via .item(), which is illegal under
                    # CUDA graph capture.
                    action_t_per_step = action_t.expand(B, T_action).clone()
                    action_sigma_per_step = action_sigma.expand(B, T_action).clone()
                    action_tokens = self.action_encoder(
                        action_latents, action_t_per_step, embodiment_id
                    )
                    action_rope_freqs = self.action_rope(action_tokens)

                    action_noise_pred = self._run_action_step(
                        static_inputs={},
                        action_tokens=action_tokens,
                        action_sigmas=action_sigma_per_step,
                        action_rope_freqs=action_rope_freqs,
                        prompt_embs=prompt_embs,
                        embodiment_id=embodiment_id,
                        kv_cache=pos_cache,
                        denoising_step=act_step,
                    )

                if neg_cache is not None and action_cfg_on:
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        action_noise_pred_neg = self._run_action_step(
                            static_inputs={},
                            action_tokens=action_tokens,
                            action_sigmas=action_sigma_per_step,
                            action_rope_freqs=action_rope_freqs,
                            prompt_embs=neg_prompt_embs,
                            embodiment_id=embodiment_id,
                            kv_cache=neg_cache,
                            denoising_step=act_step,
                            is_cfg=True,
                        )
                    action_noise_pred = action_noise_pred_neg + action_guidance_scale * (
                        action_noise_pred - action_noise_pred_neg
                    )

                action_latents = act_sched.step(
                    action_noise_pred,
                    act_sched.timesteps[act_step],
                    action_latents,
                    step_index=act_step,
                )[0]
                act_step += 1

            if step_callback is not None:
                step_callback(i + 1, num_video_steps)

        return video_latents, action_latents

    def _slice_cache_to_static_prefix(
        self,
        cache: CondKVCache,
        modality_ids: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
        token_merge: TokenMergePlan | None = None,
    ) -> CondKVCache:
        """Slice a joint-forward cache to positions attended to by action.

        The joint forward's ``kv_cache`` stores per-layer K/V for the
        full packed sequence ``[video, state, action]``. Later cached
        action-only forwards use unmasked flash attention, so action attends
        to every cached key — the cache must therefore exclude any modality
        action doesn't attend to per the pattern. Also enforces that video
        tokens are self-contained under the pattern (otherwise the captured
        K/V for video depends on tokens not in the cache and reuse would be
        unsafe). Cloning detaches the slice from CUDA-Graph static output
        buffers.

        When ``token_valid_mask`` is provided, padded-token positions are
        also dropped (e.g. padded views for ckpts where max_n_cameras >
        actual views). Without this, the cached fast path's unmasked flash
        attention would let action queries attend to padded-view K/V.
        Assumes per-view padding is uniform across the batch dim.
        """
        pattern = self._resolve_attention_mask_pattern()
        table = pattern.to_table()
        # Video rows (0, 1) must not attend to state (col 2) or action (col 3).
        if table[0][2] or table[0][3] or table[1][2] or table[1][3]:
            raise ValueError(
                f"KV cache requires a pattern where video is self-contained; "
                f"{pattern.value!r} has video attending to state/action."
            )
        # Action row (col order: [v_cond, v_denoise, state, action]) tells us
        # which modality IDs should remain in the cache.
        action_row = table[3]
        keep_modalities: list[int] = []
        if action_row[0]:
            keep_modalities.append(MODALITY_VIDEO_COND)
        if action_row[1]:
            keep_modalities.append(MODALITY_VIDEO)
        if action_row[2]:
            keep_modalities.append(MODALITY_STATE)

        ids = modality_ids[0]
        keep = torch.zeros_like(ids, dtype=torch.bool)
        for m in keep_modalities:
            keep |= ids == m
        if token_valid_mask is not None:
            assert torch.all(
                token_valid_mask == token_valid_mask[:1]
            ), "kv_cache_cond assumes per-view padding is uniform across the batch"
            # Project-local inference hooks may append auxiliary tokens after
            # the ordinary packed prefix.  Those tokens are excluded above by
            # modality, while this mask describes only the original prefix.
            # Apply it to that prefix without assuming both lengths match.
            if token_valid_mask.shape[1] > keep.shape[0]:
                raise ValueError(
                    "token_valid_mask is longer than the packed inference sequence: "
                    f"{token_valid_mask.shape[1]} > {keep.shape[0]}"
                )
            keep[: token_valid_mask.shape[1]] &= token_valid_mask[0]
        fine_idx = keep.nonzero(as_tuple=True)[0]
        # With token_merge, the mid layers captured the coarse prefix; select the
        # keep index per layer segment.
        coarse_idx = None
        if token_merge is not None:
            fine_idx, coarse_idx = token_merge.keep_indices_per_segment(keep)
        sliced = CondKVCache()
        for layer_idx, (k_layer, v_layer) in enumerate(
            zip(cache.key_cache, cache.value_cache, strict=False)
        ):
            keep_idx = (
                coarse_idx
                if token_merge is not None and token_merge.is_reduced_layer(layer_idx)
                else fine_idx
            )
            sliced.key_cache.append(k_layer.index_select(2, keep_idx).clone())
            sliced.value_cache.append(v_layer.index_select(2, keep_idx).clone())
        return sliced

    @torch.no_grad()
    def predict(
        self,
        inputs: dict | BatchFeature,
        *,
        num_video_steps: int = 36,
        num_action_steps: int = 4,
        video_guidance_scale: float = 1.0,
        action_guidance_scale: float = 1.0,
        shift: float = 1.0,
        seed: int = 42,
        predict_video: bool = False,
        joint_denoise: bool = True,
        prompt_embs: torch.Tensor | None = None,
        kv_cache_cond: bool = False,
        step_callback=None,
    ) -> dict:
        """Predict actions and optionally video via diffusion denoising.

        Args:
            inputs: Collated processor output.
            num_video_steps: Denoising steps for video / joint denoising.
            num_action_steps: Denoising steps for action (disjoint mode only).
            video_guidance_scale: CFG scale for video.
            action_guidance_scale: CFG scale for action.
            shift: Scheduler sigma shift for video.
            seed: Random seed.
            predict_video: Also return video latents.
            joint_denoise: Explicitly select joint (True) or disjoint (False)
                denoising. When None (default), auto-selects based on whether
                the attention_mask_pattern lets action attend to video_denoise.
            prompt_embs: Precomputed text embeddings. When provided, skips the
                text encoder. Always echoed back in the result dict so callers
                can cache across steps with constant language.
            kv_cache_cond: Speed optimization for action denoising. In disjoint
                mode, precomputes per-layer self-attention K/V for the static
                ``[video_cond, state]`` prefix once and reuses across all
                ``num_action_steps`` steps. In joint mode with
                ``num_action_steps > num_video_steps`` (must be a multiple),
                each outer iteration captures K/V during the joint forward and
                runs ``k - 1`` cached action sub-steps before the next video
                update, where ``k = num_action_steps // num_video_steps``.
                Unsupported with the ``FULL`` attention pattern (video not
                self-contained).
            step_callback: Optional callback invoked after each completed joint
                video denoising step with ``(completed_steps, total_steps)``.

        Returns:
            dict with:
                "action_pred": (B, T_action, max_action_dim) — when action_cfg set
                "video_latents": (B, C, T_full, H, W) — when predict_video=True
                "prompt_embs": text embeddings used for this call
        """
        data = inputs if isinstance(inputs, dict) else inputs.data
        if "inputs" in data:
            data = data["inputs"]

        device = self.device

        # ── Shared encoding ──────────────────────────────────────────────
        images = data["images"].to(device)
        videos = self._preprocess_video(images)
        latents = self.encode_video(videos)

        if prompt_embs is None:
            text_ids = data["text"].to(device)
            prompt_embs = self.encode_text(text_ids)

        num_cond_pixel = data.get("num_cond_frames", torch.zeros(1, dtype=torch.long))
        if isinstance(num_cond_pixel, torch.Tensor):
            assert torch.all(
                num_cond_pixel == num_cond_pixel[0]
            ), f"num_cond_frames must be uniform across batch, got {num_cond_pixel}"
            num_cond_pixel = num_cond_pixel[0].item()
        num_cond_latent = self._pixel_to_latent_frames(int(num_cond_pixel))
        cond_latent = self._extract_cond_latents(latents, num_cond_latent)

        result = {"prompt_embs": prompt_embs}
        has_action = self.has_action_head and "state" in data

        # ── Shared: num_gen_latent + negative prompt ─────────────────────
        neg_prompt_embs = None
        if "text_negative" in data:
            neg_prompt_embs = self.encode_text(data["text_negative"].to(device))

        num_total_pixel = data.get("num_total_frames", None)
        if num_total_pixel is not None:
            if isinstance(num_total_pixel, torch.Tensor):
                assert torch.all(
                    num_total_pixel == num_total_pixel[0]
                ), f"num_total_frames must be uniform across batch, got {num_total_pixel}"
                num_total_pixel = num_total_pixel[0].item()
            num_total_latent = self._pixel_to_latent_frames(int(num_total_pixel))
        else:
            num_total_latent = latents.shape[2] // self.max_n_cameras
        num_gen_latent = num_total_latent - num_cond_latent

        # ── Resolve joint vs disjoint denoising ────────────────────────
        if joint_denoise and has_action:
            assert num_gen_latent > 0, f"No frames to generate: num_gen_latent={num_gen_latent}"
            video_latents, action_pred = self._denoise_joint(
                data=data,
                cond_latent=cond_latent,
                prompt_embs=prompt_embs,
                neg_prompt_embs=neg_prompt_embs,
                num_gen_latent=num_gen_latent,
                num_steps=num_video_steps,
                video_guidance_scale=video_guidance_scale,
                action_guidance_scale=action_guidance_scale,
                shift=shift,
                seed=seed,
                # Only honor num_action_steps in joint mode when caching the
                # static prefix; otherwise keep legacy behavior where both
                # schedulers use num_video_steps.
                num_action_steps=num_action_steps if kv_cache_cond else None,
                use_kv_cache=kv_cache_cond,
                step_callback=step_callback,
            )
            result["action_pred"] = action_pred
            if predict_video:
                result["video_latents"] = video_latents
            return result

        # ── Decoupled denoising (disjoint attention mode) ────────────────
        if has_action:
            result["action_pred"] = self._denoise_action(
                data=data,
                cond_latent=cond_latent,
                prompt_embs=prompt_embs,
                neg_prompt_embs=neg_prompt_embs,
                num_steps=num_action_steps,
                guidance_scale=action_guidance_scale,
                seed=seed,
                use_kv_cache=kv_cache_cond,
            )

        if not predict_video:
            return result

        assert num_gen_latent > 0, f"No frames to generate: num_gen_latent={num_gen_latent}"

        result["video_latents"] = self._denoise_video(
            cond_latent=cond_latent,
            prompt_embs=prompt_embs,
            neg_prompt_embs=neg_prompt_embs,
            num_gen_latent=num_gen_latent,
            num_steps=num_video_steps,
            guidance_scale=video_guidance_scale,
            shift=shift,
            seed=seed,
            video_padded_mask=data.get("video_padded_mask"),
        )

        return result

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to remap legacy checkpoint keys.

        Old checkpoints have video tokenizer and rope weights under
        `transformer.*`. The new model expects them under
        `video_tokenizer.*` and `video_rope.*` respectively.

        Also bridges Qwen2.5-VL 4.51.3 ↔ 4.57.6 layout via
        ``_fix_state_dict_key_on_load`` (the same hook HF runs on every
        internal load path including ``from_pretrained``), so direct callers
        and ``from_pretrained`` produce equivalent post-load weights.
        """
        from groot.core.training.models.n2.utils.checkpoint_compat import (
            remap_n2_vae_key,
            resize_temporal_pos_embedding,
        )

        current_state = self.state_dict() if not strict else {}
        remapped = {}
        for key, val in state_dict.items():
            new_key = key
            if key.startswith("transformer."):
                suffix = key[len("transformer.") :]
                if any(suffix.startswith(p) for p in VIDEO_ROPE_PREFIXES):
                    new_key = f"video_rope.{suffix}"
                elif any(suffix.startswith(p) for p in VIDEO_TOKENIZER_PREFIXES):
                    new_key = f"video_tokenizer.{suffix}"
            new_key = remap_n2_vae_key(new_key)
            new_key, _ = self._fix_state_dict_key_on_load(new_key)
            if (
                not strict
                and new_key in ACTION_ROPE_SEQ_KEYS
                and new_key in current_state
                and current_state[new_key].shape != val.shape
            ):
                continue
            if (
                not strict
                and new_key == ACTION_TEMPORAL_POS_EMB_KEY
                and new_key in current_state
                and val.ndim == current_state[new_key].ndim == 3
                and val.shape[1] < current_state[new_key].shape[1]
            ):
                old_horizon = val.shape[1]
                resized_val = resize_temporal_pos_embedding(
                    val,
                    current_state[new_key],
                    key=new_key,
                )
                logging.info(
                    "[GrootN2] Resized %s horizon: %s -> %d",
                    new_key,
                    old_horizon,
                    resized_val.shape[1],
                )
                remapped[new_key] = resized_val
                continue
            remapped[new_key] = val
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    @staticmethod
    def _fix_state_dict_key_on_load(key):
        """Per-key Qwen2.5-VL 4.51.3 ↔ 4.57.6 remap on every HF load path.

        HF ``transformers`` calls this static hook for each checkpoint key on every
        load path (``from_pretrained``, ``_load_state_dict_into_meta_model``, …), so it's
        the only place a key remap reliably reaches both the explicit
        ``model.load_state_dict(...)`` callers and ``Gr00tN2Policy`` / ``Gr00tPolicy``
        which load via ``AutoModel.from_pretrained`` and bypass ``Module.load_state_dict``
        / ``_load_state_dict_pre_hook`` entirely.

        Direction is fixed by the installed transformers version (so a 4.51.3
        ckpt loaded under 4.57.6 picks up the ``.model.visual.`` /
        ``.model.language_model.`` insertion, and vice versa). Wrapper prefix is
        hardcoded to ``text_encoder.text_encoder.`` (groot's CosmosTextEncoder
        is the only Qwen wrapper in GrootN2); non-Qwen keys pass through.

        Chain to ``PreTrainedModel._fix_state_dict_key_on_load`` is guarded so that
        if a future transformers release renames or removes the parent we still
        return a well-formed result and the Qwen remap still applies.
        """
        from groot.core.training.models.n2.utils.checkpoint_compat import (
            remap_n2_vae_key,
            remap_qwen2_5_vl_key,
        )

        parent_hook = getattr(PreTrainedModel, "_fix_state_dict_key_on_load", None)
        if parent_hook is not None:
            new_key, parent_renamed = parent_hook(key)
        else:
            new_key, parent_renamed = key, False
        new_key = remap_qwen2_5_vl_key(new_key)
        new_key = remap_n2_vae_key(new_key)
        return new_key, parent_renamed or (new_key != key)

    def load_cosmos_pretrained(self, path: str, device: str = "cuda"):
        """Load a base Cosmos checkpoint to initialize the DiT weights before training.

        Only needed once at training start. The base Cosmos checkpoint stores all
        weights under flat prefixes (patch_embed.*, rope.*, transformer_blocks.*, etc.).
        This method routes them to the split architecture (video_tokenizer, video_rope,
        transformer) and handles patch_embed zero-padding for view_embed channels.

        NOT needed for inference — from_pretrained loads the already-split trained
        checkpoint directly. Called from setup.py, not from __init__.
        """
        from safetensors.torch import load_file

        shard_files = sorted(glob.glob(f"{path}/model-*.safetensors"))
        if not shard_files:
            shard_files = sorted(glob.glob(f"{path}/*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        sd: dict[str, torch.Tensor] = {}
        for sf in shard_files:
            sd.update(load_file(sf, device=device))

        if self.config.vae_type != "wan21":
            # Cosmos ships a 16-channel DiT: patch_embed and proj_out are the only
            # weights whose shape tracks the latent space, so in a wider one neither
            # is reusable. They keep their fresh init; everything else transfers.
            sd = {k: v for k, v in sd.items() if not k.startswith(("patch_embed.", "proj_out."))}

        # Handle patch_embed shape mismatch (view_embed_dim zero-padding)
        pe_key = "patch_embed.proj.weight"
        if pe_key in sd:
            ckpt_shape = sd[pe_key].shape
            model_shape = self.video_tokenizer.patch_embed.proj.weight.shape
            if ckpt_shape != model_shape and ckpt_shape[0] == model_shape[0]:
                logging.info(
                    f"[GrootN2] patch_embed shape mismatch: ckpt={ckpt_shape} "
                    f"model={model_shape}, zero-padding"
                )
                padded = torch.zeros(model_shape, dtype=sd[pe_key].dtype, device=sd[pe_key].device)
                padded[:, : ckpt_shape[1]] = sd[pe_key]
                sd[pe_key] = padded

        # Split by prefix (3-way: video_tokenizer, video_rope, transformer)
        vt_sd = {
            k: v for k, v in sd.items() if any(k.startswith(p) for p in VIDEO_TOKENIZER_PREFIXES)
        }
        rope_sd = {k: v for k, v in sd.items() if any(k.startswith(p) for p in VIDEO_ROPE_PREFIXES)}
        tf_sd = {k: v for k, v in sd.items() if k not in vt_sd and k not in rope_sd}

        missing_vt, unexpected_vt = self.video_tokenizer.load_state_dict(
            vt_sd, strict=False, assign=True
        )
        missing_rope, unexpected_rope = self.video_rope.load_state_dict(
            rope_sd, strict=False, assign=True
        )
        missing_tf, unexpected_tf = self.transformer.load_state_dict(
            tf_sd, strict=False, assign=True
        )
        all_missing = missing_vt + missing_rope + missing_tf
        all_unexpected = unexpected_vt + unexpected_rope + unexpected_tf
        if all_missing:
            logging.info(f"[GrootN2] Missing keys ({len(all_missing)}): {all_missing[:5]}...")
        if all_unexpected:
            logging.info(
                f"[GrootN2] Unexpected keys ({len(all_unexpected)}): {all_unexpected[:5]}..."
            )


# Register with HuggingFace
AutoConfig.register("GrootN2", GrootN2Config)
AutoModel.register(GrootN2Config, GrootN2)
