# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T) -- per-batch or per-position timesteps
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Accept both (B,) and (B, T) timesteps.
        # (B,) is the standard case; (B, T) is used for training-time RTC
        # where prefix positions have tau=1.0 (clean) and postfix positions
        # have the sampled noise level.
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B, T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass  # already per-position
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape (B,) or (B, T), got {timesteps.shape}."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    # Training-time Real-Time Chunking (RTC) configuration.
    # When max_rtc_delay > 0, the training loop simulates inference delays by
    # randomly choosing a prefix length d ~ Uniform(0, max_rtc_delay) per sample.
    # Prefix positions (0..d-1) are set to clean (t=1.0) and the loss is computed
    # only on the postfix (d..T-1).  At d=0 the sample reduces to standard
    # flow-matching training, so the model stays compatible with non-RTC inference.
    max_rtc_delay: int = field(
        default=0,
        metadata={
            "help": "Maximum prefix delay for training-time RTC. "
            "0 disables RTC (standard training). "
            "When > 0, each training sample randomly picks a delay d in [0, max_rtc_delay] "
            "and only computes loss on the postfix actions."
        },
    )

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action  # (B, T, action_dim)
        B, T, _ = actions.shape
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # (B,)

        # ---- Training-time Real-Time Chunking (RTC) ----
        # Reference: "Training-Time Action Conditioning for Efficient Real-Time
        # Chunking" (arXiv:2512.05964), Algorithm 1.
        #
        # When max_rtc_delay > 0 we simulate an inference delay:
        #   - Sample delay d ~ Uniform(0, max_rtc_delay) per batch element.
        #   - Prefix positions (0..d-1) get t=1.0 (clean action), so
        #     noisy_trajectory[prefix] = actions (no noise).
        #   - Postfix positions (d..T-1) get the normally-sampled t.
        #   - Loss is only computed on postfix positions.
        #
        # When d=0 for a given sample (always the case when max_rtc_delay==0),
        # the behaviour is identical to standard flow-matching training.
        use_rtc = getattr(self.config, "max_rtc_delay", 0) > 0
        if use_rtc:
            max_delay = self.config.max_rtc_delay
            # Sample a random delay per batch element: d in [0, max_delay]
            delay = torch.randint(0, max_delay + 1, (B,), device=device)  # (B,)

            # Build per-position continuous timesteps: (B, T)
            pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
            prefix_mask = pos_idx < delay.unsqueeze(1)  # (B, T) bool

            # Per-position t: prefix => 1.0 (clean), postfix => sampled t
            t_per_pos = torch.where(
                prefix_mask,
                torch.ones(B, T, device=device, dtype=t.dtype),
                t.unsqueeze(1).expand(-1, T),
            )  # (B, T)

            # Noisy trajectory with per-position interpolation:
            #   x_t = t * action + (1 - t) * noise
            # For prefix (t=1.0): x_t = action (clean)
            # For postfix: standard flow matching interpolation
            t_expanded = t_per_pos.unsqueeze(-1)  # (B, T, 1)
            noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions

            # Velocity target is the same for all positions
            velocity = actions - noise

            # Discretize per-position timesteps for the action encoder: (B, T)
            t_discretized_per_pos = (t_per_pos * self.num_timestep_buckets).long()

            # For the DiT's global AdaLN conditioning we use the postfix
            # timestep (same as the sampled t), since that reflects the noise
            # level being denoised.
            t_discretized_global = (t * self.num_timestep_buckets).long()  # (B,)

            # Encode actions with per-position timesteps
            action_features = self.action_encoder(
                noisy_trajectory, t_discretized_per_pos, embodiment_id
            )

            # Build postfix-only loss mask: (B, T, 1) -- expanded to action_dim later
            postfix_mask = (~prefix_mask).unsqueeze(-1).float()  # (B, T, 1)
        else:
            # Standard flow-matching training (no RTC)
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            # Convert (continuous) t -> discrete if needed
            t_discretized_global = (t[:, 0, 0] * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy_trajectory, t_discretized_global, embodiment_id)
            postfix_mask = None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized_global,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute loss.
        action_mask = action_input.action_mask  # (B, T, action_dim)
        loss_per_element = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask

        if postfix_mask is not None:
            # RTC: only compute loss on postfix positions
            combined_mask = action_mask * postfix_mask  # (B, T, action_dim)
            loss = (loss_per_element * postfix_mask).sum() / combined_mask.sum().clamp(min=1.0)
        else:
            loss = loss_per_element.sum() / action_mask.sum()

        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        prefix_actions: torch.Tensor = None,
        num_prefix_steps: int = 0,
    ) -> BatchFeature:
        """
        Generate action predictions via flow-matching denoising.

        When ``prefix_actions`` is provided the method implements the inference
        algorithm from "Training-Time Action Conditioning for Efficient
        Real-Time Chunking" (arXiv:2512.05964, Algorithm 2):

        * Prefix positions (0 .. num_prefix_steps-1) are clamped to the known
          clean actions and their per-token timestep is set to
          ``num_timestep_buckets`` (i.e. t=1.0, fully clean).
        * Postfix positions follow the normal Euler denoising schedule.
        * The DiT's global AdaLN timestep uses the postfix schedule.

        This matches the training-time conditioning (``forward()`` with
        ``max_rtc_delay > 0``) so the model sees the same per-token timestep
        pattern it was trained on.

        Args:
            backbone_output: Output from the vision-language backbone.
            action_input: Action input features (state, embodiment_id, etc.).
            prefix_actions: Optional tensor of shape (B, K, action_dim) containing
                known actions to fix during denoising (inpainting). These are
                typically carried over from a previous prediction for temporal
                consistency in real-time action chunking.
            num_prefix_steps: Number of leading action steps to clamp to
                prefix_actions at each denoising iteration. Must be <= action_horizon.
                Ignored if prefix_actions is None.
        """

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        T = self.config.action_horizon
        actions = torch.randn(
            size=(batch_size, T, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        # Inpainting setup: initialize prefix positions with known actions
        # so the denoising process starts from a better initial point.
        use_inpainting = (
            prefix_actions is not None
            and num_prefix_steps > 0
            and num_prefix_steps <= T
        )
        if use_inpainting:
            prefix_actions = prefix_actions.to(dtype=actions.dtype, device=device)
            actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            if use_inpainting:
                # Per-position timesteps: prefix = num_timestep_buckets (t=1.0,
                # clean), postfix = current denoising step.
                t_per_pos = torch.full(
                    (batch_size, T), fill_value=t_discretized, dtype=torch.long, device=device,
                )
                t_per_pos[:, :num_prefix_steps] = self.num_timestep_buckets

                # Clamp prefix to clean actions before encoding
                actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

                action_features = self.action_encoder(actions, t_per_pos, embodiment_id)
            else:
                # Standard: single timestep for all positions
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # DiT global timestep: always the postfix denoising timestep
            global_timestep = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=global_timestep,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -T:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

            # Clamp prefix positions back after the Euler step.
            if use_inpainting:
                actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
