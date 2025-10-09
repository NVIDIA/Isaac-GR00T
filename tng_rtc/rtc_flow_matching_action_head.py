import torch
from typing import TypeAlias, Literal

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from transformers.feature_extraction_utils import BatchFeature


MAX_GUIDANCE_WEIGHT: float = 5.0
PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]

class RTCFlowmatchingActionHead(FlowmatchingActionHead):
    
    @torch.no_grad()
    def get_realtime_action(self, backbone_output: BatchFeature, action_input: BatchFeature, A_prev: torch.Tensor, d: int, s: int, H: int, max_guidance_weight = MAX_GUIDANCE_WEIGHT, weight_function: PrefixAttentionSchedule = "exp", start_actions = None) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        if start_actions is not None:
            actions = start_actions
        else:
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.config.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
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
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embeds.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            f_actions = actions + (1 - t_cont) * pred_velocity

            weights = get_prefix_weights(
                    d, H - s, H, weight_function, device
                )
            error = (A_prev - f_actions) * weights[:, None]
            
            # In original paper, gradients are calculated and later used in velocity calculation, but not using them seems to work better
            # g = torch.autograd.grad(outputs=f_actions, inputs=actions, grad_outputs=error)[0]
            
            inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
            if abs(t_cont) < 1e-8:
                c = max_guidance_weight
            else:
                c = (1 - t_cont) / t_cont
            guidance_weight = min(c * inv_r2, max_guidance_weight)

            realtime_velocity = pred_velocity + guidance_weight * error

            actions = actions + dt * realtime_velocity
        
        return BatchFeature(data={"action_pred": actions})


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule, device) -> torch.Tensor:
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32, device=device)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32, device=device)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule == "linear" or schedule == "exp":
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, min=0, max=1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (torch.exp(torch.tensor(1.0)) - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return torch.where(idx >= end, torch.zeros_like(w), w)
