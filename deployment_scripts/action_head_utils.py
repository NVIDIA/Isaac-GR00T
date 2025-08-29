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

import torch
from transformers.feature_extraction_utils import BatchFeature


def action_head_pytorch_forward(self, backbone_output, action_input):

    backbone_output = self.process_backbone_output(backbone_output)

    # Get vision and language embeddings.
    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id

    # Embed state.
    state_features = self.state_encoder(action_input.state, embodiment_id)

    # Set initial actions as the sampled noise.
    batch_size = vl_embs.shape[0]
    device = vl_embs.device
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.config.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )

    # This attribute is used to ensure the same actions is used for both PyTorch and TensorRT inference
    if hasattr(self, "init_actions"):
        actions = self.init_actions.expand((batch_size, -1, -1))
    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    # Run denoising steps.
    for t in range(num_steps):
        t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory.
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # Run model forward.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output, embodiment_id)

        pred_velocity = pred[:, -self.action_horizon :]

        # Update actions using euler integration.
        actions = actions + dt * pred_velocity
    return BatchFeature(data={"action_pred": actions})
