# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
GPU integration test for Gr00tPolicy._get_action() with a real model.

This top-down test exercises the full inference pipeline:
  Gr00tPolicy._get_action()
    → processor.__call__() (VLM tokenization + state/action normalization)
    → model.get_action() (backbone forward + DiT diffusion denoising)
    → processor.decode_action() (denormalization + action decoding)

Covers modules with 0% or low coverage that cannot be tested on CPU:
  - gr00t_n1d7.py (model forward)
  - qwen3_backbone.py (VLM backbone)
  - dit.py / alternate_vl_dit.py (transformer)
  - flowmatching_modules.py (diffusion scheduler)
  - embodiment_conditioned_mlp.py (action head)

Requires GPU, HF_TOKEN (for gated download), and model weights.
Weights are cached under the shared drive in CI or ``~/.cache/g00t/models/`` locally;
if absent, ``resolve_shared_model_path`` downloads using ``HF_TOKEN``.
"""

from pathlib import Path

import numpy as np
import pytest
from test_support.runtime import resolve_shared_model_path
import torch


EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"


def _prepare_model_path() -> Path:
    """Resolve model weights (uses ``HF_TOKEN`` if the shared cache is empty)."""
    return resolve_shared_model_path(MODEL_REPO_ID)


def _build_observation(policy, batch_size=1, seed=42):
    """Build a synthetic observation that matches the policy's modality config.

    Uses a fixed seed so that failures are reproducible.
    """
    rng = np.random.RandomState(seed)
    mc = policy.modality_configs

    video_horizon = len(mc["video"].delta_indices)
    state_horizon = len(mc["state"].delta_indices)

    obs = {"video": {}, "state": {}, "language": {}}

    for k in mc["video"].modality_keys:
        obs["video"][k] = rng.randint(
            0, 255, (batch_size, video_horizon, 256, 256, 3), dtype=np.uint8
        )

    embodiment_val = policy.embodiment_tag.value
    norm_params = policy.processor.state_action_processor.norm_params[embodiment_val]["state"]
    for k in mc["state"].modality_keys:
        dim = int(norm_params[k]["dim"])
        obs["state"][k] = rng.randn(batch_size, state_horizon, dim).astype(np.float32)

    language_key = mc["language"].modality_keys[0]
    obs["language"][language_key] = [["pick up the red cube"]] * batch_size

    return obs


@pytest.mark.gpu
@pytest.mark.timeout(300)
class TestGr00tPolicyGPU:
    """End-to-end GPU inference through Gr00tPolicy."""

    @pytest.fixture(scope="class")
    def policy(self):
        model_path = _prepare_model_path()

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        return Gr00tPolicy(
            embodiment_tag=EMBODIMENT_TAG,
            model_path=str(model_path),
            device="cuda:0",
        )

    def test_policy_loads_on_gpu(self, policy):
        assert policy.model is not None
        assert policy.processor is not None
        device = next(policy.model.parameters()).device
        assert device.type == "cuda"

    def test_get_action_keys_match_config(self, policy):
        obs = _build_observation(policy, batch_size=1)
        action, _ = policy.get_action(obs)
        expected_keys = set(policy.modality_configs["action"].modality_keys)
        assert set(action.keys()) == expected_keys

    def test_get_action_shapes(self, policy):
        obs = _build_observation(policy, batch_size=1)
        action, _ = policy.get_action(obs)
        action_horizon = len(policy.modality_configs["action"].delta_indices)
        for key, arr in action.items():
            assert arr.dtype == np.float32, f"{key} dtype mismatch"
            assert arr.ndim == 3, f"{key} should be (B, T, D), got ndim={arr.ndim}"
            assert arr.shape[0] == 1, f"{key} batch size mismatch"
            assert arr.shape[1] == action_horizon, f"{key} horizon mismatch"

    def test_get_action_values_finite_and_bounded(self, policy):
        """Action values should be finite and within a reasonable magnitude."""
        obs = _build_observation(policy, batch_size=1)
        action, _ = policy.get_action(obs)
        for key, arr in action.items():
            assert np.all(np.isfinite(arr)), f"{key} contains NaN or Inf"
            assert np.all(np.abs(arr) < 1e4), (
                f"{key} has values with |v| >= 1e4: max_abs={np.max(np.abs(arr)):.2f}. "
                "This suggests the model output or denormalization is broken."
            )

    def test_get_action_batch(self, policy):
        obs = _build_observation(policy, batch_size=2)
        action, _ = policy.get_action(obs)
        for key, arr in action.items():
            assert arr.shape[0] == 2, f"{key}: expected batch=2, got {arr.shape[0]}"

    def test_get_action_deterministic(self, policy):
        """Same input + same torch seed must produce the same output."""
        obs1 = _build_observation(policy, batch_size=1, seed=99)
        obs2 = _build_observation(policy, batch_size=1, seed=99)
        torch.manual_seed(0)
        action1, _ = policy.get_action(obs1)
        torch.manual_seed(0)
        action2, _ = policy.get_action(obs2)
        for key in action1:
            np.testing.assert_array_equal(
                action1[key],
                action2[key],
                err_msg=f"{key}: same input + same seed produced different outputs — "
                "model may have uncontrolled stochasticity beyond the diffusion noise",
            )

    def test_get_action_sensitive_to_input(self, policy):
        """Different inputs must produce different outputs — model is not degenerate."""
        obs_a = _build_observation(policy, batch_size=1, seed=0)
        obs_b = _build_observation(policy, batch_size=1, seed=12345)
        action_a, _ = policy.get_action(obs_a)
        action_b, _ = policy.get_action(obs_b)
        any_differ = False
        for key in action_a:
            if not np.allclose(action_a[key], action_b[key], atol=1e-6):
                any_differ = True
                break
        assert any_differ, (
            "Model returned identical actions for completely different observations — "
            "the model may be ignoring its input (degenerate)."
        )
