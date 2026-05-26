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
GPU integration test for Gr00tPolicy._get_action() with the real model architecture.

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

Requires GPU and HF_TOKEN (for gated metadata download), but not model weights.
``tests/conftest.py`` sets ``GROOT_SKIP_HF_MODEL_WEIGHTS=1`` so
``from_pretrained`` constructs the architecture from config without reading
multi-GB safetensor shards.

``test_warmup_model_load`` owns the model-load cost under a wide timeout so
a slow checkpoint read does not consume the per-test budget of every
inference test in the class.
"""

import time

import numpy as np
import pytest
import torch


EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"


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


@pytest.fixture(scope="module")
def policy(request):
    """Shared Gr00tPolicy on ``cuda:0`` reused across all tests in this module."""
    if not request.node.nodeid.endswith("::test_warmup_model_load"):
        print(
            f"[gpu_policy] WARNING: model load triggered by {request.node.nodeid!r}, "
            "not test_warmup_model_load. Include test_warmup_model_load in the "
            "selection when debugging model construction failures.",
            flush=True,
        )

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    t0 = time.perf_counter()
    p = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=MODEL_REPO_ID,
        device="cuda:0",
    )
    # This CI test skips trained weights, so the model can emit dummy rot6d
    # action components that are finite but not valid rotations. Keep the
    # architecture path covered without converting relative EEF actions back
    # into absolute poses.
    p.processor.state_action_processor.use_relative_action = False
    print(f"[gpu_policy] Gr00tPolicy load took {time.perf_counter() - t0:.1f}s", flush=True)
    return p


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_warmup_model_load(policy):
    """Load the policy under a wide timeout and assert it is on GPU."""
    assert policy.model is not None
    assert policy.processor is not None
    device = next(policy.model.parameters()).device
    assert device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.timeout(120)
class TestGr00tPolicyGPU:
    """End-to-end GPU inference through Gr00tPolicy."""

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

    def test_get_action_accepts_different_inputs(self, policy):
        """Different synthetic observations should both complete without numeric failures."""
        for seed in (0, 12345):
            obs = _build_observation(policy, batch_size=1, seed=seed)
            action, _ = policy.get_action(obs)
            for key, arr in action.items():
                assert np.all(np.isfinite(arr)), f"{key} contains NaN or Inf"
