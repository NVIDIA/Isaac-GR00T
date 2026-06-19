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
Standalone inference smoke tests.

Loads Gr00tPolicy and LeRobotEpisodeLoader once per embodiment variant (module-scoped
fixture), then calls the internal Python functions directly — no subprocess overhead.

Variants exercised: LIBERO, DROID, SimplerEnv-Fractal, SimplerEnv-Bridge.

Environment variables (optional, per-embodiment):
  INFERENCE_TEST_LIBERO_MODEL_PATH            – LIBERO checkpoint path override
  INFERENCE_TEST_LIBERO_DATASET_PATH          – LIBERO dataset path override
  INFERENCE_TEST_DROID_MODEL_PATH             – DROID checkpoint path override
  INFERENCE_TEST_DROID_DATASET_PATH           – DROID dataset path override
  INFERENCE_TEST_SIMPLERENV_FRACTAL_MODEL_PATH   – SimplerEnv-Fractal model override
  INFERENCE_TEST_SIMPLERENV_FRACTAL_DATASET_PATH – SimplerEnv-Fractal dataset override
  INFERENCE_TEST_SIMPLERENV_BRIDGE_MODEL_PATH    – SimplerEnv-Bridge model override
  INFERENCE_TEST_SIMPLERENV_BRIDGE_DATASET_PATH  – SimplerEnv-Bridge dataset override
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import evaluate_single_trajectory
from gr00t.policy.gr00t_policy import Gr00tPolicy
import pytest
from test_support.runtime import get_root, resolve_demo_dataset, resolve_model_checkpoint_path
import torch


ROOT = get_root()

# scripts/deployment/ is not a Python package; add it to sys.path so we can
# import standalone_inference_script directly.
_DEPLOY_DIR = str(ROOT / "scripts" / "deployment")
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from standalone_inference_script import run_single_trajectory  # noqa: E402


@dataclass(frozen=True)
class InferenceVariant:
    """Configuration for one embodiment variant of the inference smoke tests."""

    id: str
    embodiment_tag: str
    hf_repo_id: str
    hf_subdir: str | None
    dataset_name: str
    model_env_var: str = ""
    dataset_env_var: str = ""

    def __str__(self) -> str:
        return self.id


LIBERO = InferenceVariant(
    id="libero",
    embodiment_tag="LIBERO_PANDA",
    hf_repo_id="nvidia/GR00T-N1.7-LIBERO",
    hf_subdir="libero_10",
    dataset_name="libero_demo",
    model_env_var="INFERENCE_TEST_LIBERO_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_LIBERO_DATASET_PATH",
)

DROID = InferenceVariant(
    id="droid",
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    hf_repo_id="nvidia/GR00T-N1.7-DROID",
    hf_subdir=None,
    dataset_name="droid_sample",
    model_env_var="INFERENCE_TEST_DROID_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_DROID_DATASET_PATH",
)

SIMPLERENV_FRACTAL = InferenceVariant(
    id="simplerenv_fractal",
    embodiment_tag="SIMPLER_ENV_GOOGLE",
    hf_repo_id="nvidia/GR00T-N1.7-SimplerEnv-Fractal",
    hf_subdir=None,
    dataset_name="simplerenv_fractal_sample",
    model_env_var="INFERENCE_TEST_SIMPLERENV_FRACTAL_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_SIMPLERENV_FRACTAL_DATASET_PATH",
)

SIMPLERENV_BRIDGE = InferenceVariant(
    id="simplerenv_bridge",
    embodiment_tag="SIMPLER_ENV_WIDOWX",
    hf_repo_id="nvidia/GR00T-N1.7-SimplerEnv-Bridge",
    hf_subdir=None,
    dataset_name="simplerenv_bridge_sample",
    model_env_var="INFERENCE_TEST_SIMPLERENV_BRIDGE_MODEL_PATH",
    dataset_env_var="INFERENCE_TEST_SIMPLERENV_BRIDGE_DATASET_PATH",
)

VARIANTS = [LIBERO, DROID, SIMPLERENV_FRACTAL, SIMPLERENV_BRIDGE]

DEVICE_BASE_MODEL_REPO = "nvidia/GR00T-N1.7-3B"
DEVICE_DROID_DATASET_NAME = "droid_sample"
DEVICE_DROID_EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


def _model_path(variant: InferenceVariant) -> str:
    return str(
        resolve_model_checkpoint_path(
            hf_repo_id=variant.hf_repo_id,
            hf_subdir=variant.hf_subdir,
            path_override_env=variant.model_env_var,
            repo_root=ROOT,
        )
    )


def _dataset_path(variant: InferenceVariant) -> str:
    return str(
        resolve_demo_dataset(
            dataset_name=variant.dataset_name,
            path_override_env=variant.dataset_env_var,
            repo_root=ROOT,
        )
    )


def _assert_real_demo_file(dataset_path) -> None:
    demo_check_file = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
    assert demo_check_file.is_file(), f"Demo dataset parquet not found: {demo_check_file}"
    assert b"git-lfs" not in demo_check_file.read_bytes()[:50], (
        f"Demo data file is a Git LFS pointer, not real data: {demo_check_file}"
    )


@dataclass
class LoadedVariant:
    """Holds the pre-loaded policy and dataset loader for one variant."""

    variant: InferenceVariant
    policy: Gr00tPolicy
    loader: LeRobotEpisodeLoader
    embodiment_tag: EmbodimentTag
    model_path: str
    dataset_path: str


@pytest.fixture(scope="module", params=VARIANTS, ids=str)
def loaded_variant(request, load_hf_model_weights):
    """Load Gr00tPolicy + LeRobotEpisodeLoader once per variant for the whole module."""
    variant: InferenceVariant = request.param
    model_path = _model_path(variant)
    dataset_path = _dataset_path(variant)
    embodiment_tag = EmbodimentTag.resolve(variant.embodiment_tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with load_hf_model_weights():
        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
        )

    modality = policy.get_modality_config()

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality,
    )

    yield LoadedVariant(
        variant=variant,
        policy=policy,
        loader=loader,
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        dataset_path=dataset_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_standalone_inference_pytorch(loaded_variant: LoadedVariant) -> None:
    """Run standalone inference in PyTorch mode for a few steps on trajectory 0."""
    v = loaded_variant
    run_single_trajectory(
        policy=v.policy,
        loader=v.loader,
        traj_id=0,
        embodiment_tag=v.embodiment_tag,
        steps=20,
        action_horizon=8,
    )


@pytest.mark.edge_device
@pytest.mark.timeout(1800)
def test_device_standalone_inference_script_pytorch() -> None:
    """Mirror the original device.test job with the base 3B model and DROID demo data."""
    assert torch.cuda.is_available(), "CUDA not available"

    model_path = resolve_model_checkpoint_path(
        hf_repo_id=DEVICE_BASE_MODEL_REPO,
        path_override_env="INFERENCE_TEST_DEVICE_MODEL_PATH",
        repo_root=ROOT,
    )
    dataset_path = resolve_demo_dataset(
        dataset_name=DEVICE_DROID_DATASET_NAME,
        path_override_env="INFERENCE_TEST_DEVICE_DATASET_PATH",
        global_env_var="DROID_DEMO_DATASET_PATH",
        repo_root=ROOT,
    )
    _assert_real_demo_file(dataset_path)

    env = os.environ.copy()
    env["GROOT_SKIP_HF_MODEL_WEIGHTS"] = "0"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "deployment" / "standalone_inference_script.py"),
        "--model-path",
        str(model_path),
        "--dataset-path",
        str(dataset_path),
        "--embodiment-tag",
        DEVICE_DROID_EMBODIMENT_TAG,
        "--traj-ids",
        "0",
        "--action-horizon",
        "8",
        "--inference-mode",
        "pytorch",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, (
        "Device standalone inference script failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Output tail:\n{result.stdout[-12000:]}"
    )


@pytest.mark.gpu
@pytest.mark.edge_device
@pytest.mark.timeout(600)
def test_standalone_inference_invalid_traj_id(loaded_variant: LoadedVariant) -> None:
    """Out-of-range traj_id should raise an index error, not UnboundLocalError."""
    v = loaded_variant
    n = len(v.loader)
    assert n < 999, f"Expected dataset to have fewer than 999 trajectories, got {n}"
    with pytest.raises((IndexError, KeyError)):
        v.loader[999]


@pytest.mark.gpu
@pytest.mark.edge_device
@pytest.mark.timeout(600)
def test_open_loop_eval_with_checkpoint(loaded_variant: LoadedVariant) -> None:
    """Run evaluate_single_trajectory from open_loop_eval directly."""
    v = loaded_variant
    evaluate_single_trajectory(
        policy=v.policy,
        loader=v.loader,
        traj_id=0,
        embodiment_tag=v.embodiment_tag,
        steps=5,
        action_horizon=8,
    )
