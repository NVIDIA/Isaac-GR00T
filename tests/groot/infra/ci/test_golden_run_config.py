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
"""Tests for the public YAML-backed golden-run registry."""

from pathlib import Path

import pytest

from groot.infra.ci.compute_type import ComputeType
from groot.infra.ci.golden_run_config import CONFIG_BACKED_LAUNCHER, get_variant, load_variants


def test_loads_n2_minwam_sim_oss_variant_from_default_yaml():
    variant = get_variant("n2-minwam-sim")

    assert variant.run_type == "n2-minwam-sim"
    assert variant.label == "N2 Min-WAM Sim OSS Training"
    assert variant.launcher.endswith("launch_n2_minwam_sim_oss_golden.py")
    assert variant.launcher.startswith(("groot/", "oss/groot/"))
    assert variant.experiment is None
    assert variant.data_config is None
    assert variant.training_config is None
    assert variant.eval is None
    assert variant.artifacts is not None
    assert variant.artifacts.base_model.repo_id == "nvidia/GR00T-N2-YAM-Pretrain-Bidir-200K"
    assert variant.artifacts.base_model.revision == "checkpoint-200000"
    assert variant.artifacts.dataset.repo_id == "nvidia/GR00T-N2-YAM-Stack-Bowls-200Traj"
    assert variant.artifacts.dataset.revision == "main"
    assert variant.artifacts.dataset.local_name == "yam_workstation_stack_bowls_20260430_170700"
    assert variant.artifacts.final_model.repo_id == "nvidia/GR00T-N2-MinWAM-Stack-Bowls-PostTrain"
    assert variant.artifacts.final_model.revision == "main"
    assert variant.platform(ComputeType.H100).num_gpus == 96
    assert variant.platform(ComputeType.H100).num_nodes == 13
    assert variant.platform(ComputeType.H100).batch_size == 4

    with pytest.raises(ValueError, match="does not support compute_type 'gb300'"):
        variant.platform(ComputeType.GB300)
    with pytest.raises(ValueError, match="does not support compute_type 'gb200'"):
        variant.platform(ComputeType.GB200)


def test_public_default_yaml_contains_oss_golden_runs():
    assert tuple(load_variants()) == ("n2-minwam-sim",)


def test_unknown_variant_lists_valid_values():
    with pytest.raises(
        ValueError,
        match="Valid variants: n2-minwam-sim",
    ):
        get_variant("missing")


def test_rejects_unknown_platform(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
    platforms:
      badgpu:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="unknown platform 'badgpu'"):
        load_variants(config)


def test_loads_variant_branches(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    branches:
      - "main"
      - "release/sample"
    eval:
      type: "yam"
      tasks: ["Task"]
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].branches == ("main", "release/sample")


def test_loads_variant_data_and_training_configs(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    data_config: "configs/data.yaml"
    training_config: "configs/train.yaml"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    variant = load_variants(config)["sample"]

    assert variant.launcher == CONFIG_BACKED_LAUNCHER
    assert variant.data_config == "configs/data.yaml"
    assert variant.training_config == "configs/train.yaml"


def test_loads_variant_experiment(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    experiment: "n2_yam_simulation_stack_bowls"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    variant = load_variants(config)["sample"]

    assert variant.launcher == CONFIG_BACKED_LAUNCHER
    assert variant.experiment == "n2_yam_simulation_stack_bowls"
    assert variant.data_config is None
    assert variant.training_config is None


def test_requires_data_and_training_configs_together(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    data_config: "configs/data.yaml"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="must set both data_config and training_config"):
        load_variants(config)


def test_rejects_launcher_with_data_and_training_configs(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "custom.py"
    data_config: "configs/data.yaml"
    training_config: "configs/train.yaml"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="launcher must be omitted"):
        load_variants(config)


def test_rejects_launcher_with_experiment(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "custom.py"
    experiment: "n2_yam_simulation_stack_bowls"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="launcher must be omitted"):
        load_variants(config)


def test_requires_launcher_for_legacy_variant(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="launcher is required"):
        load_variants(config)


def test_eval_checkpoint_step_defaults_to_10000(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].eval.checkpoint_step == 10000


def test_eval_checkpoint_step_override(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
      checkpoint_step: 30000
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].eval.checkpoint_step == 30000


def test_eval_checkpoint_step_must_be_positive(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
      checkpoint_step: 0
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match=r"eval\.checkpoint_step must be a positive integer"):
        load_variants(config)


def test_eval_num_eval_nodes_defaults_to_1(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].eval.num_eval_nodes == 1


def test_eval_num_eval_nodes_override(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
      num_eval_nodes: 4
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].eval.num_eval_nodes == 4


def test_eval_num_eval_nodes_must_be_positive(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval:
      type: "yam"
      tasks: ["Task"]
      num_eval_nodes: 0
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match=r"eval\.num_eval_nodes must be a positive integer"):
        load_variants(config)


def test_empty_eval_block_disables_eval(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval: {}
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    assert load_variants(config)["sample"].eval is None


def test_loads_hf_artifacts(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval: {}
    artifacts:
      base_model:
        repo_id: "org/base"
        revision: "checkpoint-1"
      dataset:
        repo_id: "org/data"
        revision: "main"
        local_name: "sample_dataset"
      final_model:
        repo_id: "org/final"
        revision: "main"
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    artifacts = load_variants(config)["sample"].artifacts

    assert artifacts is not None
    assert artifacts.base_model.repo_id == "org/base"
    assert artifacts.base_model.revision == "checkpoint-1"
    assert artifacts.dataset.repo_id == "org/data"
    assert artifacts.dataset.local_name == "sample_dataset"
    assert artifacts.final_model.repo_id == "org/final"


def test_rejects_unknown_artifact_field(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval: {}
    artifacts:
      base_model:
        repo_id: "org/base"
        revision: "main"
      dataset:
        repo_id: "org/data"
        revision: "main"
        local_name: "sample_dataset"
      final_model:
        repo_id: "org/final"
        revision: "main"
      surprise: "nope"
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="artifacts has unknown field"):
        load_variants(config)


def test_requires_artifact_blocks_when_artifacts_are_set(tmp_path: Path):
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample"
    run_type: "sample"
    launcher: "launcher.py"
    eval: {}
    artifacts:
      base_model:
        repo_id: "org/base"
        revision: "main"
      final_model:
        repo_id: "org/final"
        revision: "main"
    platforms:
      h100:
        num_gpus: 1
        num_nodes: 1
        batch_size: 1
"""
    )

    with pytest.raises(ValueError, match="artifacts.dataset must be a non-empty mapping"):
        load_variants(config)
