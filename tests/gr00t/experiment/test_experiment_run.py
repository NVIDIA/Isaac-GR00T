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
GPU integration test for experiment.run() with a single GPU and max_steps=1.

This is the highest fan-out test in the project — a single call to run()
exercises almost the entire training stack:
  experiment.run()
    → Config.validate()
    → MODEL_REGISTRY → Gr00tN1d7Pipeline.setup()
      → DatasetFactory (generate_stats, generate_rel_stats, ShardedSingleStepDataset)
      → Gr00tN1d7Processor (from_pretrained, set_statistics, save_pretrained)
      → Gr00tN1d7 model (from_pretrained with checkpoint)
    → TrainingArguments + Gr00tTrainer
    → trainer.train(max_steps=1)
    → trainer.save_model()

Covers modules at 0% coverage: experiment.py, trainer.py, model_pipeline.py,
and heavily exercises sharded_*.py, factory.py, processing_gr00t_n1d7.py.

Requires GPU, HF_TOKEN (for gated download), and model weights.
Weights are cached under the shared drive in CI or ``~/.cache/g00t/models/`` locally;
if absent, ``resolve_shared_model_path`` downloads using ``HF_TOKEN``.

Training data is the LIBERO ``libero_demo`` bundle: see ``resolve_libero_demo_dataset_path``
(``LIBERO_DEMO_DATASET_PATH``, in-repo ``demo_data/libero_demo`` with Git LFS, or
``TEST_CACHE_PATH/datasets/libero_demo``).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from test_support.runtime import (
    get_root,
    resolve_libero_demo_dataset_path,
    resolve_shared_model_path,
)
import torch


REPO_ROOT = get_root()
EMBODIMENT_TAG = "libero_sim"
MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"


def _prepare_model_path() -> Path:
    """Resolve model weights (uses ``HF_TOKEN`` if the shared cache is empty)."""
    return resolve_shared_model_path(MODEL_REPO_ID)


@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_experiment_run_single_gpu(tmp_path, monkeypatch):
    """Run experiment.run() for 1 training step on a single GPU.

    This verifies that the entire training pipeline — config validation, data
    loading, model initialization, one forward+backward pass, and checkpoint
    saving — completes without errors.
    """
    model_path = _prepare_model_path()
    dataset_path = resolve_libero_demo_dataset_path(REPO_ROOT)

    # Ensure single GPU, no distributed (monkeypatch restores env after the test)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    output_dir = tmp_path / "experiment_output"

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": EMBODIMENT_TAG,
                    }
                ],
                "video_backend": "torchcodec",
                "shard_size": 64,
                "num_shards_per_epoch": 1,
                "multiprocessing_context": "fork",
            },
        }
    )

    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True

    config.training.start_from_checkpoint = str(model_path)
    config.training.output_dir = str(output_dir)
    config.training.max_steps = 1
    config.training.save_steps = 1
    config.training.global_batch_size = 2
    config.training.num_gpus = 1
    config.training.dataloader_num_workers = 0
    config.training.use_wandb = False
    config.training.optim = "adamw_torch"
    config.training.bf16 = True
    config.training.tf32 = True
    config.training.fp16 = False
    config.training.gradient_checkpointing = False
    config.training.use_ddp = False
    config.training.eval_strategy = "no"

    run(config)

    assert output_dir.exists(), "Output directory was not created"
    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    assert len(checkpoint_dirs) >= 1, (
        f"Expected at least one checkpoint, found: {list(output_dir.iterdir())}"
    )

    ckpt = checkpoint_dirs[0]
    model_files = list(ckpt.glob("*.safetensors")) + list(ckpt.glob("*.bin"))
    assert len(model_files) >= 1, (
        f"Checkpoint {ckpt.name} contains no model weight files: {list(ckpt.iterdir())}"
    )
    for mf in model_files:
        assert mf.stat().st_size > 0, f"Model file {mf.name} is empty"

    experiment_cfg = output_dir / "experiment_cfg"
    assert experiment_cfg.is_dir(), "experiment_cfg directory missing"
    config_yaml = experiment_cfg / "config.yaml"
    assert config_yaml.exists(), "config.yaml not saved"
    config_text = config_yaml.read_text()
    assert "max_steps: 1" in config_text, "Saved config.yaml missing expected max_steps setting"
    assert EMBODIMENT_TAG in config_text, (
        f"Saved config.yaml missing embodiment_tag '{EMBODIMENT_TAG}'"
    )

    processor_dir = output_dir / "processor"
    assert processor_dir.is_dir(), "processor directory missing"
    processor_cfg = processor_dir / "processor_config.json"
    assert processor_cfg.exists(), "processor_config.json not saved"

    processor_data = json.loads(processor_cfg.read_text())
    assert "processor_class" in processor_data, (
        "processor_config.json missing 'processor_class' field"
    )

    # Verify training actually ran: trainer_state.json records the training progress
    trainer_state_path = ckpt / "trainer_state.json"
    assert trainer_state_path.exists(), (
        f"trainer_state.json missing from {ckpt.name} — training may not have run"
    )
    trainer_state = json.loads(trainer_state_path.read_text())
    assert trainer_state.get("global_step", 0) >= 1, (
        f"global_step is {trainer_state.get('global_step')}, expected >= 1"
    )
    log_history = trainer_state.get("log_history", [])
    loss_entries = [e for e in log_history if "loss" in e]
    if loss_entries:
        last_loss = loss_entries[-1]["loss"]
        assert np.isfinite(last_loss), f"Training loss is not finite: {last_loss}"

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
