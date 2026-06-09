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

import logging
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)


def _broadcast_save_decision(save_flag: int, metric_value: float) -> tuple[int, float]:
    """Broadcast rank-0's `(save_flag, metric_value)` decision to every rank.

    HF's eval loop populates ``metrics`` only on rank-0 in some configurations,
    so the "should we save?" branch must be decided there and synced. Without
    this, non-rank-0 ranks would skip the collective ``Trainer.save_model``
    call below and rank-0 would deadlock inside the consolidated-gather.

    Returns the broadcast pair on every rank. Single-rank or non-distributed
    case: returns the input unchanged.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return save_flag, metric_value

    backend = dist.get_backend()
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if backend == "nccl"
        else torch.device("cpu")
    )
    # Use float64 so the metric reaches every rank bit-for-bit identical to
    # rank-0 — the value is interpolated into the checkpoint directory name
    # and must agree across ranks.
    payload = torch.tensor(
        [float(save_flag), float(metric_value)], device=device, dtype=torch.float64
    )
    dist.broadcast(payload, src=0)
    return int(payload[0].item()), float(payload[1].item())


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        processor_dir: Path | None = None,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class BestMetricCheckpointCallback(TrainerCallback):
    """Save a copy of the model whenever an evaluation metric improves.

    Works under DDP, DeepSpeed ZeRO 1/2/3, and PyTorch FSDP: the save is
    delegated to ``Trainer.save_model``, which handles parameter
    consolidation under sharded backends. The callback's own job is to
    agree across ranks on whether this eval round should save, and to
    manage the best-checkpoint directory (mkdir, optional ``exp_cfg_dir``
    copy, previous-best cleanup).
    """

    def __init__(
        self,
        metric_name: str,
        trainer: Trainer,
        *,
        greater_is_better: bool = True,
        exp_cfg_dir: Path | None = None,
    ):
        """
        Args:
            metric_name: Key in the eval ``metrics`` dict to track.
            trainer: The owning ``Trainer``; needed to call
                ``trainer.save_model`` (the sharded-aware save path).
            greater_is_better: True when the metric should be maximized.
            exp_cfg_dir: Directory copied alongside each best checkpoint.
        """
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None
        self._trainer = trainer

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        save_flag = 0
        metric_value = 0.0
        if state.is_world_process_zero and metrics is not None:
            current = metrics.get(self.metric_name, None)
            if current is not None:
                is_better = (
                    current > self.best_metric
                    if self.greater_is_better
                    else current < self.best_metric
                )
                if is_better:
                    save_flag = 1
                    metric_value = float(current)

        save_flag, metric_value = _broadcast_save_decision(save_flag, metric_value)
        if save_flag == 0:
            return

        self.best_metric = metric_value

        best_checkpoint_dir = (
            Path(args.output_dir)
            / f"checkpoint-{state.global_step}-best-{self.metric_name}_{metric_value}"
        )

        if state.is_world_process_zero:
            best_checkpoint_dir.mkdir(exist_ok=True)

        # Wait for rank-0's mkdir to land before any rank enters save_model;
        # otherwise on NFS a non-rank-0 rank can race past the mkdir and have
        # save_model fail to find the directory.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        # Collective on every rank: gathers params under ZeRO-3 / FSDP, then
        # writes from rank-0. Calling this only on rank-0 would deadlock the
        # rest of the process group inside the gather.
        self._trainer.save_model(str(best_checkpoint_dir))

        if state.is_world_process_zero:
            if self.exp_cfg_dir is not None and self.exp_cfg_dir.exists():
                exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                logger.info(
                    "Copying experiment config directory %s to %s",
                    self.exp_cfg_dir,
                    exp_cfg_dst,
                )
                shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            logger.info(
                "Best checkpoint saved to %s with metric %s = %s",
                best_checkpoint_dir,
                self.metric_name,
                metric_value,
            )

            if self._best_checkpoint_dir is not None and Path(self._best_checkpoint_dir).exists():
                shutil.rmtree(self._best_checkpoint_dir)

            self._best_checkpoint_dir = str(best_checkpoint_dir)
