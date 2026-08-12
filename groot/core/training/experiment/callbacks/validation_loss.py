# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-dataset validation loss callback for distributed training.

Evaluates each validation dataset independently and logs ``val/loss_{name}``
per dataset, plus an aggregate ``val/loss`` (mean of per-dataset losses).
Runs every ``eval_steps`` training steps via a ``TrainerCallback``.

All ranks participate in the forward passes; losses are all-reduced so that
every rank has the same aggregated result.  Only rank 0 logs to wandb.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import torch
import torch.distributed as dist
from transformers.trainer_callback import TrainerCallback

from groot.core.data.types import BaseProcessor, ShardedDataset

logger = logging.getLogger(__name__)


class ValidationLossCallback(TrainerCallback):
    """Callback that computes per-dataset validation loss during training.

    Args:
        val_datasets: Dict mapping embodiment tag -> ShardedDataset.
        processor: Data processor (toggled eval/train for augmentation control).
        data_collator: Collator to batch samples into model-ready dicts.
        eval_steps: Run validation every N training steps.
        val_total_samples: Number of samples to draw *per dataset*.
        eval_batch_size: Per-device batch size for val forward passes.
        val_loss_seed: Deterministic seed for shard sampling.
    """

    def __init__(
        self,
        val_datasets: dict[str, ShardedDataset],
        processor: BaseProcessor,
        data_collator,
        eval_steps: int,
        val_total_samples: int,
        eval_batch_size: int,
        val_loss_seed: int,
    ):
        self.val_datasets = val_datasets
        self.processor = processor
        self.data_collator = data_collator
        self.eval_steps = eval_steps
        self.val_total_samples = val_total_samples
        self.eval_batch_size = eval_batch_size
        self.val_loss_seed = val_loss_seed
        self.trainer = None  # Set after construction (e.g. callback.trainer = trainer)

    # ------------------------------------------------------------------
    # Shard planning (deterministic, rank-independent)
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_shards(
        dataset: ShardedDataset,
        total_samples: int,
        seed: int,
    ) -> tuple[list[int], int]:
        """Return (shard_indices, actual_total) for a single dataset.

        Shuffles shard indices deterministically, then accumulates shard
        lengths until ``total_samples`` is covered.  All ranks get the
        same plan.
        """
        rng = np.random.default_rng(seed)
        n_shards = len(dataset)
        all_ids = list(range(n_shards))
        rng.shuffle(all_ids)

        accumulated = 0
        selected: list[int] = []
        idx = 0
        while accumulated < total_samples:
            shard_id = all_ids[idx % n_shards]
            accumulated += dataset.get_shard_length(shard_id)
            selected.append(shard_id)
            idx += 1
            if idx >= n_shards and accumulated < total_samples:
                break

        return selected, min(accumulated, total_samples)

    # ------------------------------------------------------------------
    # Shard loading + strided yielding
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_strided_samples(
        dataset: ShardedDataset,
        shard_indices: list[int],
        total_samples: int,
        rank: int,
        world_size: int,
    ):
        """Yield this rank's strided slice of the dataset.

        All ranks load the same shards in the same order.  Rank *r*
        yields sample *i* only when ``i % world_size == r``.
        A background thread prefetches the next shard.
        """

        def _load_shard(shard_idx: int):
            return dataset.get_shard(shard_idx)

        global_idx = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load_shard, shard_indices[0])

            for i in range(len(shard_indices)):
                try:
                    shard_data = future.result()
                except Exception:
                    logger.warning("Failed to load val shard, skipping", exc_info=True)
                    shard_data = []

                if i + 1 < len(shard_indices):
                    future = executor.submit(_load_shard, shard_indices[i + 1])

                for sample in shard_data:
                    if global_idx >= total_samples:
                        return
                    if global_idx % world_size == rank:
                        yield sample
                    global_idx += 1

    # ------------------------------------------------------------------
    # Main callback entry point
    # ------------------------------------------------------------------

    def on_step_end(self, args, state, control, model=None, **kwargs):
        global_step = state.global_step
        if global_step == 0 or global_step % self.eval_steps != 0:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        device = next(model.parameters()).device

        total = self.val_total_samples

        model.eval()
        self.processor.eval()

        per_ds_losses: dict[str, float] = {}

        try:
            for name, dataset in self.val_datasets.items():
                shard_indices, actual_total = self._plan_shards(dataset, total, self.val_loss_seed)

                # Collect samples for this rank
                samples = list(
                    self._iter_strided_samples(
                        dataset, shard_indices, actual_total, rank, world_size
                    )
                )

                # Collate into batches and run forward passes
                # Track loss + any aux scalar losses the model returns
                loss_accumulators: dict[str, torch.Tensor] = {}
                batch_count = torch.tensor(0, device=device, dtype=torch.long)

                for batch_start in range(0, len(samples), self.eval_batch_size):
                    batch_samples = samples[batch_start : batch_start + self.eval_batch_size]
                    batch = self.data_collator(batch_samples)

                    # Move batch to device
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    with torch.no_grad():
                        outputs = model(**batch)

                    # Collect all scalar losses from outputs
                    out_dict = outputs if isinstance(outputs, dict) else vars(outputs)
                    for key, val in out_dict.items():
                        if isinstance(val, torch.Tensor) and val.numel() == 1:
                            if key not in loss_accumulators:
                                loss_accumulators[key] = torch.tensor(0.0, device=device)
                            loss_accumulators[key] += val.detach()

                    batch_count += 1

                # All-reduce across ranks
                if dist.is_initialized():
                    for acc in loss_accumulators.values():
                        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)

                if batch_count.item() > 0:
                    ds_metrics = {k: (v / batch_count).item() for k, v in loss_accumulators.items()}
                    per_ds_losses[name] = ds_metrics
                else:
                    logger.warning(f"No val batches for dataset '{name}', skipping")
        finally:
            model.train()
            self.processor.train()

        # Log via HF Trainer (routes to wandb, tensorboard, etc.)
        if per_ds_losses and self.trainer is not None:
            log_dict: dict[str, float] = {}

            # Use eval_ prefix so HF's WandbCallback rewrites to eval/ instead of train/
            # Per-dataset metrics: eval/val/{metric}_{embodiment_tag}
            for ds_name, metrics in per_ds_losses.items():
                for metric_key, metric_val in metrics.items():
                    log_dict[f"eval_val/{metric_key}_{ds_name}"] = metric_val

            # Aggregate across datasets: eval/val/{metric} = mean over datasets
            all_metric_keys = {k for m in per_ds_losses.values() for k in m}
            for metric_key in all_metric_keys:
                vals = [m[metric_key] for m in per_ds_losses.values() if metric_key in m]
                if vals:
                    log_dict[f"eval_val/{metric_key}"] = sum(vals) / len(vals)

            self.trainer.log(log_dict)
            logger.info(
                f"[step {global_step}] val metrics: "
                + ", ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
            )
