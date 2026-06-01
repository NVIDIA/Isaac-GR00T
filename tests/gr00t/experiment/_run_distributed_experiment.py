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

"""torchrun entry point used by the multi-GPU experiment pytest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


EMBODIMENT_TAG = "libero_sim"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-gpus", required=True, type=int)
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": EMBODIMENT_TAG,
                    }
                ],
                "video_backend": "torchcodec",
                "shard_size": 64,
                "num_shards_per_epoch": args.num_gpus,
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

    config.training.start_from_checkpoint = str(args.model_path)
    config.training.skip_weight_loading = True
    config.training.output_dir = str(args.output_dir)
    config.training.max_steps = 1
    config.training.save_steps = 1
    config.training.save_total_limit = 1
    config.training.global_batch_size = args.num_gpus
    config.training.num_gpus = args.num_gpus
    config.training.dataloader_num_workers = 0
    config.training.use_wandb = False
    config.training.optim = "adamw_torch"
    config.training.bf16 = True
    config.training.tf32 = True
    config.training.fp16 = False
    config.training.gradient_checkpointing = False
    config.training.use_ddp = True
    config.training.eval_strategy = "no"
    config.training.save_only_model = True

    return config


def _write_rank_metadata(output_dir: Path) -> None:
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    rank_dir = output_dir / "distributed_rank_metadata"
    rank_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "current_cuda_device": torch.cuda.current_device(),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (rank_dir / f"rank_{rank}.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = _parse_args()

    visible_cuda_device_count = torch.cuda.device_count()
    if visible_cuda_device_count != args.num_gpus:
        raise RuntimeError(
            f"Expected {args.num_gpus} visible CUDA devices, got {visible_cuda_device_count}"
        )

    from gr00t.experiment.experiment import run

    run(_build_config(args))

    tensor = torch.tensor(
        [dist.get_rank() + 1],
        device=f"cuda:{torch.cuda.current_device()}",
    )
    dist.all_reduce(tensor)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
    if tensor.item() != expected:
        raise RuntimeError(f"Unexpected all_reduce result: {tensor.item()} != {expected}")

    _write_rank_metadata(args.output_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
