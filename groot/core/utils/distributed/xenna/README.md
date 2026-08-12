# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Xenna

A thin layer on [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html) for
managing groups of actors and multi-stage data pipelines.

You can always do everything Xenna does with raw Ray actors.
Xenna automates what's tedious: lifecycle, work distribution with backpressure, SPMD
coordination, and streaming data between stages.
Worker classes are plain Python — no base class, no decorators.

## Use Cases

**Data pipelines** — Download a large HuggingFace dataset, untar, transform, upload.
Each stage runs at its own scale with backpressure between stages.

**VLM data loading** — Stage-by-stage pipeline: remote storage → local buffer → extract
JPEG frames from MP4 → feed to training.

**SPMD inference** — Run vLLM or any tensor-parallel model across multiple GPUs.
Xenna handles NCCL rendezvous, rank assignment, and work-stealing dispatch across
replicas.

**RL hybrid flow** — Coordinate actor, critic, reward, and reference model workers.
WorkerGroup's `broadcast()` maps naturally to the
[verl hybrid flow](https://verl.readthedocs.io/en/latest/hybrid_flow.html) pattern.

## Quick Start

```python
from groot.core.utils.distributed import xenna

class Processor:
    def setup(self, worker_metadata):
        self.rank = worker_metadata.worker_index

    def process(self, item):
        return item * 2

with xenna.WorkerGroup(Processor, num_workers=4, resources=xenna.Resources(cpus=1)) as g:
    for result in g.map_unordered("process", range(100)):
        print(result)
```

## WorkerGroup

A managed group of Ray actors running the same class.

```python
with WorkerGroup(
    worker_cls,              # Any Python class
    num_workers=8,
    resources=Resources(gpus=1),
    worker_kwargs={"model": "my-model"},  # passed to worker_cls()
) as g:
    # Same call on ALL workers, returns list of results in worker order
    results = g.broadcast("train", step=0)

    # Distribute items across workers, yield in completion order
    for r in g.map_unordered("process", items):
        ...

    # Auto-batch items, then distribute
    for r in g.map_batched("process", items, batch_size=32):
        ...
```

If the worker class has a `setup()` method, it runs after actor creation.
It can optionally accept `worker_metadata` (a `WorkerMetadata` with `worker_index`,
`num_workers`, `node_id`, and SPMD fields).

## Pipeline

Chain WorkerGroups into a streaming pipeline.
All stages run concurrently via threads + bounded queues.
Backpressure propagates end-to-end.

```python
pipeline = xenna.Pipeline(stages=[
    xenna.StageSpec(Downloader, resources=xenna.Resources(cpus=1), num_workers=8),
    xenna.StageSpec(Extractor,  resources=xenna.Resources(cpus=2), num_workers=4),
    xenna.StageSpec(Captioner,  resources=xenna.Resources(gpus=1), num_workers=4),
])

for result in pipeline.run(input_data=urls):
    print(result)
```

Workers must have a `process(item) -> result` method (or `process(batch) -> list` with
`batch_size > 1`). Return `None` to drop an item; return a `list` to fan out.

## SPMD

For tensor-parallel models, set `Resources(gpus=N, spmd=True)`. Each logical worker
becomes N actors coordinated via NCCL. Xenna handles rank assignment, master address
discovery, and port allocation.
Only rank 0's result is returned.

```python
class VLLMWorker:
    def setup(self, worker_metadata):
        os.environ["RANK"] = str(worker_metadata.rank)
        os.environ["WORLD_SIZE"] = str(worker_metadata.world_size)
        os.environ["MASTER_ADDR"] = worker_metadata.master_addr
        os.environ["MASTER_PORT"] = str(worker_metadata.master_port)

        self.llm = vllm.LLM(
            model="my-model",
            tensor_parallel_size=worker_metadata.world_size,
            distributed_executor_backend="external_launcher",
        )

    def process(self, prompts):
        return [o.outputs[0].text for o in self.llm.generate(prompts)]

# 8 independent TP=4 replicas, work-stealing across all of them
with xenna.WorkerGroup(
    VLLMWorker, num_workers=8, resources=xenna.Resources(gpus=4, spmd=True),
) as g:
    for result in g.map_unordered("process", prompt_batches):
        print(result)
```

See `scripts/xenna_vllm_example.py` for a complete runnable example.

## API Summary

| Export | Description |
| --- | --- |
| `WorkerGroup` | Managed group of Ray actors with `broadcast`, `map_unordered`, `map_batched` |
| `Pipeline` | Streaming multi-stage pipeline with bounded queues |
| `StageSpec` | Stage config: worker class, resources, num_workers, batch_size |
| `Resources` | CPU/GPU requirements per worker, with `spmd` flag |
| `WorkerMetadata` | Passed to `setup()`: worker_index, num_workers, rank, world_size, master_addr/port |
| `NO_ITEM` | Sentinel for queue-backed iterators (signals "temporarily empty, don't block") |
