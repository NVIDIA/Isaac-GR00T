---
name: ray
description: Launch and reason about GR00T Ray clusters backed by OSMO using `groot.infra.job`. Use when the user wants to create a Ray cluster for training workflows, inspect the generated workflow, or get the workflow id needed by future launchers.
---
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

# ray - OSMO Ray Cluster Launching

## Codex Notes

- Treat slash-command examples in this document as shorthand for the equivalent
  natural-language user request.
- Prefer `groot.infra.job` for Ray cluster creation when the user asks for the
  package-backed launcher.
- Only launch real clusters when the user clearly asks to create or start one.
  For planning, dry runs, or reviews, inspect/render configuration without
  submitting an OSMO workflow.
- If the user asks for legacy pool operations, existing workflow debugging, or
  Ray job submission against an already-running workflow, use the internal
  `gear-ray` skill instead.

## Intent Routing

- If the user wants a Ray cluster created by the package launcher, use
  `groot.infra.job`.
- If the user mentions a workflow id, treat it as an OSMO-backed Ray cluster id.
- If the user needs to submit a Ray job to an existing workflow, use `gear ray`
  job commands or a Ray dashboard port-forward as appropriate.

## Start a Ray Cluster

Run from the repo root or package root, depending on the active checkout.

```bash
uv run python -m groot.infra.job start-cluster \
  --cluster-name n2-minwam-backpack \
  --pool <osmo-pool-name> \
  --gpu-type gb200 \
  --num-nodes 2 \
  --image rayproject/ray:2.48.0-py310 \
  --priority high \
  --detach
```

The command prints:

- `Cluster ID: <workflow-id>` for future training launchers.
- `Workflow ID: <workflow-id>` for OSMO inspection.
- Follow-up commands for status, logs, dashboard forwarding, and cancellation.

Use `--detach` or `--no-wait` for asynchronous submission. Without it, the
launcher waits for the workflow to reach `RUNNING` and for the Ray dashboard on
the head task to answer.

Use `--pool` for the target OSMO pool name. Use `--gpu-type gb200`, `--gpu-type
h100`, or `--gpu-type l40` for the default GPU count, and pass `--gpus`,
`--cpus`, `--memory-gib`, or `--storage-gib` when the target pool needs explicit
resource overrides. Use `--template <path-or-name>` to supply a custom OSMO
workflow template. Use `--set NAME=VALUE` for template-specific overrides that
are not first-class CLI flags.

## Render or Inspect Without Launching

For reviews and tests, import the package and render the OSMO workflow config.
This calls OSMO dry-run rendering but does not submit the workflow.

```python
from groot.infra.job import ClusterSpec, RayCluster

config = RayCluster(
    ClusterSpec(
        cluster_name="n2-minwam-backpack",
        pool="<osmo-pool-name>",
        gpu_type="gb200",
        num_nodes=2,
        image="rayproject/ray:2.48.0-py310",
    )
).render_config()
```

The generated template starts a Ray head on the `master` task, workers connect
to that head through OSMO task host resolution, and the Ray dashboard is exposed
on port `8265`.

## Validation

For package changes, run:

```bash
uv run pytest tests/groot/infra/job
uv run ruff check groot/infra/job tests/groot/infra/job
python /home/agent007/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/ray
```
