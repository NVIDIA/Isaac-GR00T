---
name: post-train
description: Trigger OSS-only post-training pipelines that run from the isaac-groot-n2 package root, optionally followed by OSS evaluation in the same CI pipeline.
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
# Post-Train Skill

Use this skill when the user asks to trigger or inspect an OSS post-train run,
especially the `n2-minwam-sim` H100 training run. If the user asks for
post-train and evaluation together, pass `--eval` so both jobs run in
one GitLab pipeline and the evaluation consumes the produced checkpoint.

## Workflow

1. Run all commands from the current worktree.
2. Do not edit or invoke the `golden-run` skill for post-train work.
3. Trigger the dedicated post-train CI pipeline through this script:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py [n2-minwam-sim-h100] [priority]
```

To run post-train followed by evaluation in the same pipeline:

```bash
uv run python .codex/skills/post-train/scripts/post_train.py n2-minwam-sim-h100 high --eval
```

Cluster readiness waits up to 3600 seconds by default. For a slower cluster,
set an optional local override before triggering; the script forwards it to the
pipeline:

```bash
export POST_TRAIN_CLUSTER_WAIT_TIMEOUT=5400
```

The default run is the first configured variant in the OSS package config at `groot/infra/ci/golden_runs.yaml`, currently `n2-minwam-sim` on H100. The script sets `POST_TRAIN_E2E_LEAF=true`, which selects `staging.posttrain.h100`.
With `--eval`, it also sets `OPEN_LOOP_EVAL_E2E_LEAF=true` and
`OPEN_LOOP_EVAL_AFTER_POST_TRAIN=true`, which selects
`staging.evaluation_openloop.h100` after post-train.

## Required Environment

Set these before triggering locally if GitLab CI variables are not already present:

```bash
export GITLAB_HOST=gitlab-master.nvidia.com
export GITLAB_PROJECT_PATH=gr00t-release/isaac-gr00t-n2
export GITLAB_TOKEN=<token>
```

If a token is needed, first check `~/.credentials`.
