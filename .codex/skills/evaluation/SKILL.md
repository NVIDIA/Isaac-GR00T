---
name: evaluation
description: Trigger OSS evaluation CI for an existing post-training checkpoint, or trigger post-training and evaluation together in one GitLab pipeline.
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
# Evaluation Skill

Use this skill when the user asks to run OSS evaluation, evaluate an
existing checkpoint, evaluate a post-train checkpoint, or run post-train followed
by evaluation.

## Workflow

1. Run all commands from the current OSS package worktree.
2. For an existing checkpoint, trigger the dedicated evaluation CI pipeline:

```bash
uv run python .codex/skills/evaluation/scripts/evaluation.py n2-minwam-sim-h100 high --checkpoint <hf-uri-or-shared-path>
```

3. To produce a checkpoint and evaluate it in the same GitLab pipeline:

```bash
uv run python .codex/skills/evaluation/scripts/evaluation.py n2-minwam-sim-h100 high --post-train
```

The default run is the first configured OSS variant in
`groot/infra/ci/golden_runs.yaml`, currently `n2-minwam-sim` on H100. The script
sets `OPEN_LOOP_EVAL_E2E_LEAF=true`, which selects
`staging.evaluation_openloop.h100`. With `--post-train`, it also sets
`POST_TRAIN_E2E_LEAF=true`, so the eval job consumes the post-train dotenv
artifact `FINAL_CHECKPOINT_PATH`.

## Required Environment

Set these before triggering locally if GitLab CI variables are not already
present:

```bash
export GITLAB_HOST=gitlab-master.nvidia.com
export GITLAB_PROJECT_PATH=gr00t-release/isaac-gr00t-n2
export GITLAB_TOKEN=<token>
```

If a token is needed, first check `~/.credentials`.
