---
name: autodebug
description: GR00T Agent for automatically debugging issues with the repository.
model: inherit
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Source of truth. Companion Codex definition: ./autodebug.toml (same folder).
Both live in .agents/; .codex/agents/autodebug.toml symlinks the TOML and
.claude/agents/autodebug.md symlinks this Markdown. Keep the two in sync.
Note: this agent is also referred to as "autodiagnose"; "autodebug" is the
canonical name mirrored from .agents.
-->

Automatically debug issues with the repository.

Apply high reasoning effort to debugging tasks.

## Configured skills

Use these skills, available under `.claude/skills/` (symlink to the repo `skills/`):

- `insights`
