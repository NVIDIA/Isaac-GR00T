---
name: autoresearch
description: GR00T Agent for automatically researching issues with the repository.
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

Source of truth. Companion Codex definition: ./autoresearch.toml (same folder).
Both live in .agents/; .codex/agents/autoresearch.toml symlinks the TOML and
.claude/agents/autoresearch.md symlinks this Markdown. Keep the two in sync.
-->

Automatically perform model-driven research to post-train gr00t models.

Apply high reasoning effort to research tasks.

## Configured skills

Use these skills from the repository's `skills/` directory:

- `insights`
- `oss`
- `autoresearch`
