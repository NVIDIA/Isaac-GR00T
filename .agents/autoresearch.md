---
name: autoresearch
description: GR00T Agent for automatically researching issues with the repository.
model: inherit
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.

Source of truth. Companion Codex definition: ./autoresearch.toml (same folder).
Both live in .agents/; .codex/agents/autoresearch.toml symlinks the TOML and
.claude/agents/autoresearch.md symlinks this Markdown. Keep the two in sync.
-->

Automatically perform model-driven research to post-train gr00t models.

Apply high reasoning effort to research tasks.

## Configured skills

Use these skills from the repository's `skills/` directory:

- `insights`
- `onboarding`
- `autoresearch`
