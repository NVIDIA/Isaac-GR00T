---
name: insights
description: Search and synthesize source-grounded findings from the repository's `.insights/` corpus for downstream GR00T agents. Use when an agent needs prior research results, experiment learnings, FAQ or support knowledge, debugging clues, cross-source evidence, or a structured evidence handoff before answering, planning, debugging, post-training, or evaluation.
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
-->

# Insights Retrieval

Retrieve relevant evidence from `.insights/`, preserve its scope and status,
and return a concise, cited handoff that another agent can safely use.

## Grounding Rules

- Treat `.insights/<source>/` as a collection of source-specific, derived
  knowledge. Discover sources dynamically; do not assume only known folders
  such as `autoresearch` or `faq` exist.
- Use the directory name as provenance, not as a guarantee of authority.
- Treat an insight as evidence only for the model, task, dataset, metric,
  configuration, and time or iteration described in its surrounding context.
- Keep direct results separate from cross-task priors, hypotheses, pending
  work, branch pointers, and infrastructure failures.
- Never turn `submitted`, `running`, missing-score, or resumable/fatal failure
  entries into success or regression claims.
- Never treat an infrastructure failure as evidence that the underlying idea
  failed. Report it as an execution constraint.
- Do not merge results across variants or metrics as though they were directly
  comparable. Label cross-scope evidence as directional.
- Resolve what labels such as `worked`, `improved`, or `regressed` compare
  against: baseline, previous run, or current best. If the reference is not
  explicit, preserve the source label and score without inventing one; a
  regression from the best can still be above baseline.
- Prefer explicit dates, iterations, scores, and status fields over filename
  ordering or filesystem modification time when judging freshness.
- Use repository code or documentation as the current source of truth for
  present behavior. Use insights for learned context and historical evidence;
  flag any disagreement with current sources.
- Cite every material factual claim. Do not fill gaps with plausible details.
- Search and report by default. Modify `.insights/` only when the user
  explicitly asks to create or update insight artifacts.

## Retrieval Workflow

### 1. Establish the request

Extract the subject, decision, and required level of detail. Capture any
provided scope dimensions:

- model or family;
- task, embodiment, dataset, or benchmark;
- training, inference, evaluation, debugging, or FAQ topic;
- configuration, ablation, metric, status, or date range.

If a scope dimension is absent, search broadly and expose the ambiguity in the
handoff instead of silently choosing one variant.

### 2. Inventory the corpus

Run from the repository root. List the current corpus rather than relying on a
hard-coded source map:

```bash
rg --files -uu .insights | sort
```

If `.insights/` is absent or contains no readable files, stop and return a
grounding gap. Do not manufacture an answer from the skill text.

Inspect filenames and document structure before loading many files:

```bash
rg -n '^#{1,6} ' .insights
```

### 3. Search in passes

Build a small query set containing exact identifiers plus aliases or nearby
concepts. For example, search both an experiment ID and its model/task name, or
both `history dropout` and `history crop` when the decision concerns visual
history.

Use literal, case-insensitive searches for user-provided terms so punctuation
is not interpreted as a regular expression:

```bash
rg -n -i -F -C 3 -- '<term>' .insights
```

Repeat for the most discriminating terms. Search filenames, headings, and body
text. Start exact, then broaden to aliases, components, failure modes, and
outcomes. Search across all source folders unless the user explicitly limits
the source.

Avoid using the number of keyword matches as a relevance score. Rank candidates
by:

1. exact scope match;
2. explicit measured or documented outcome;
3. complete context and traceable provenance;
4. relevance to the requested decision;
5. explicit freshness.

### 4. Read context and classify evidence

Read the complete matching section and enough document context to capture the
title, scope, qualifiers, baseline, metric, status, and provenance. For short
documents, read the whole file. Obtain stable line references with:

```bash
nl -ba <matched-file>
```

Classify each candidate before synthesizing it:

- **Direct result:** A completed, scoped observation with an explicit outcome.
- **Directional prior:** Relevant evidence from another task, model, or metric.
- **Hypothesis or recommendation:** A proposed interpretation or next step,
  not yet established by the corpus.
- **Negative result:** A completed, scoped regression or disproven expectation.
- **Operational constraint:** An infrastructure, timeout, data, or tooling
  problem that affected execution.
- **Pending evidence:** Submitted, running, incomplete, or scoreless work.

Preserve source vocabulary when it carries status meaning, but paraphrase
instead of copying long passages.

### 5. Reconcile sources

Group compatible evidence and keep incompatible scopes separate. When sources
agree, say what exactly agrees and under which shared scope. When they conflict:

- report both findings with citations;
- identify differences in model, task, metric, configuration, or time;
- avoid selecting a winner unless a source-of-truth rule or stronger direct
  evidence justifies it;
- state what additional evidence would resolve the conflict.

Assign confidence to the synthesized claim, not to the source as a whole:

- **High:** Multiple compatible direct results, or one authoritative current
  source with unambiguous evidence.
- **Medium:** One direct, well-scoped result.
- **Low:** Only indirect, cross-scope, stale, incomplete, or ambiguous evidence.

Explain the confidence in a short phrase.

## Downstream Handoff

Return the smallest useful structure for the receiving agent. Use this fuller
contract when the result will feed another agent or a nontrivial decision:

```markdown
## Insight handoff

- Question: <the question or decision searched>
- Finding: <direct answer; distinguish evidence from inference>
- Confidence: <high|medium|low> — <brief reason>
- Scope: <model/task/dataset/metric/configuration/time represented>

### Evidence
- <claim or outcome> — `.insights/<source>/<file>:<line>`

### Caveats and conflicts
- <directional-only evidence, incompatible results, staleness, or none>

### Unknowns
- <missing evidence or none>

### Suggested downstream use
- <how the receiving agent can safely apply the finding>
```

Include experiment IDs, scores, baselines, statuses, and branch or artifact
pointers only when relevant to the decision. Keep each citation adjacent to the
claim it supports. Use repository-relative paths so the handoff remains
portable across worktrees.

Adapt the final section to the consumer:

- **FAQ:** Lead with a concise answer, keep experimental claims qualified,
  cite the best source sections, and identify unanswered parts. Do not present
  a research prior as product behavior or policy.
- **AutoResearch:** Separate ideas to build on, negative results to avoid,
  operational failures, pending experiments, and remaining hypotheses.
- **AutoDebug:** Separate observed symptoms, known causes, prior fixes or
  regressions, and the next discriminating checks.
- **Post-training or evaluation:** Preserve exact variant, checkpoint, dataset,
  metric, and execution environment details; do not substitute nearby variants.

For a simple lookup, compress the handoff to a direct answer, one caveat if
needed, and the supporting citations.

## No-Answer Behavior

If no relevant grounded evidence exists, report:

1. the scope and terms searched;
2. the sources or files considered;
3. the missing fact or unresolved decision;
4. the best next source or owner to consult, if the repository identifies one.

Do not claim that the answer does not exist outside the searched corpus. Do not
invent a fallback contact, result, or recommendation.
