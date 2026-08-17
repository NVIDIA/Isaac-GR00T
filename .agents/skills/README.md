# GR00T Agentic Workflows and Skills

> [!WARNING]
> The agentic workflows in this repository are officially supported only with
> **Codex, Claude Code, and Cursor**. Other coding agents may be able to read the
> instructions, but are not officially supported or tested.

Run the agent from the OSS package root so it discovers the repository
guidance, agent definitions, and skills. In Codex, name a skill with
`$skill-name`. In Claude Code, use the corresponding `/skill-name` command. In
Cursor, ask the agent to use the named repository skill. Natural-language
requests also work when they clearly match a skill's purpose.

## Contents

- [Choose a workflow](#choose-a-workflow)
- [Agents versus skills](#agents-versus-skills)
- [Onboarding, training, evaluation, and serving](#onboarding-training-evaluation-and-serving)
- [AutoResearch Agent](#autoresearch)
  - [AutoResearch visualization](#autoresearch-visualization)
- [AutoDebug Agent](#autodebug)
- [FAQ Agent](#grounded-faq-and-insights)
- [How the pieces work together](#how-the-pieces-work-together)

## Choose a workflow

| What you want to do | What to call | Type | What to provide | Sample prompt |
| --- | --- | --- | --- | --- |
| Set up GR00T; train, evaluate, or serve a policy locally | [`onboarding`](onboarding/SKILL.md) | Skill | Local paths for data, caches, and outputs; Hugging Face access; GPU details | `Use the onboarding skill to validate my setup, then dry-run training on <dataset>.` |
| Automatically search for a better post-training recipe | [`autoresearch`](../.agents/autoresearch.md) | Agent | An `autoresearch.yaml`, working train/eval wrappers, and an existing baseline result | `Use the autoresearch agent to validate <config> and summarize the plan without launching.` |
| Render an AutoResearch study | [`visualize-autoresearch`](visualize-autoresearch/SKILL.md) | Skill | The study's `state.json` | `Use the visualize-autoresearch skill to render <state.json>.` |
| Diagnose a failure and defend the root cause with evidence | [`autodebug`](../.agents/autodebug.md) | Agent | Bug report, clean checkout, available evidence, and any granted compute targets | `Use the autodebug agent to investigate <report> without implementing a fix.` |
| Ask a grounded product, support, or documentation question | [`faq`](../.agents/faq.md) | Agent | The question and any relevant version or workflow context | `Use the faq agent to answer <question> from repository docs and published Insights.` |

## Agents versus skills

An **agent** is a task-focused bundle of instructions and enabled skills. Use
one when the goal spans a complete workflow and you want the repository to
select and coordinate its supporting skills:

- [`autoresearch`](../.agents/autoresearch.md) combines Insights, local OSS
  execution, and the AutoResearch controller to improve a post-training recipe.
- [`autodebug`](../.agents/autodebug.md) combines Insights with the recorded
  AutoDebug evidence and verification loop.
- [`faq`](../.agents/faq.md) combines repository documentation with Insights to
  produce concise, source-grounded answers.

A **skill** is a reusable, narrower procedure. Name a skill directly when you
need only that operation, such as rendering a study, dry-running training, or
inspecting an existing AutoDebug incident.

**Insights** is a local, read-only knowledge graph shipped in this repository's
`.insights/` directory. All three repository agents access it through the
internal [`insights` skill](insights/SKILL.md) to retrieve prior findings and
experiment learnings. Insights supplies prior knowledge to help prioritize an
answer, experiment, or diagnostic check; it does not replace current code, live
measurements, or the evidence required to verify a root cause.

To use a full agent, ask the harness to use the repository agent by name and
state the outcome rather than prescribing each skill:

```text
Use the autoresearch agent to validate this study, launch it only after I
approve the execution plan, and report the best trial or first failure.
```

```text
Use the autodebug agent to investigate this report and return a defended root
cause or an inconclusive evidence record.
```

```text
Use the faq agent to answer this question from repository documentation and
published Insights, citing the supporting sources.
```

## Onboarding, training, evaluation, and serving

Use the `onboarding` skill for the supported local GR00T workflow. It starts with a
dry run when setup is unclear and uses `python -m groot.infra.oss` for training,
open-loop evaluation, combined train/eval, and policy serving. It does not
substitute internal GitLab, OSMO, or golden-run infrastructure.

Example request:

```text
Use the onboarding skill to dry-run post-training on /data/my_lerobot_dataset with
embodiment tag <tag>, then run a two-step smoke test and report the checkpoint.
```

See the [minimal end-to-end example](../docs/minimal_example.md) for the manual
equivalent and exact installation requirements.

## AutoResearch

Use AutoResearch when a working baseline already exists and you want the agent
to propose, train, evaluate, and compare focused recipe changes. The user's
train and evaluation scripts own the compute backend; they may run locally or
submit to remote infrastructure. `autoresearch.yaml` is the only study
configuration interface, and `state.json` is the durable progress record.

Start by asking the agent to validate the study. Launch only after reviewing the
resolved output directory, trial budget, parallelism, baseline metric, and
scripts that will execute:

```text
Use the autoresearch agent to validate /absolute/path/to/autoresearch.yaml.
If preflight passes, summarize the execution plan but do not launch yet.
```

Then launch or resume explicitly:

```text
Use the autoresearch agent to run or resume
/absolute/path/to/autoresearch.yaml. Monitor progress and report the first
actionable failure.
```

The [AutoResearch guide](autoresearch/README.md) explains baselines, wrapper
contracts, study configuration, trials, waves, and included examples.

### AutoResearch visualization

Use `visualize-autoresearch` to turn a study's `state.json` into a portable,
self-contained HTML report without modifying study state:

```text
Use the visualize-autoresearch skill to render
/absolute/path/to/output_dir/state.json and return the HTML path.
```

The report shows trial lineage, scores, improvements and regressions, active or
failed stages, and per-trial details.

## AutoDebug

AutoDebug runs a recorded evidence loop: reproduce the exact problem, identify
the first failing boundary, test one falsifiable explanation at a time, and
challenge a candidate cause with a distinct control. It reports either a
verified root cause or an inconclusive evidence record. When implementation is
authorized, a separate coding agent applies the fix and AutoDebug independently
verifies the resulting commit.

Prepare a report from the
[bug report template](autodebug/references/bug-report-template.md), then ask:

```text
Use the autodebug agent with /absolute/path/to/report.md. Debug
/absolute/path/to/clean/gr00t/oss on my granted compute, defend any root-cause
claim with a distinct recorded check, and do not implement a fix unless I
authorize it.
```

The full [AutoDebug guide](../.agents/autodebug.md) documents its incident
record, compute targets, outcomes, and optional fix-verification flow.

## Grounded FAQ and Insights

Use the `faq` agent for product, support, and documentation questions. It can
search the internal, read-only Insights projection for prior research results,
debugging clues, experiment learnings, and support knowledge, then ground its
answer in repository documentation and returned records. Insights are
directional prior knowledge; current code and live evidence remain the source
of truth for present behavior.

```text
Use the faq agent to answer <exact question> from repository documentation and
published Insights. Preserve the relevant model, task, version, configuration,
and date, and cite the supporting sources.
```

AutoResearch and AutoDebug may also use Insights internally to prioritize
experiments or diagnostic checks. Users do not need to invoke the internal
skill directly.

### If Insights returns weak matches

If the FAQ agent reports weak Insights matches, retry with a scoped question
that names the distinctive symptom or decision plus the relevant version,
embodiment, and phase. The retriever currently relies on literal token matching,
so a long sentence with many generic words can dilute the useful terms, and
different word forms or synonyms may not match.

If the first results are irrelevant:

1. Retry with two to four distinctive keywords from the failure, metric, model,
   or embodiment.
2. Try the terminology used in logs or configuration, including alternate word
   forms or likely synonyms.
3. Add one scope term at a time to distinguish otherwise similar records.
4. Report the query and projection hash when no grounded result remains.

The FAQ agent continues to use the deterministic Insights search. It does not
replace it by scanning or loading the full `.insights/` projection, and it does
not infer an answer from a weak match.

## How the pieces work together

- Insights can prioritize hypotheses for AutoResearch or AutoDebug, but it does
  not prove current behavior or a new incident's root cause.
- AutoResearch records every candidate and result in `state.json`;
  `visualize-autoresearch` renders that record without changing it.
- AutoDebug may prepare an immutable fix package, but diagnosis and
  implementation remain separate so the fix can be independently verified.
- The `onboarding` skill is the execution surface for supported local onboarding,
  training, evaluation, and policy-serving workflows.

The canonical definitions live in this directory and under
[`../.agents/`](../.agents/). Read the linked `SKILL.md` or agent file when you
need the complete operational contract, safety boundaries, or direct CLI
equivalent.
