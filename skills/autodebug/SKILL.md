---
name: autodebug
description: Diagnose GR00T code, environment, dependency, configuration, data, preprocessing, training, inference, deployment, evaluation, and platform problems on user-granted local or remote compute; defend root-cause conclusions with recorded falsifiable checks; and verify an authorized downstream fix. Use for internal or OSS GR00T checkouts and releases when a failure is unclear or requires evidence from reports, logs, videos, configs, traces, metrics, saved inputs, or live execution.
---

# GR00T AutoDebug

Run the repository workflow; do not replace it with an improvised checklist.

## Mission and limits

Work only in checkouts and on compute targets the user grants. Never assume access to a
machine, credential, repository, model, dataset, or robot, and never store credentials in an
incident or target configuration. Apply the same workflow across GR00T versions and internal
or OSS code that is actually available; do not specialize the diagnosis to N2 or one N1.x
release unless the report does.

Prefer the smallest discriminating offline check. Use unit or CPU checks, saved inputs,
simulation, short smoke runs, logs, configs, or static inspection before expensive training.
Do not require a full training run or real-robot execution when a smaller check can answer the
question. Use expensive training or a robot only when necessary, explicitly authorized, and
bounded.

End diagnosis with exactly one evidence-backed result:

- **Verified root cause:** state the earliest failing boundary, defended cause, evidence,
  reasonable alternatives addressed, remaining uncertainty, and recommended action. The
  action may change code, parameters, data, dependencies, environment, deployment, or
  workflow.
- **Inconclusive:** state every meaningful check tried, what each established or rejected,
  remaining unknowns, and the single best next source of evidence when one exists.

Diagnosing a root cause does not mean a fix is solved. Reserve `resolved` for a downstream
implementation that passes the built-in frozen verification.

## Start or resume

Use the supplied report, or normalize the request with
[bug-report-template.md](references/bug-report-template.md). Required sections may say
`unknown`; do not invent values.

```bash
uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py new \
  --report <report.md> \
  --workspace <checkout-to-debug> \
  --target-config <targets.json>
```

`--target-config` is optional. Without it, the checkout is available as target `local`.
For multi-platform problems, the user grants access through local paths or authenticated
SSH aliases; never put credentials in the target file:

```json
{
  "version": 1,
  "targets": {
    "gpu-workstation": {
      "kind": "ssh",
      "host": "gpu-workstation.example.com",
      "workspace": "/workspace/project"
    },
    "edge-device": {
      "kind": "ssh",
      "host": "edge-device.example.com",
      "workspace": "/opt/project"
    }
  }
}
```

`new` automatically searches the published `.insights/` projection using the report's
problem and observed-behavior sections. The lookup is recorded in `incident.json` with
the projection hash and retrieved record IDs. Missing knowledge is a recorded grounding
gap, not a reason to stop diagnosis.

If the same report and workspace already have an incident, `new` resumes it without
repeating the intake lookup. Read the returned `next` action and follow it exactly. Use
`show` at any time:

```bash
uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py show \
  --incident <incident.json>
```

Inspect the granted targets and collect a read-only probe without advancing the incident:

```bash
uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py targets \
  --incident <incident.json>
uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py probe \
  --incident <incident.json> \
  --target <target-name> \
  --command '<focused command>'
```

For a platform-parity problem, run the same command, commit, checkpoint, input, and config
on every required target. Hash shared assets and compare the resulting logs or metrics with
a local acceptance command. That comparator becomes the ordinary frozen reproducer.

## Progress record and active monitoring

Use the runner-managed `incident.json` and its referenced logs and artifacts as the only
durable progress record. Run lifecycle actions through the runner so each command, outcome,
hash, phase, and artifact is recorded. Never hand-edit the incident or create a parallel
status file. Preserve complete output in the hashed per-run logs; use the compact `summary`
field on every recorded run to scan progress without replacing the raw evidence.

While a command or investigation is actively running, keep an observable execution session
alive and post a concise status report in the chat at least once every 10 minutes. Derive the
report from the incident and current session, and include:

- the current incident phase and status;
- new evidence recorded since the previous report;
- the command or hypothesis currently under test;
- the next check or one actionable blocker;
- the incident path and newest relevant log or artifact.

Report a phase change, failure, or completion immediately rather than waiting for the next
interval. Stop periodic reports when no work is running, the incident is waiting for user
input, blocked or terminal, or the user cancels. Do not detach a process and call it
monitored; use a pollable tool or execution session, and never claim background monitoring
unless that session is alive and its results continue to enter the runner ledger.

## Enforced loop

1. Run the smallest command that demonstrates the reported failure. The runner repeats
   it in a clean baseline worktree before accepting the reproduction. The failure
   signature must be text that distinguishes the reported bug from an infrastructure or
   harness error:

   ```bash
     uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py reproduce \
       --incident <incident.json> \
       --command '<smallest failing command>' \
       --failure-signature '<exact reported error or assertion>'
   ```

2. If the failure does not reproduce, the runner has already moved the incident to
   `waiting`. Query Insights using the exact failure, scope, and attempted fixes, then
   inspect the available artifacts, present qualified suggestions, and ask for the
   single most useful missing item. Do not call `pause` again:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py insights \
     --incident <incident.json> \
     --query '<symptom, version, embodiment, phase, and attempted fixes>'
   ```

   After the item arrives, resume the existing incident:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py resume \
     --incident <incident.json> \
     --note '<new evidence or corrected reproduction condition>'
   ```

   A waiting state is not a diagnosis or terminal result. Use `pause --need ...` only
   when an active diagnosis or verification step discovers a new human dependency.

3. Read [error-taxonomy.md](references/error-taxonomy.md). Find the earliest boundary
   where expected and observed behavior differ. Read only the related solved cases
   returned by the incident:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py cases \
     --incident <incident.json>
   ```

   Repeat the `insights` command with the refined boundary, exact symptom, version,
   embodiment, and attempts when that scope differs from the intake query. Read the
   recorded finding and paired chunk. Use Insights and solved cases only to prioritize
   discriminating checks. Neither source establishes the cause of this incident.

4. Test one hypothesis with one focused command. Keep all other factors fixed:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py diagnose \
     --incident <incident.json> \
     --hypothesis '<one falsifiable hypothesis>' \
     --outcome <supported|rejected|inconclusive> \
     --target <optional-target-name> \
     --command '<discriminating check>'
   ```

   A nonzero check cannot be recorded as supported. Use the command output, not a
   plausible narrative, to choose the outcome.

5. After a supported check, defend the candidate root cause automatically before reporting
   it or preparing a fix. Run a distinct check through `diagnose` and keep every unrelated
   factor fixed. Prefer one of these defenses:

   - change only the suspected causal factor and observe the predicted symptom change;
   - compare a matched positive and negative control;
   - replay the same saved input through the boundary on both sides of the suspected cause;
   - restore the causal factor and reproduce the symptom again when doing so is safe.

   Use the same revision, checkpoint, input, config, and target unless one is the factor under
   test. Repeat enough matched trials to distinguish a stochastic result. A second invocation
   of the same command is not an independent defense unless repeatability itself is the
   hypothesis. Record the defense with the same `diagnose` command and a hypothesis beginning
   `Root-cause defense for:`. If the defense is rejected, nonzero, or inconclusive, do not
   claim the cause; update priorities and continue the loop.

6. Report `verified root cause` only when the incident contains both the supported candidate
   and a distinct supported defense whose observed result matches the causal prediction.
   Cite the exact run logs and separate observed facts from inference. Otherwise report
   `inconclusive` with the compact run summaries, full-log links, rejected alternatives,
   remaining uncertainty, and the best next evidence. Do not call either outcome a verified
   fix.

7. Only when implementation is authorized, create an immutable fix package for a separate
   implementation agent:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py handoff \
     --incident <incident.json> \
     --analysis '<why the evidence supports this diagnosis and what remains unknown>' \
     --change '<one repository-backed change>' \
     --target '<diagnosis-workspace-relative file or directory allowed to change>' \
     --regression '<related check that must remain passing>' \
     --asset '<absolute path to a runner-owned check or fixture, when used>'
   ```

   State the taxonomy boundary, evidence, rejected alternatives, and remaining
   uncertainty in `--analysis`. Repeat `--target` or `--regression` only when necessary.
   Put runner-owned acceptance helpers beside `incident.json`, declare each one with
   `--asset`, reference it by absolute path, and do not let the implementation agent edit
   it. Declare every helper dependency too; the runner does not infer arbitrary dynamic
   imports or files opened only inside application logic. The runner freezes each declared
   file by SHA-256. The package contains the
   report, command and log ledger, hypothesis outcomes, supported diagnosis, proposed
   scope, and frozen verification plan.

8. Do not implement the change in the diagnosis checkout. Give the reported `fix_artifact`
   to a separate coding
   agent. It must use a different Git worktree, enter the same repository subdirectory as
   the diagnosis workspace, and commit the implementation. For an `oss/` diagnosis, pass
   `<separate-worktree>/oss` when resuming:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py resume \
     --incident <incident.json> \
     --note '<downstream implementation summary>' \
     --implementation-workspace <separate-worktree>/<same-subdirectory> \
     --implementation-ref <commit-sha>
   ```

9. Run the built-in verification automatically after the implementation commit returns:

   ```bash
   uv run --no-project --python 3.10 python skills/autodebug/scripts/autodebug.py verify \
     --incident <incident.json>
   ```

   The runner first confirms that the frozen failure still occurs in a clean worktree at the
   baseline commit. It then creates a fresh worktree at the implementation commit, runs the
   original check and every frozen regression, and writes an immutable verification result.
   It reports `resolved` only when the baseline fails with the expected signature, the
   implementation passes without that signature, and every regression passes. An ordinary
   fix failure returns to diagnosis; missing baseline evidence waits; an integrity failure
   blocks the decision. Follow the recorded next action without delegating verification to a
   second skill.

The runner alone marks an incident resolved. It does so only when the original command
fails with the frozen signature in a clean worktree at the diagnosis revision, a
downstream implementation commit was recorded, and the original plus every frozen
regression pass in a fresh clean worktree at that commit without emitting the failure
signature.

The diagnosis checkout must start with no tracked changes and remain unchanged. Put
temporary helper scripts or fixtures beside `incident.json`, and make them read the
checkout through their current working directory. AutoDebug writes its incident
directory; implementation belongs to the downstream agent in a separate worktree.
External file arguments used by the reproducer or regressions are automatically frozen by
SHA-256. Repository paths directly named by those commands become protected verification
paths and cannot be changed by the implementation. Declare other runner-owned files with
`--asset`. Verdict commands cannot name external directories; pass explicit files or a
runner-owned manifest so directory membership cannot change silently.

Use `pause --need '<one specific item and how to capture it>'` when human or external
evidence can unblock the next experiment. Use `block --reason '<constraint and exact
unblock condition>'` only when access, hardware, dependencies, or agreed scope prevent
all useful experiments. Both states are resumable; `blocked` does not mean universally
unfixable.

Use the repository's local training and evaluation entrypoints as leaf executors when a
check needs them. A launch command passing proves only that the process started; evidence
counts after the command reaches a result and records its logs, metrics, and artifacts.

The current resolved path requires a committed repository change. External calibration,
machine state, deployed package, or remote-only changes may be diagnosed and handed off,
but remain waiting or blocked until a future implementation-receipt adapter can freeze and
verify that non-repository state.

Solved cases prioritize the first test; they never prove the current cause. Follow
`skills/autodebug/infra/cases/README.md` when contributing a newly verified case.
Cases are not restricted to N2: matching symptoms and subsystems remain eligible across
GR00T versions, while an exact version match receives only a tie-breaking preference.
