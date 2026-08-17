# AutoDebug evaluation tasks

This file tracks representative problems used to evaluate the AutoDebug agent.
Each run should use one incident report, one isolated worktree, and explicit exit
criteria. The agent is evaluated on evidence quality, hypothesis testing, and
the usefulness of its handoff artifact—not on whether every issue can be fixed
without additional evidence.

## GitHub issues

| Order | Task | Initial test | Compute | Status |
| --- | --- | --- | --- | --- |
| 1 | [#424: action chunk consumption and control rate](https://github.com/NVIDIA/Isaac-GR00T/issues/424) | Reproduce the action scheduling behavior and test chunk length, execution horizon, and control-rate assumptions independently. | CPU | Ready |
| 2 | [#529: invalid Parquet or dataset integrity](https://github.com/NVIDIA/Isaac-GR00T/issues/529) | Identify the first invalid input and distinguish damaged data from loader or schema handling. | CPU | Ready |
| 3 | [#417: attention-mask model plumbing](https://github.com/NVIDIA/Isaac-GR00T/issues/417) | Reproduce the historical N1.5 code path and verify the smallest mask-handling correction with a regression test. | CPU or one L40 | Ready; historical version |
| 4 | [#473: NumPy, CPU, and CUDA device boundary](https://github.com/NVIDIA/Isaac-GR00T/issues/473) | Locate the first unexpected device or type conversion and test the boundary directly. | One L40 | Ready |
| 5 | [#426: headless SimplerEnv or SAPIEN startup hang](https://github.com/NVIDIA/Isaac-GR00T/issues/426) | Capture startup logs and isolate rendering, display, driver, and simulator initialization. | One L40 | Ready |
| 6 | [#712: workstation and Jetson Orin parity](https://github.com/NVIDIA/Isaac-GR00T/issues/712) | Replay identical inputs and compare intermediate outputs across the L40 workstation and Orin until the first material divergence. | One L40 and one Orin | In progress |

Issue #282 is retained as a triage and FAQ check, not an execution benchmark,
because the reported path became obsolete after later refactoring.

## Internal customer task

### Geely: relative EEF training and open-loop evaluation error

**Report:** A model trained with relative end-effector actions produces an error
during open-loop evaluation. Geely has shared what they already tried. Those
attempts must be copied into the incident report verbatim—with commands,
parameter changes, and observed outcomes—before the agent proposes duplicate
experiments. The exact attempt list is not yet recorded in this repository.

**Evidence to attach:**

- training and evaluation configs and commands;
- code version, checkpoint, embodiment, and action representation;
- normalization statistics and action-dimension ordering;
- one fixed evaluation episode with observations, target actions, predictions,
  decoded actions, logs, and metrics;
- Geely's attempted fixes and the result of each attempt.

**First diagnostic test:** Trace one fixed episode through the training target
construction and the open-loop evaluation decoder. Compare coordinate frame,
units, sign, action ordering, relative-action accumulation, normalization, and
time alignment at each boundary.

**Exit criteria:**

- **Verified fix:** the original failure is reproduced, one causal change
  removes it, the fixed-episode comparison is within a declared tolerance, and
  frozen regression checks pass;
- **Needs evidence:** a required artifact or platform is unavailable, and the
  handoff names the exact missing item and the next discriminating test;
- **Unsupported:** tested hypotheses are rejected and the remaining explanation
  is clearly separated from verified findings.

Start on CPU for config and data-path checks. Use one L40 only when checkpoint
inference is required.
