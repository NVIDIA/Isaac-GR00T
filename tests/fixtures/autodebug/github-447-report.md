# GR00T bug report

## Problem

An explicit ZMQ policy-server host is ignored on the supplied GR00T revision. The case
is open for diagnosis.

## Expected behavior

Starting the policy server with host `127.0.0.1` and port `5555` uses
`tcp://127.0.0.1:5555`. Omitting the host keeps the existing wildcard default.

## Observed behavior

The server uses the wildcard/all-interface endpoint even when an explicit host is
requested.

## Reproduction

- Repository: `NVIDIA/Isaac-GR00T`
- Revision: `94bc87c58d8357d3023e95088b3aa0f02aa47625`
- Scope: the surviving core policy-server API; the CLI named in the original report is
  absent from this revision
- Frequency: every invocation with an explicit host
- Smallest known failing case: from the checkout root, run
  `python3 /path/to/gr00t/oss/tests/fixtures/autodebug/github_447_acceptance.py`

## Acceptance check

A solution-independent CPU regression in
`oss/tests/fixtures/autodebug/github_447_acceptance.py` must fail before the change and pass
after it. It
must prove both that an explicit host produces `tcp://127.0.0.1:5555` and that the
default still produces `tcp://*:5555`, while preserving port and token behavior.

## Environment

- GR00T revision: `94bc87c58d8357d3023e95088b3aa0f02aa47625`
- Robot, simulator, model weights, dataset, and network peer: not required
- Compute needed: local CPU, under one minute

## Evidence

- Public symptom report: https://github.com/NVIDIA/Isaac-GR00T/issues/447
- Do not browse the issue discussion or later commits during the held-out run; they
  disclose the proposed solution

## Attempts

No solution-bearing attempt is supplied to the debugging agent.

## Ownership and status

- Reporter/contact: public GitHub reporter
- Current status: open for held-out evaluation
- Missing evidence or access: none for the core CPU regression
