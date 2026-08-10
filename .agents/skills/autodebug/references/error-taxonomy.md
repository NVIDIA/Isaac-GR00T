# GR00T error taxonomy

Use this file to choose what to test, not to assign a cause from the symptom alone.
Locate the earliest boundary where expected and observed behavior diverge, then test one
factor within that boundary.

```text
request -> environment -> data -> preprocessing -> model/training -> policy output
        -> action decoding -> transport/timing -> controller/robot -> task outcome
        -> evaluation
```

| Boundary | Common factor families | Smallest useful checks |
|---|---|---|
| Request and environment | wrong commit, missing package, architecture, device, stale cache, path, permissions | capture commit and resolved command; import/config preflight; compare package and hardware manifests |
| Data and labels | schema, missing keys, synchronization, episode boundaries, action/state alignment, corrupt or repeated video | validate metadata; inspect one episode; hash consecutive frames; align timestamps and dimensions |
| Preprocessing | resize/padding, crop, normalization, camera ordering, train/deploy mismatch | save one raw input and final model tensor; compare shapes, ranges, keys, and transforms |
| Model and training | wrong checkpoint, frozen modules, unequal compute, under/overfitting, optimizer/config drift | resolve the effective config; compare trainable parameters and curves; run a controlled small overfit |
| Policy output and action decoding | representation, units, sign, frame, horizon, gripper convention, chunk stitching | compare raw and decoded actions for one saved input; plot boundary jumps; round-trip conversions |
| Transport and timing | stale frames/actions, queues, clock mismatch, serialization, inference latency | add correlation IDs; measure frame/action age; compare timestamps at every hop |
| Controller and robot | tracking delay, saturation, limits, safety filter, calibration, dynamics | compare command with measured state; replay a known smooth command; inspect clipping and limits |
| Task and evaluation | ambiguous success oracle, early termination, seed, task distribution, too few trials | freeze task/seed/oracle; inspect per-phase and per-task results; rerun matched trials |

For premature gripper closure, lag, or zigzag motion, do not start by tuning the model.
First align the video, policy input, raw output, decoded command, controller command, and
measured state on one timeline. The first signal that changes identifies the next boundary
to investigate.
