# Real-robot items that remain hardware-gated

The rollout and Speed-RL software parameters are implemented and tested. The following values and
acceptance gates cannot be inferred from code or simulation and must remain explicit deployment
inputs.

## Certified limits

- Per-component measured TCP twist thresholds for both arms:
  `vx, vy, vz, wx, wy, wz`.
- Robot-certified Cartesian velocity and acceleration limits for plain TOPPRA.
- Confirmation that the configured `0.7, 1.0, 1.3, 1.6` Speed-RL levels remain inside the robot,
  workspace, payload and task-specific limits.
- Independent robot watchdog, emergency stop, joint, collision, singularity and torque limits.

The software speed-violation label does not stop the robot and is not a substitute for these
hardware protections.

## Required real-robot gates

- Eight synchronous calibration episodes: two valid episodes at each speed, with explicit tracking
  review before the next speed is opened.
- Plain asynchronous rollout verification before asynchronous Speed-RL is enabled.
- Greedy asynchronous Speed-RL verification before asynchronous online training is enabled.
- Five final greedy episodes meeting the configured task success, abort, violation and duration
  acceptance criteria.

## Deferred reward work

The first implementation uses the confirmed safe-success reward:

```text
reward = safe_success * speed_scale^2
```

Task-time shaping, negative failure rewards and any duration-aware discount remain deferred until
their units and hardware acceptance criteria are agreed. They must not be introduced into an
existing checkpoint without an explicit reward-contract version change.
