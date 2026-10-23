# Unified real-robot rollout

This is the single laptop-side entrypoint for synchronous/asynchronous TOPPRA and the independent
Speed-RL extension. The frozen rollout and Kion client remain unchanged. The rollout process is the
only 250 Hz owner of dual-arm Servo and hand commands; ObservationGUILite observes at 30 Hz and
labels/records episodes without driving the robot.

## Fresh-laptop deployment

```bash
git clone <YOUR_REPOSITORY_URL> Isaac-GR00T
git clone <OBSERVATION_GUI_REPOSITORY_URL> ObservationGUILite-v1.0.0
cd Isaac-GR00T
uv sync --all-extras
uv pip install --python .venv/bin/python --no-deps av==15.1.0 dearpygui==2.0.0
cp gr00t/eval/real_robot/TOPPRA/rollout/rollout.env.example rollout.local.env
```

Edit `rollout.local.env` once:

- set `ROS_SETUP` and `ROBOT_ROS_SETUP` so the selected Python can import ROS and Kion messages;
- keep the SSH tunnel endpoint as `POLICY_SERVER_HOST=127.0.0.1` and port `47866`;
- check all three camera topics against the robot;
- keep `TTS_SAMPLES=1` for normal synchronous rollout;
- keep `DRY_RUN=1` until observation, server, TOPPRA and labels have been checked.

If the server is reached through SSH, run this in a separate laptop terminal:

```bash
ssh -N -L 47866:127.0.0.1:47866 -p <SSH_PORT> <CLOUD_USER>@<CLOUD_HOST>
```

Then perform the dry-run preflight:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/robot_ws/devel/setup.bash
.venv/bin/python -c \
  "from gr00t.eval.real_robot.TOPPRA.kion_client.client import load_ros_types; print(load_ros_types())"
.venv/bin/python -m gr00t.eval.real_robot.TOPPRA.speed_rl.probe_server \
  --server-host 127.0.0.1 --server-port 47866 --requests 10 --warmup-requests 2
bash gr00t/eval/real_robot/TOPPRA/rollout/launch.sh rollout.local.env
```

### Pipeline-only test without wrist cameras

When the physical wrist cameras are absent, start the following explicit test relay in a separate
terminal after exporting the same ROS network variables as `rollout.local.env`:

```bash
cd /path/to/Isaac-GR00T
set -a
source rollout.local.env
set +a
source "$ROS_SETUP"
test -z "$ROBOT_ROS_SETUP" || source "$ROBOT_ROS_SETUP"
"$PYTHON_BIN" -m gr00t.eval.real_robot.TOPPRA.rollout.mock_inputs
```

It copies the head `CompressedImage` stream to both configured wrist-camera topics. Compensated
force continues to come exclusively from the robot's real `/force_compensation_node`. This only
verifies ROS observation, server inference, TOPPRA and episode-recording connectivity. The images
are not real wrist views, so never use them to judge task success. The relay publishes no force,
Servo or hand commands and refuses to start if a real publisher already owns either wrist-camera
topic. The real-motion client also detects this relay and refuses to create/configure Servo while it
is active.

For this test keep these settings:

```text
ROLLOUT_MODE=plain
INFERENCE_MODE=sync
TTS_SAMPLES=1
CONTROL_INTERFACE=both
DRY_RUN=1
DISABLE_PINCH=1
```

In a second terminal start `launch.sh`. It starts the rollout client and registers the GUI control
services, but deliberately does not start a Docker container. In a third terminal launch the GUI:

```bash
bash gr00t/eval/real_robot/TOPPRA/rollout/launch_gui.sh rollout.local.env
```

Use either the terminal or GUI to start the episode, wait for at least one inference and TOPPRA
plan, label it as failure, and quit the rollout client. Stop the GUI and mock relay with Ctrl-C.
`DRY_RUN=1` does not create a Servo publisher or call the Servo configuration services.

The plain client only needs action output. The Speed-RL probe additionally requires the exact
`last_block_pre_norm` feature contract from the updated server.

## Rollout modes

Start with these values in `rollout.local.env`:

```text
ROLLOUT_MODE=plain
INFERENCE_MODE=sync
TTS_SAMPLES=1
DRY_RUN=1
```

Synchronous rollout has one candidate and does not use multi-candidate TTS. For raw asynchronous
testing change only `INFERENCE_MODE=async`; it continues to use the frozen handoff/refill logic.
Set `MAX_LATENCY_S` at or above the measured end-to-end P99 before an async hardware run. TTS values
of 4/8 deliberately multiply candidate computation and payload size and are not normal rollout
defaults.

For Speed-RL use:

```text
ROLLOUT_MODE=speed-rl
INFERENCE_MODE=sync
SPEED_RL_PHASE=calibration
POLICY_CHUNK_HORIZON=40
TOPPRA_EXECUTION_HORIZON=30
TOPPRA_SPEED_MIN=0.7
TOPPRA_SPEED_MAX=1.6
TOPPRA_SPEED_STEP=0.3
ONLINE_EPISODES=100
LEFT_TWIST_THRESHOLDS=VX,VY,VZ,WX,WY,WZ
RIGHT_TWIST_THRESHOLDS=VX,VY,VZ,WX,WY,WZ
```

The measured-twist thresholds must come from hardware certification; the code has no guessed
defaults. Full phase and async gates are documented in `../speed_rl/README.md`.

Every mode additionally requires these certified, base-frame Cartesian gates before `DRY_RUN=0`:

```text
LEFT_WORKSPACE_BOUNDS=xmin,xmax,ymin,ymax,zmin,zmax
RIGHT_WORKSPACE_BOUNDS=xmin,xmax,ymin,ymax,zmin,zmax
MAX_TARGET_POSITION_ERROR_M=...
MAX_TARGET_ROTATION_ERROR_RAD=...
```

Targets are checked before every Servo publication. Leaving any value blank fails closed; values
must come from the robot/workcell owner rather than from software defaults.

For the SpeedTuning-style comparison backend, keep synchronous single-candidate inference and use:

```text
ROLLOUT_MODE=speed-rl-baseline
INFERENCE_MODE=sync
TTS_SAMPLES=1
BASELINE_ACTION_FREQUENCY=30
BASELINE_K_SKIP=10
BASELINE_SPEED_MIN=1.0
BASELINE_SPEED_MAX=4.0
BASELINE_SPEED_STEP=0.5
```

It uses separate `logs/kion_speed_rl_baseline/` state and updates absolute TCP targets at 30 Hz
through SE(3) interpolation while the sole Servo owner continues publishing at 250 Hz. Every fresh
40-action chunk executes exactly `k_skip=10` targets and then discards the remainder. It does not
provide TOPPRA velocity/acceleration guarantees; see `../speed_rl/INTERPOLATION_BASELINE.md` before
any hardware run.

## ObservationGUILite and episode labels

Keep the rollout client running with `CONTROL_INTERFACE=gui` or `both`, then launch:

```bash
bash gr00t/eval/real_robot/TOPPRA/rollout/launch_gui.sh rollout.local.env
```

The default native GUI runtime avoids a Docker/Python 3.13 download and uses `.venv`. Set
`OBSERVATION_GUI_RUNTIME=docker` only when the container registries and Astral/GitHub downloads are
reachable. The GUI overlay creates no action-controller or Servo publisher.

The panel controls `/gr00t_rollout/{start,success,failure,abort,reset,ready,approve}` and displays
the latched status. `success`/`failure` stop consuming trajectories and retain the last target;
`abort` closes Servo. After finalization, `reset` closes Servo/pinch and enters the resetting state;
`ready` is a separate operator confirmation that the robot and scene are actually restored. A new
`start` is refused until both commands have completed. The GUI automatically begins passive dataset
recording on `start`, saves success/failure episodes with an external outcome sidecar, and discards
aborted episodes.

Reset defaults to release-only manual operation:

```text
RESET_MODE=manual
```

After the robot owner independently verifies the SDK home service, `RESET_MODE=go-home` additionally
requests `/zj_humanoid/upperlimb/go_home/dual_arm` after Servo is released. A successful Trigger
response only means the request was accepted; it never replaces visual inspection, task-object
reset, and the explicit `Reset Complete / Ready` confirmation.

Persistent outputs are:

```text
logs/kion_rollout/<episode>/metadata.json, tcp_tracking.csv, outcome.json
logs/kion_speed_rl/episodes/<episode>/metadata.json, tcp_tracking.csv, outcome.json
logs/kion_speed_rl/state/                 # replay, checkpoint and phase state
logs/observation_gui_rollout/app/datasets/parcel_rollout_view/
```

Do not delete the Speed-RL state directory between launches. A `finalization_error` blocks the next
episode so a failed checkpoint or label write cannot be silently skipped.
