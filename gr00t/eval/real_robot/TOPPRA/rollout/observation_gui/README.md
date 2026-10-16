# ObservationGUILite rollout view

This overlay reuses ObservationGUILite without allowing its 30 Hz control loop to publish robot
actions. The Isaac-GR00T rollout process remains the only owner of the 250 Hz dual-arm Servo and
hand commands. The overlay omits ObservationGUILite's action-controller objects entirely; it only
subscribes to observations and rollout status.

Start a plain or Speed-RL client with `--control-interface gui` or `both`, then run from the
Isaac-GR00T repository root:

```bash
bash gr00t/eval/real_robot/TOPPRA/rollout/launch_gui.sh rollout.local.env
```

The default `OBSERVATION_GUI_RUNTIME=native` reuses Isaac-GR00T's Python 3.12 environment. Install
the two GUI-only wheels once after `uv sync`:

```bash
uv pip install --python .venv/bin/python --no-deps av==15.1.0 dearpygui==2.0.0
```

Set `OBSERVATION_GUI_RUNTIME=docker` to retain the original container workflow. The native mode is
recommended on restricted robot-lab networks because the upstream image otherwise downloads a
managed Python 3.13 runtime from Astral/GitHub during every first build.

The overlay is assembled under `logs/observation_gui_rollout/app`; it does not modify the supplied
ObservationGUILite checkout. Recorded datasets survive either runtime under that directory.
Override its parent with `OBSERVATION_GUI_RUNTIME_ROOT` if required.

The GUI exposes Start, Success, Failure, Abort and Speed Calibration Approval buttons and displays
trajectory, inference, replay and safety status from `/gr00t_rollout/status`. On Start it enters
passive recording. Success/failure saves the episode under
`logs/observation_gui_rollout/app/datasets/parcel_rollout_view` with the external outcome in the
debug-event JSON sidecar; abort discards it. Use the fixed `/gr00t_rollout` namespace for this first
version.
