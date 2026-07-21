# G1-Dex3 GR00T policy server (TRI `policy_interfaces` gRPC)

Serves the finetuned Unitree G1 (bimanual, Dex3 hands, upper body) GR00T N1.7
checkpoint to the existing TRI robot stack, so **anzu's `RosRobotPolicyRunner`
runs unchanged**. One process on the workstation dGPU loads `Gr00tPolicy`
in-memory *and* speaks the `policy_interfaces` gRPC contract
(`MultiarmObservation → PosesAndGrippers`); anzu on the G1's Jetson Orin connects
over ethernet.

```
anzu RosRobotPolicyRunner (Orin) ──gRPC (ethernet)──▶ this server (workstation dGPU)
        MultiarmObservation                              ├─ MultiarmObservation → GR00T obs
        ◀── PosesAndGrippers ────────────────────────────┤  Gr00tPolicy.get_action (chunk)
                                                          └─ GR00T action → PosesAndGrippers
```

## Files

- `gr00t_g1_policy_server.py` — `GrootG1Policy(Policy)` + `main()`. Implements the
  duck-typed `get_policy_metadata` / `reset_batch` / `step_batch` the server
  calls, with a per-client open-loop action buffer (GR00T predicts a chunk; the
  contract returns one action per step).
- `conversions.py` — `MultiarmObservation ↔ GR00T` nested dicts, incl. rot6d↔matrix
  (reusing `gr00t.data.state_action.pose.EndEffectorPose`).

## Model I/O (embodiment tag `new_embodiment`)

State & action are 32-dim: `left_arm_eef[0:9] | left_hand[9:16] | right_arm_eef[16:25] | right_hand[25:32]`.
Arms are 9D `xyz + rot6d`; hands are 7D Dex3 joint targets. Video keys:
`ego_view`, `left_wrist`, `right_wrist`. Arm actions are **relative EEF** —
`Gr00tPolicy` un-relativizes to absolute using the state we feed it, so as long
as the state EEF pose comes from anzu's observation the returned pose is already
in anzu's frame.

## 1. Install (co-install gate — do this first)

Run the server inside the **main Isaac-GR00T venv** (it needs the full gr00t +
torch stack); the only extra dependency is TRI's `policy_interfaces`:

```bash
cd <Isaac-GR00T>
uv sync --all-extras                          # full gr00t + torch
source .venv/bin/activate
uv pip install -e <path/to>/policy_interfaces # brings drake/grpcio/protobuf/opencv

# GATE: confirm both trees import in one interpreter before going further.
python -c "import gr00t; import policy_interfaces.robot_gym; import pydrake.math; print('ok')"
```

`policy_interfaces` pins `numpy~=1.26.4` (matches gr00t's `==1.26.4`). The
versions to watch if the import fails are `drake`, `protobuf~=6.31`,
`opencv-python~=4.6`.

## 2. Launch the server (workstation)

```bash
# convenience wrapper (env-overridable: MODEL_PATH, SERVER_URI, OPEN_LOOP_STEPS, S3_CACHE_DIR)
MODEL_PATH=<path/to>/finetuned_gr00t bash gr00t/eval/real_robot/G1Dex3/launch_gr00t_g1_policy.sh

# or directly:
python gr00t/eval/real_robot/G1Dex3/gr00t_g1_policy_server.py \
    --model-path <path/to>/finetuned_gr00t \
    --device cuda \
    --open-loop-steps 8 \
    --server-uri 0.0.0.0:50051
```

Binds `0.0.0.0:50051` by default so the Orin can reach it. Wait for
`Started Server loop on 0.0.0.0:50051...` before starting the client.

**Loading a checkpoint from S3.** `--model-path` (and the `MODEL_PATH` env var)
also accepts an `s3://bucket/prefix` URI. On startup the server downloads every
object under the prefix into a local cache and loads from there — objects already
cached with a matching size are skipped, so restarts don't re-download:

```bash
MODEL_PATH=s3://my-bucket/checkpoints/finetuned_gr00t \
    bash gr00t/eval/real_robot/G1Dex3/launch_gr00t_g1_policy.sh
```

The cache directory is `--s3-cache-dir` / `$S3_CACHE_DIR` / `$GR00T_S3_CACHE_DIR`,
defaulting to `~/.cache/gr00t/s3/<bucket>/<prefix>`. Standard AWS credentials
apply (env vars, `~/.aws/credentials`, or an instance/role profile — resolved by
`boto3`). `get_policy_metadata` still reports the original `s3://` URI as the
checkpoint path.

## 3. Point anzu at the workstation (Orin)

No code change — set `RosRobotPolicyRunner`'s `lbm_policy_client` `server_uri`
to `<workstation-ip>:50051` (the `LbmPolicyClientConfig.server_uri` param in
`robot_policy/config/lbm_policy_client.yaml`), then launch the client:

```bash
ros2 launch unitree_tri ros_robot_policy_runner.launch.py policy_yaml:=lbm_policy_client.yaml
```

## 4. How this maps onto the existing G1 deployment

This server is a **drop-in replacement for the vla_foundry diffusion-policy G1
server** — same anzu client, same gRPC port, same procman group. The reference
diffusion deployment (vla_foundry, branch `policy-interfaces-inference`) is:

```bash
# examples/deployment/launch_g1_policy.sh
CUDA_VISIBLE_DEVICES=0 uv run --group inference python \
  vla_foundry/inference/robotics/g1/g1_inference_diffusion_policy.py \
  --interface grpc --checkpoint_directory <ckpt> --num_flow_steps 8 --device cuda --open_loop_steps 24
```

For GR00T you run `gr00t_g1_policy_server.py --model-path ...` instead; everything
downstream is unchanged. `g1/g1_field_mapping.yaml` in that repo independently
matches this dir's `G1DexConversionConfig` defaults (cameras, `/current_*_hand_ee_link`,
and the `{side}_hand_{thumb_0..index_1}_joint` order), which is why the name maps
above are trustworthy.

**Two deploy paths, one caveat:**
- **Standalone (recommended):** `launch_g1_policy.sh` (diffusion) / `launch_gr00t_g1_policy.sh`
  (GR00T) + the `ros2 launch` client above. Self-contained and current.
- **procman one-shot** (`demonstrator/launch_procman.py`, "🔮 Policy Inference" group)
  launches server + client together, but on this vla_foundry branch its hardcoded
  server command is **stale**: it calls `robotics/inference_policy.py --field_mapping_path
  robotics/g1_field_mapping.yaml`, but that mapping moved to `robotics/g1/g1_field_mapping.yaml`
  and the current G1 entry point is `robotics/g1/g1_inference_diffusion_policy.py`. Use the
  standalone path (or fix the procman path) until anzu's procman is updated.

Transport note: the G1 diffusion server also supports `--interface dds`, but anzu's
`lbm_policy_client.yaml` uses the gRPC client (`type: lbm_policy_client`), so gRPC is
the active path — which is what this server speaks.

## Name maps (`G1DexConversionConfig`)

Defaults are the actual anzu names, traced from anzu's
`robot_policy_system_params.yaml` (topics) and GR00T's `convert_mcap_to_lerobot.py`
(camera / EE / hand-joint mapping) — so no edits should be needed in the nominal
setup. Override by passing a configured `G1DexConversionConfig` to `GrootG1Policy`
if your deployment differs.

| Map | Default | Source |
| --- | ------- | ------ |
| `camera_map` | `/head_camera/zed_node/left/...→ego_view`, `/left_wrist_camera/...→left_wrist`, `/right_wrist_camera/...→right_wrist` | `visuo` keys are the camera ROS topics; ego_view = ZED head **left** eye (head right unused) |
| `arm_state_pose_map` | `left_arm_eef→/current_left_hand_ee_link`, `right_arm_eef→/current_right_hand_ee_link` | `poses` keys are the actual-EE-pose `PoseStatus.topic_name` |
| `arm_action_out_map` | `/ee_target_left`, `/ee_target_right` | anzu publishes these as `PoseStamped` (frame `pelvis`) |
| `hand_joint_map` | `{side}_hand_{thumb_0,thumb_1,thumb_2,middle_0,middle_1,index_0,index_1}_joint`, in that order | order = GR00T `HAND_JOINT_ORDER` (checkpoint index order); names from anzu Dex3 URDF |

**The one thing to verify on-robot:** the Dex3 `HandStatus` joint-name *strings*
in the live observation. The *order* is fixed by training (do not reorder), but
confirm the exact names match (e.g. `ros2 topic echo` the assembled keyframe, or
check the hand-state joint names) — if anzu uses different strings, update
`hand_joint_map` values (keeping the order). Also confirm hand state arrives via
`grippers` (assumed) rather than `joint_position`.

## Verify

1. **Co-install gate** — the `python -c "import ..."` above.
2. **Conversion round-trip** — `pytest conversions_test.py` (rot6d↔matrix inverse,
   shapes, 14 named grippers).
3. **Loopback** — launch the server against `finetuned_gr00t`, then from a second
   process use `policy_interfaces`' `LbmPolicyClient` to send a synthetic
   `MultiarmObservation` and confirm a real `PosesAndGrippers` returns and
   `get_policy_metadata().is_language_conditioned` is `True`.
4. **On-robot** — point anzu at the workstation; move the G1 upper body slowly.
