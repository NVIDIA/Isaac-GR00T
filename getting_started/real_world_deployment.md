# GR00T Real-World Deployment Guide

This guide covers building an end-to-end real-world VLA pipeline—from data collection and training to deployment—with practical engineering recommendations.

## Overview

A typical GR00T real-world deployment workflow includes:

1. **[Hardware Preparation](#1-hardware-and-environment-preparation-device-requirements)**: Verify that the robot platform, sensors, and compute resources are ready.
2. **[Data Collection](#2-data-collection)**: Choose an appropriate teleoperation setup and collect at least 100 valid episodes.
3. **[Data Preprocessing](#3-data-preprocessing)**: Clean data, align timestamps, and convert to LeRobot format.
4. **[Model Training](#4-vla-model-training)**: Fine-tune GR00T N1.*.
5. **[Model Evaluation](#validation)**: Run open-loop evaluation to validate convergence and model quality.
6. **[Deployment Setup](#5-deployment-and-closed-loop-control)**: Build a ZMQ Server-Client architecture.
7. **[Closed-Loop Testing](#5-deployment-and-closed-loop-control)**: Run closed-loop control on real hardware and monitor jittering and stop-and-go behavior.
8. **[Optimization](#6-common-issues-jittering-and-stop-and-go)**: Tune RTC parameters and trajectory smoothing strategies based on real-world performance.

## 1. Hardware and Environment Preparation (Device Requirements)

Ensure your robot hardware, sensor pipeline, and control interfaces are stable and available.

### Robot Platform

- **Recommended platforms**: Robotic arms with SDK-level control support (e.g., Franka, UR, Piper, SO101).
- **Basic requirements**:
  - Real-time joint state feedback.
  - High-frequency action execution (30 FPS recommended).
  - Stable control interface.

### Multimodal Sensors

| Sensor Type | Specification | Purpose |
|-------------|---------------|---------|
| **Wrist-mounted camera** | 30 FPS, RGB | Capture close-range manipulation visuals |
| **Third-person camera (3rd view)** | 30 FPS, RGB | Capture global scene context |
| **Robot proprioceptive state** | Real-time acquisition | Joint states and gripper state |

### Compute Resources

- **Training phase**: NVIDIA GPU servers (e.g., H100 or H20) are recommended for larger batch sizes.
- **Deployment phase**: Edge hardware such as Jetson AGX Thor supports on-device inference.

> For details, see the [hardware recommendation guide](hardware_recommendation.md).

### Teleoperation Devices

Teleoperation device selection is critical for data quality.

### Teleoperation Device Comparison

In the table below:
- **Embodiment dependency**: how similar the teleoperation device and target robot must be in joint topology, degrees of freedom (DoF), and workspace. Higher dependency implies harder cross-embodiment transfer.
- **Operational intuition**: how naturally operator inputs map to robot motion. Higher intuition means faster onboarding and lower demonstration error.

| Device Type | Cost Level (Reference) | Embodiment Dependency | Operational Intuition | Notes |
|-------------|------------------------|-----------------------|-----------------------|-------|
| **Keyboard/Gamepad/SpaceMouse/Joylo** | Low | Low: command mapping via keys/controls | Medium: requires adaptation to key-motion mapping | Low entry cost; a good starting point and useful in mobile scenarios |
| **Master-Slave arm systems** | Medium | High: master/slave arms usually require similar kinematics and workspace | High: near one-to-one human-robot mapping | Suitable for single-robot setups; commonly used by robot OEMs; can reduce the risk of reaching joint limits during demonstrations |
| **UMI / Fast-UMI / Pika Sense** | Medium | Low: hardware-agnostic action representation reusable across arms | High: after calibration, end-effector (EEF) following is intuitive | Suitable for training general VLA models; low-DoF arms may still hit joint limits |
| **VR-based teleoperation** | Medium (headset + rendering + network) | Low: mainly depends on software integration | Medium: depends on immersive visual feedback and tracking quality | A flexible solution, but with higher integration overhead |
| **Glove / Motion Capture** | High (commercial mocap suite + data gloves) | Low: retarget through kinematic mapping to different embodiments | High: intuitive full-hand/full-body control | Suitable for full-body control and dexterous-hand tasks |
| **Exoskeleton** | High | High: usually requires matched joint structure | High: natural action correspondence | Extendable to multi-joint humanoid control |

## 2. Data Collection

Key considerations for data collection:

### Timestamp Synchronization

- The FPS of both camera streams should be strictly matched, and capture triggers should be as synchronized as possible.
- Joint state sampling frequency should exceed camera FPS to enable accurate downsampling.
- Record full timestamps during collection for downstream temporal alignment.

### Action Representation

- If training and collection use the same embodiment (e.g., master-slave arms), log joint-space `Joint States` during collection. For task-space models, compute EEF pose via forward kinematics (FK) in post-processing.
- If embodiments differ (e.g., collect with UMI, deploy on Piper), directly record task-space EEF pose during collection.

### Data Distribution

- Current imitation-learning-based models perform more reliably in previously seen scenarios. In early-stage experiments, start with data collection and validation in a limited domain.
- After pipeline validation, gradually expand the domain by varying lighting, object placement, and initial robot poses to improve generalization.

### Scene Consistency

- Keep third-person camera extrinsics fixed and ensure a rigid wrist-camera mount.
- In early experiments, prioritize scene consistency; avoid varying lighting, object placement, or initial robot poses.

### Joint Limits

- If collecting joint-space data, avoid operating near joint limits to reduce the number of samples in those regions.

## 3. Data Preprocessing

Raw data must be cleaned, synchronized, and converted before training.

### Trajectory Filtering

Data filtering is recommended in two stages: script-based filtering and manual review.

#### Script Filtering

- Check image timestamps and remove samples with:
  1. Excessive latency in a single camera stream.
  2. Excessive timestamp difference between the two camera streams.
- Detect and remove abnormal jumps in robot state sequences.

#### Manual Filtering

Replay trajectories with synchronized visualization to catch issues missed by scripts:

- Remove samples with poor synchronization between image and action sequences.
- Remove blurry frames.
- Remove failed task executions.
- Remove low-quality trajectories (e.g., redundant paths, discontinuous actions).

### Trajectory Preprocessing

1. Timestamp alignment: align camera frames and robot joint states to a shared time base.
2. Head-tail trimming: remove idle segments at the start and end of trajectories.
3. Split long trajectories (several minutes) into multiple subtasks.

### Format Conversion

Convert all data to a standard format (e.g., LeRobot) for GR00T compatibility:

- See the [data preparation guide](data_preparation.md) for format requirements.
- Use the provided conversion scripts to convert data to GR00T LeRobot format.

## 4. VLA Model Training

### Training Parameter Configuration

**Dataset size recommendations**

For single-task `finetune`:

- **Minimum data size**: Prepare at least **100 valid episodes**. For very narrow task domains, ~30 episodes may suffice. A capture frequency of 20–50 Hz is recommended for manipulation tasks.
- **Episode length**: No hard limit, but each episode must contain a complete action cycle with idle frames removed. Split overly long episodes into subtasks.
- **Recommended data size**: 200+ episodes usually provide more stable performance.

**Core parameters**

- **Input/output mode**: Default to `State-relative Action Prediction`. Compared with `Absolute Action`, it converges more easily and improves inter-chunk consistency.
- **Training space**: Both joint space and task space are valid. For low-DoF arms, joint space is often preferred to reduce singularity-related risks.
- **Action Chunk Size**: Default is 16. If combined with RTC to mitigate stop-and-go, set it to at least 32.
- **Batch Size**: Increase the batch size as much as GPU memory allows.

> For additional training options, see the [fine-tuning guide](finetune_new_embodiment.md).

**Compute resources**

- Fine-tuning requires significantly less compute than pretraining.
- A single compute node (8 x H100 or 8 x H20) is usually sufficient.

### Validation

After training, run open-loop validation to confirm convergence, then proceed to closed-loop deployment validation.

> Open-loop validation is only a preliminary check. Final performance must be verified with closed-loop testing on real robots. For details, see the [fine-tuning guide](finetune_new_embodiment.md).

## 5. Deployment and Closed-Loop Control

### System Architecture

GR00T supports two inference modes:

1. **Direct `Gr00tPolicy` usage**: Suitable when model inference and robot control run on the same machine.
2. **ZMQ Server-Client architecture**: Suitable for real-world deployment and decouples local robot control (`Local Client`) from remote inference (`Model Server`).

For real-world deployment, **ZMQ inference service** is recommended:

- Move compute-intensive inference to GPU servers.
- Keep robot-side control code lightweight.
- Avoid installing the full inference dependency stack on the robot side.

### On-Device Deployment Logic

Deployment code has two phases: **initialization** and the **main control loop**.
The pseudo-code below uses a synchronous workflow, which may cause stop-and-go. See later sections for mitigation via asynchronous execution + RTC.

**Pseudo-code workflow:**

```python
# ========== Initialization ==========
# 1. Initialize and test cameras
hand_camera = initialize_hand_camera()  # e.g., OrbbecSDK
env_camera = initialize_env_camera()    # e.g., RealSense
test_cameras()  # Show preview and verify normal operation

# 2. Connect and test robot
robot = connect_robot()  # e.g., Piper SDK
robot.enable()
robot.reset_to_initial_position()
test_robot()  # Send test command and verify robot response

# 3. Connect and test GR00T model server
gr00t_client = connect_to_gr00t_server(host, port)
if not gr00t_client.ping():
    raise ConnectionError("Failed to connect to model server")
test_model()  # Send test observation and verify inference

# ========== Main control loop ==========
while True:
    # 1. Acquire sensor data
    hand_image = hand_camera.get_frame()
    env_image = env_camera.get_frame()
    joint_states = robot.get_joint_states()
    gripper_state = robot.get_gripper_state()

    # 2. Format observation
    observation = format_observation(
        hand_image,
        env_image,
        joint_states,
        gripper_state,
        task_description,
    )

    # 3. Model inference (via ZMQ)
    actions = gr00t_client.get_action(observation)

    # 4. Trajectory post-processing
    actions_arm = actions["joint_states"]
    actions_arm = smooth_trajectory(actions_arm)  # smoothing
    actions_arm = check_safety_limits(actions_arm)  # safety checks

    # 5. Execute actions
    for action_step in actions_arm:
        robot.execute_action(action_step)
        sleep(1.0 / 30.0)  # 30 FPS
```

### Key Implementation Notes

**Important notes:**

- **Image format**: Use compressed formats such as JPG to reduce transmission bandwidth.
- **Safe operation**:
  - **Soft Limits**: Add joint-angle and EEF pose range checks. If a predicted action exceeds workspace bounds, raise an alarm and stop immediately.
  - **E-Stop logic**: Bind an emergency stop hotkey (e.g., Space) on the control PC, or use a physical E-Stop switch.
- **Action smoothing**: Apply interpolation and smoothing to predicted action sequences.

> For more deployment details, see the [policy API guide](policy.md).

## 6. Common Issues: Jittering and Stop-and-Go

The most common issues in real-world deployment are **jittering** and **stop-and-go**.

### Fixing Jittering

**Jittering** here refers to visible shaking or vibration of the end-effector or joints during task execution.

Jittering typically originates from **inconsistent model outputs** or **insufficient robot-side control quality**. Analyze these two components separately to localize the issue. The suggestions below are general guidelines and may not apply to every robot platform or control stack — always verify against your own hardware and environment.

```mermaid
flowchart TD
    A[Jittering observed] --> B[Save & visualize action chunks in 3D]
    B --> C{Where is the jitter?}
    C -->|Inside each chunk| D[Case A: Model undertrained or poor data quality]
    C -->|Between consecutive chunks| E[Case B: Inconsistent chunk predictions]
    C -->|Chunks look smooth| F[Case C: Robot hardware / low-level control issue]
    D --> D1[Add more data, train longer, check train/eval consistency]
    E --> E1[Use state-relative actions + RTC chunking strategy]
    F --> F1[Check drive control, interpolation, hardware status]
```

**Diagnosis and mitigation**

1. **Save and visualize Action Chunks**
   - Save all predicted `Action Chunks`.
   - Visualize continuous TCP (tool center point) trajectories in 3D.
   - **Note**: Convert joint-space outputs to task space via FK before visualization.

2. **Analyze visualization results**

   **Case A: Significant jitter inside each chunk**
   - **Cause**: The model is undertrained, or data quality is insufficient.
   - **Solution**: Improve data quality, add more training data, or train longer. Keep training and validation environments consistent.

   **Case B: Significant jitter between chunks**
   - **Cause**: Inconsistent adjacent `Action Chunk` predictions.
   - **Solution**:
     - Use `State-relative Action Prediction`. Predicting actions relative to the current state produces a more uniform output distribution, making the network easier to train.
     - Use RTC (`Real-Time Chunking`) or similar strategies.

   **Case C: Little jitter after visualization**
   - **Cause**: Likely a robot hardware or low-level control issue.
   - **Solution**: Check drive control, interpolation, and hardware status.

**Quantitative diagnostic metrics**

Trajectory jitter can also be quantified using these three metrics:

**Metric 1: Mean intra-chunk acceleration magnitude**

Measures intra-chunk smoothness. Only valid under fixed sampling frequency.

Formula: $a_t = pos_{t+1} - 2 \cdot pos_t + pos_{t-1}$

```python
def metric_intra_accel(chunks):
    """
    Args:
        chunks: numpy array with shape (N_chunks, Chunk_Length, Joint_Dim)

    Returns:
        float: Mean acceleration magnitude
    """
    velocity = np.diff(chunks, axis=1)  # first-order difference
    acceleration = np.diff(velocity, axis=1)  # second-order difference
    acc_magnitude = np.linalg.norm(acceleration, axis=-1)  # L2 norm per step
    return np.mean(acc_magnitude)
```

**Metric 2: Position jump at chunk boundary (L2 distance)**

Measures position continuity between chunks by comparing the last executed step of `Chunk[i]` with step 0 of `Chunk[i+1]`.

```python
def metric_boundary_jump(chunks, execute_steps=None):
    """
    Args:
        chunks: numpy array with shape (N_chunks, Chunk_Length, Joint_Dim)
        execute_steps: number of executed steps per chunk; if None, use full chunk length

    Returns:
        float: Mean position jump
    """
    chunks = np.array(chunks)
    exec_steps = chunks.shape[1] if execute_steps is None else execute_steps

    last_frame_prev = chunks[:-1, exec_steps - 1, :]  # last frame of previous chunk
    first_frame_curr = chunks[1:, 0, :]  # first frame of current chunk
    jumps = np.linalg.norm(first_frame_curr - last_frame_prev, axis=-1)  # Euclidean distance
    return np.mean(jumps)
```

**Metric 3: Cosine similarity of velocity direction at chunk boundary**

Measures velocity-direction consistency between chunks. Values closer to 1 indicate better consistency.

```python
def metric_momentum_shift(chunks, execute_steps=None):
    """
    Args:
        chunks: numpy array with shape (N_chunks, Chunk_Length, Joint_Dim)
        execute_steps: number of executed steps per chunk; if None, use full chunk length

    Returns:
        float: Mean cosine similarity
    """
    chunks = np.array(chunks)
    exec_steps = chunks.shape[1] if execute_steps is None else execute_steps

    # velocity at the end of previous chunk
    idx = exec_steps - 1
    if idx < 1:
        raise ValueError("execute_steps must be >= 2 to compute end velocity")
    v_end = chunks[:-1, idx, :] - chunks[:-1, idx - 1, :]

    # velocity at the start of current chunk
    v_start = chunks[1:, 1, :] - chunks[1:, 0, :]

    # cosine similarity
    dot_product = np.sum(v_end * v_start, axis=-1)
    norm_prev = np.linalg.norm(v_end, axis=-1)
    norm_curr = np.linalg.norm(v_start, axis=-1)
    epsilon = 1e-8
    cosine_sim = dot_product / (norm_prev * norm_curr + epsilon)

    return np.mean(cosine_sim)
```

### Fixing Stop-and-Go
Stop-and-Go here refers to a behavior in which the robot intermittently pauses during motion, producing periodic stop-and-go behavior.

#### Root Cause

In **synchronous single-step closed-loop** control, stop-and-go occurs when the **end-to-end latency** (observation capture → VLA inference → action conversion) exceeds control-frequency requirements.

- **Control-frequency requirement**: At 30 FPS, latency must stay below ~33 ms.
- **Typical latency sources**: Data capture, network transfer, model inference, and post-processing often exceed 33 ms combined.
- **Consequence**: The next prediction is not ready when the current action finishes, causing pauses.

#### Solutions

**Option 1: Optimize the inference pipeline (direct but difficult)**

Reduce full workflow latency below 33 ms:

- Optimize network bandwidth (reduce transfer time).
- Use edge inference (reduce network latency).
- Quantize the VLA model (speed up inference).
- Use a smaller model (e.g., ACT).

**Limitation**: For VLA models, meeting strict real-time requirements through optimization alone is often impractical.

**Option 2: Use algorithmic scheduling strategies (recommended)**

When direct optimization is insufficient, use one or more of the following:

- **Asynchronous Inference**: A background thread runs inference while the main thread executes actions.
- **Receding Horizon**: Execute only the first few steps of each `Action Chunk` before triggering a new inference.
- **Temporal Ensemble**: Aggregate predictions across multiple timesteps.
- **Real-Time Chunking (RTC)**: Overlap the start of the current prediction with unexecuted steps from the previous one.

**Recommended strategy**: `Asynchronous Inference + RTC` is usually the most effective.

#### Real-Time Chunking (RTC) Details

**Principle**

RTC treats action prediction as an inpainting problem: overlapping the start of the new prediction with unexecuted steps from the previous one ensures smooth transitions.

**Applicability**

- Validated for **diffusion / flow-based** VLA policies.
- Requires `Action Chunk` length ≥ 32 steps.
- Should be combined with asynchronous inference.

**Implementation essentials**

1. **Predict longer Action Chunks**:
   - Increase from the default 16 steps to at least 32.
   - Provide a larger soft fusion window.

2. **Asynchronous inference architecture**:
   - **Background thread**: Continuously infer, capture observations, and prepare action batches.
   - **Main thread**: Execute the current action sequence.
   - Buffer predictions in a queue to avoid blocking.

3. **Action fusion mechanism**:
   - Use RTC for soft fusion in the overlap region.
   - Ensure smooth transitions between adjacent chunks.

**Pseudocode: Async Inference + RTC**

In the RTC (Real-Time Chunking) framework, two key parameters control how adjacent action chunks overlap and transition:

- **`overlap`**: The number of action steps retained from the previous prediction to constrain the current one, ensuring temporal consistency between consecutive chunks.
- **`frozen`**: The number of steps that remain completely frozen (i.e., not updated by the new prediction), typically set to match the inference latency.

Below is a simplified async inference + RTC loop. Note that official RTC support for GR00T is coming soon; the current implementation may require manual adaptation.

```
actions = policy.infer(obs)                        # blocking first call

loop:
    for i in range(action_horizon):
        if i == action_horizon - overlap - 1:
            future = async policy.infer(new_obs)   # non-blocking
        robot.execute(actions[i])
        if i == action_horizon - frozen - 1:
            actions = future.get()                 # swap in next chunk
            break                                  # discard frozen tail
```


