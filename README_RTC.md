# Real-Time Chunking (RTC) Implementation for GR00T N1.6

## Overview

Real-Time Chunking (RTC) is a real-time inference strategy that leverages previously predicted actions to improve the coherence and temporal consistency of current predictions. It treats the action sequence as an **in-painting problem**, where part of the previous prediction is passed as a constraint to the current inference, enabling smoother action transitions.

Note: this version does not support automatically calculate fix size and overlap size by latency.

Tips:
1. Try to use longer action chunk size. Better use 32 than 16.


## Usage

There are two main ways to run RTC-inference with GR00T N1.6:

1. **Standalone Mode**  
   - Update the `MODEL_PATH` variable in the script with your model path.
   - Run the inference process directly:  
     ```bash
     uv run python scripts/standalone_inference_RTC.py
     ```
   - This script loads the model and executes real-time chunked inference.
   - Check the output file `scripts/joint_trajectories_RTC.png`.


2. **Server-Client Mode**  
   For deployment, server-client mode is recommended.
   - Start the RTC inference server:  
     ```bash
     uv run python gr00t/eval/run_gr00t_server.py \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path <Your_absolute_path_to_the_model>/checkpoint-20000 \
        --device cuda:0 \
        --host 0.0.0.0 \
        --port 5555
     ```
   - Then run the client to send tasks and receive predictions:  
     ```bash
     uv run python scripts/test_piper_client_RTC.py
     ```
   - This setup enables flexible, scalable, and remote RTC-based policy inference.

For both modes, ensure RTC parameters are set appropriately in your config or checkpoint (`inference_rtc_overlap_steps`, `inference_rtc_frozen_steps`, `rtc_ramp_rate`, `pridict_horizon`).  
For detailed implementation and script modifications, refer to the sections below and the provided example scripts.




---

## Key Parameters

Defined in `Gr00tN1d6ActionHead` class in `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`:

```python
# Real-time chunking parameters
self.inference_rtc_overlap_steps: int | None = 16  # Number of overlap steps
self.inference_rtc_frozen_steps: int | None = 6    # Number of frozen steps
self.rtc_ramp_rate: float = 6.0                    # Ramp rate for gradual transition
self.pridict_horizon = 32                          # Prediction horizon
```

| Parameter | Description |
|-----------|-------------|
| `inference_rtc_overlap_steps` | Number of action steps retained from the previous prediction to constrain the current prediction |
| `inference_rtc_frozen_steps` | Number of steps that are completely frozen (not updated), typically set to the inference latency |
| `rtc_ramp_rate` | Exponential ramp rate for the transition from frozen to fully updated steps |
| `pridict_horizon` | Total action length predicted per inference |

---

## Core Modifications

### 1. Model Layer (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`)

#### 1.1 RTC Core Logic - `get_action_with_features()` Method

```python
# Initialize actions as random noise
actions = torch.randn(
    size=(batch_size, self.config.action_horizon, self.action_dim),
    dtype=vl_embeds.dtype,
    device=device,
)

# RTC: Override the first overlap steps with the previous prediction
use_rtc = False
if self.inference_rtc_overlap_steps is not None:
    if "action" not in action_input.keys():
        print("WARNING: action.* is mandatory when using Realtime chunking, we will not use RTC")
    else:
        # Place the previous overlap portion at the beginning of current actions
        actions[:, : self.inference_rtc_overlap_steps, :] = action_input["action"][:, : self.inference_rtc_overlap_steps, :]
        use_rtc = True
```

#### 1.2 Velocity Update Strength Control - `vel_strength`

```python
vel_strength = torch.ones_like(actions)
if use_rtc:
    # Frozen steps: velocity update strength = 0
    vel_strength[:, : self.inference_rtc_frozen_steps, :] = 0.0
    
    # Intermediate steps: use exponential ramp
    intermediate_steps = self.inference_rtc_overlap_steps - self.inference_rtc_frozen_steps
    t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
    ramp = 1 - torch.exp(-self.rtc_ramp_rate * t)
    ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
    ramp = ramp[1:-1]  # only take the middle part
    
    vel_strength[:, self.inference_rtc_frozen_steps : self.inference_rtc_overlap_steps, :] = ramp[None, :, None]
```

#### 1.3 Modified Euler Integration Update

```python
# Original: actions = actions + dt * pred_velocity
# RTC:      Use vel_strength to control update magnitude
actions = actions + dt * pred_velocity * vel_strength
```

---

### 2. Policy Layer (`gr00t/policy/gr00t_policy.py`)

#### 2.1 Support for Action Input

Added action handling in `_to_vla_step_data()` method:

```python
def _to_vla_step_data(self, observation: dict) -> VLAStepData:
    if 'action' in observation:
        actions = observation["action"]
    else:
        actions = {}
    return VLAStepData(
        images=observation["video"],
        states=observation["state"],
        actions=actions,  # Pass in the previous action
        text=observation["language"][self.language_key][0],
        embodiment=self.embodiment_tag,
    )
```

#### 2.2 Modified Unbatch Logic

```python
def _unbatch_observation(self, observation):
    # ...
    for i in range(batch_size):
        if 'action' in value:
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "action": {k: v[i] for k, v in value["action"].items()},  # Added
                "language": {k: v[i] for k, v in value["language"].items()},
            }
        else:
            # Original logic...
```

---

### 3. Server Layer (`gr00t/eval/run_gr00t_server.py`)

Added `clip_outliers` configuration option:

```python
@dataclass
class ServerConfig:
    clip_outliers: bool = True
    """Whether to clip normalized values to [-1, 1]"""

def main(config: ServerConfig):
    # ...
    policy.processor.clip_outliers = config.clip_outliers
    policy.processor.state_action_processor.clip_outliers = config.clip_outliers
```

> **Note**: It is recommended to disable `clip_outliers` during RTC inference to avoid inconsistencies in the action normalization/denormalization process.

---

## Inference Workflow

### Client Side Example

Refer to `scripts/standalone_inference_RTC.py` and `scripts/test_piper_client_RTC.py`:

```python
# Parameter configuration
inference_rtc_frozen_steps = 6
inference_rtc_overlap_steps = 16
pridict_horizon = 32
next_state_index = pridict_horizon - inference_rtc_overlap_steps - 1

predicted_action = None

for step in range(num_steps):
    # Step 1: Prepare state
    if step == 0:
        joint_states = initial_joint_states  # (1, 1, 6)
        gripper_distance = initial_gripper   # (1, 1, 1)
    else:
        # Use a specific position from the previous prediction as current state
        joint_states = predicted_action['joint_states'][:, next_state_index:next_state_index+1, :]
        gripper_distance = predicted_action['gripper_distance'][:, next_state_index:next_state_index+1, :]

    # Step 2: Build observation
    observation = {
        "video": {...},
        "state": {
            "joint_states": joint_states,
            "gripper_distance": gripper_distance,
        },
        "language": {...}
    }

    # Step 3: From the second step onwards, add action for RTC
    if step > 0:
        # Create padding arrays
        pad_joint = np.zeros((1, pridict_horizon, 6), dtype=np.float32)
        pad_gripper = np.zeros((1, pridict_horizon, 1), dtype=np.float32)
        
        # Extract overlap portion and place at the beginning
        valid_joint = predicted_action['joint_states'][:, pridict_horizon - inference_rtc_overlap_steps:, :]
        valid_gripper = predicted_action['gripper_distance'][:, pridict_horizon - inference_rtc_overlap_steps:, :]
        
        pad_joint[:, :inference_rtc_overlap_steps, :] = valid_joint
        pad_gripper[:, :inference_rtc_overlap_steps, :] = valid_gripper
        
        observation['action'] = {
            "joint_states": pad_joint,
            "gripper_distance": pad_gripper,
        }

    # Step 4: Inference
    predicted_action, info = policy.get_action(observation)
```

### RTC Action Timeline

```
Step 0: 
  Initial random noise → Denoise → predicted_action[0:32]
                                          ↓
                                    Execute action[0:16]

Step 1:
  ┌─────────────────────────────────────────────────────────┐
  │ Overlap (16 steps)               │ New prediction (16)  │
  │ [frozen: 6] [ramp: 10]           │                      │
  └─────────────────────────────────────────────────────────┘
       ↑
  From Step 0's action[16:32]
       │
       └→ Normalized and placed at action[:16]

  During denoising:
  - action[0:6]:   Completely frozen (vel_strength=0)
  - action[6:16]:  Gradual update (vel_strength: 0→1 exponential ramp)
  - action[16:32]: Fully updated (vel_strength=1)
```

## References

- [Isaac-GR00T PR #320](https://github.com/NVIDIA/Isaac-GR00T/pull/320/files) - Original RTC implementation reference
