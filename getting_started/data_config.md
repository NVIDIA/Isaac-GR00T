# How to prepare your modality configuration

## Overview

The modality configuration defines how your robot's data should be loaded, processed, and interpreted by the model. This configuration bridges your dataset's physical structure (defined in `meta/modality.json`) and the model's data processing pipeline.

Each embodiment requires a Python configuration file that specifies:
- Which observations to use (video cameras, proprioceptive states)
- How to sample data temporally (current frame, historical frames, future action horizons)
- How actions should be interpreted and transformed
- Which language annotations to use

## Configuration Structure

A modality configuration is a Python dictionary containing four top-level keys: `"video"`, `"state"`, `"action"`, and `"language"`. Each key maps to a `ModalityConfig` object.

Here's the [SO-100 example](../examples/SO100/so100_config.py):

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

so100_config = {
    "video": ModalityConfig(...),
    "state": ModalityConfig(...),
    "action": ModalityConfig(...),
    "language": ModalityConfig(...),
}

register_modality_config(so100_config)
```

## Understanding `ModalityConfig`

Each `ModalityConfig` specifies two required fields and several optional ones:

### Required Fields

**1. `delta_indices` (list[int])**

Defines which temporal offsets to sample relative to the current timestep. This enables:
- **Historical context**: Use negative indices (e.g., `[-2, -1, 0]`) to include past observations
- **Current observation**: Use `[0]` for the current timestep
- **Future actions**: Use positive indices (e.g., `list(range(0, 16))`) for action prediction horizons

Examples:
```python
# Single current frame for video
delta_indices=[0]

# Last 3 frames for video (temporal stacking)
delta_indices=[-2, -1, 0]

# 16-step action prediction horizon
delta_indices=list(range(0, 16))
```

**2. `modality_keys` (list[str])**

Specifies which keys to load from your dataset. These keys **must match** the keys defined in your `meta/modality.json` file.

For the SO-100 example:
- **Video keys**: Must match keys in `meta/modality.json` under `"video"` (e.g., `"front"`, `"wrist"`)
- **State keys**: Must match keys in `meta/modality.json` under `"state"` (e.g., `"single_arm"`, `"gripper"`)
- **Action keys**: Must match keys in `meta/modality.json` under `"action"` (e.g., `"single_arm"`, `"gripper"`)
- **Language keys**: Must match keys in `meta/modality.json` under `"annotation"` (e.g., `"annotation.human.action.task_description"`)

### Optional Fields

**3. `sin_cos_embedding_keys` (list[str] | None)**

Specifies which state keys should use sine/cosine encoding. Best for dimensions that are in radians (e.g., joint angles). If not specified, min-max normalization is used. Note that this will duplicate the number of dimensions by 2, and is only recommended for proprioceptive states.

```python
"state": ModalityConfig(
    delta_indices=[0],
    modality_keys=["single_arm", "gripper"],
    sin_cos_embedding_keys=["single_arm"],  # Apply sin/cos to joint angles
)
```

**4. `mean_std_embedding_keys` (list[str] | None)**

Specifies which keys should use mean/standard deviation normalization instead of min-max normalization.

**5. `action_configs` (list[ActionConfig] | None)**

Required for the `"action"` modality. Defines how each action modality should be interpreted and transformed. The list must have the same length as `modality_keys`, and each element corresponds to the action modality for the corresponding `modality_key`. See more details in the [Action Modality](#understanding-actionconfig) section.

## Configuring Each Modality

### Video Modality

Defines which camera views to use:

```python
"video": ModalityConfig(
    delta_indices=[0],  # Current frame only
    modality_keys=[
        "front",  # Must match a key in meta/modality.json under "video"
    ],
)
```

For multiple cameras:
```python
"video": ModalityConfig(
    delta_indices=[0],
    modality_keys=["front", "wrist"],
)
```

### State Modality

Defines proprioceptive observations (joint positions, gripper states, etc.):

```python
"state": ModalityConfig(
    delta_indices=[0],  # Current state
    modality_keys=[
        "single_arm",      # Must match keys in meta/modality.json under "state"
        "gripper",
    ],
)
```

### Action Modality

Defines the action space and prediction horizon:

```python
"action": ModalityConfig(
    delta_indices=list(range(0, 16)),  # Predict 16 steps into the future
    modality_keys=[
        "single_arm",      # Must match keys in meta/modality.json under "action"
        "gripper",
    ],
    action_configs=[
        # One ActionConfig per modality_key
        # single_arm
        ActionConfig(
            rep=ActionRepresentation.RELATIVE,  # relative control of the single arm
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
        # gripper
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,  # absolute control of the gripper
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        ),
    ],
)
```

#### Understanding `ActionConfig`

Each `ActionConfig` has three required fields and one optional field:

**1. `rep` (ActionRepresentation)**

Defines how actions should be interpreted:
- `RELATIVE`: Actions are deltas from the current state (introduced in the UMI paper)
- `ABSOLUTE`: Actions are target positions

Using relative actions will lead to smoother actions, but might suffer from drifting. If you want to use relative actions, please make sure the state and action stored in the dataset are absolute, and the absolute to relative will be handled in the processor.

**2. `type` (ActionType)**

Specifies the control space:
- `EEF`: End-effector/Cartesian space control (Expecting a 9-dimensional vector: x, y, z positions + rotation 6D)
- `NON_EEF`: Joint space control and other non-EEF control spaces (joint angles, positions, gripper positions, etc.)

**3. `format` (ActionFormat)**

Defines the action representation format:
- `DEFAULT`: Standard format (e.g., joint angles, gripper positions)
- `XYZ_ROT6D`: 3D position + 6D rotation representation for end-effector control
- `XYZ_ROTVEC`: 3D position + rotation vector for end-effector control

**4. `state_key` (str | None)**

Optional. Specifies the corresponding reference state key for computing relative actions when `rep=RELATIVE`. If not provided, the system will use the action key as the reference state key.

Example with `state_key`:
```python
"joint_pos_action_left": ActionConfig(
    rep=ActionRepresentation.RELATIVE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
    state_key="joint_pos_obs_left",  # Use this state to compute relative action
)
```

### Language Modality

Defines which language annotations to use:

```python
"language": ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.action.task_description"],  # Must match annotation keys in meta/modality.json
)
```

## Complete Example: SO-100

Here's the complete SO-100 configuration with explanations:

```python
so100_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}
```

## Key Relationships with `meta/modality.json`

The modality configuration's `modality_keys` must reference keys that exist in your dataset's `meta/modality.json`:

**Example `meta/modality.json`:**
```json
{
    "state": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "action": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "video": {
        "front": {"original_key": "observation.images.front"},
        "wrist": {"original_key": "observation.images.wrist"},
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
```

The system will:
1. Use `modality_keys` to look up the corresponding entries in `meta/modality.json`
2. Extract the correct slices from the concatenated state/action arrays
3. Apply the specified transformations (normalization, action representation conversion)

## Registering Your Configuration

After defining your configuration, register it so it's available to the training and inference pipelines:

```python
from gr00t.configs.data.embodiment_configs import register_modality_config

your_modality_config = {
    ...
}

register_modality_config(your_modality_config)
```

Save your configuration to a Python file and pass the path to the `modality_config_path` argument when running the finetuning script.
