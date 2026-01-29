import torch
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

np.set_printoptions(precision=4, suppress=False)

# Load model
policy = Gr00tPolicy(
    model_path="/home/lancel/gr00t/exps/piper_joint_relative_action32_1202_n16/checkpoint-20000",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Avoid RTC fusion inconsistency
policy.processor.clip_outliers = False
policy.processor.state_action_processor.clip_outliers = False


# Check the model's required input format
modality_config = policy.get_modality_config()
print("Modality config:", modality_config)

inference_rtc_frozen_steps = 6 # 4
inference_rtc_overlap_steps = 16 # 8

pridict_horizon = 32 # 16
next_state_index = pridict_horizon - inference_rtc_overlap_steps - 1 # state should be the last state of the previous action.

# Inference to get actions
for i in range(5):

    if i == 0:
        joint_states = np.random.rand(1, 1, 6).astype(np.float32)
        gripper_distance = np.random.rand(1, 1, 1).astype(np.float32)
    else:
        joint_states = predicted_action['joint_states'][:, next_state_index: next_state_index + 1, :]
        gripper_distance = predicted_action['gripper_distance'][:,next_state_index: next_state_index + 1, :]

    # Construct observation (adjust according to your data format)
    observation = {
        "video": {
            "hand_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
            "third_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
        },
        "state": {
            "joint_states": joint_states,
            "gripper_distance": gripper_distance,
        },
        "language": {
            "annotation.human.action.task_description": [["pick and place task"]],
        }
    }

    if i > 0:
        # Pad to the same shape as the original predicted_action
        original_joint_shape = predicted_action['joint_states'].shape
        original_gripper_shape = predicted_action['gripper_distance'].shape

        pad_joint = np.zeros(original_joint_shape, dtype=predicted_action['joint_states'].dtype)
        pad_gripper = np.zeros(original_gripper_shape, dtype=predicted_action['gripper_distance'].dtype)

        # Take overlap and pad to the back
        valid_joint = predicted_action['joint_states'][:, pridict_horizon - inference_rtc_overlap_steps : pridict_horizon, :]
        valid_gripper = predicted_action['gripper_distance'][:, pridict_horizon - inference_rtc_overlap_steps : pridict_horizon, :]

        pad_joint[:, :inference_rtc_overlap_steps, :] = valid_joint
        pad_gripper[:, :inference_rtc_overlap_steps, :] = valid_gripper

        predicted_action['joint_states'] = pad_joint
        predicted_action['gripper_distance'] = pad_gripper
        
        observation['action'] = {
            "joint_states": predicted_action['joint_states'],
            "gripper_distance": predicted_action['gripper_distance'],
        }

    predicted_action, info = policy.get_action(observation)
    print(f"Step {i}: action[joint_states]=\n{predicted_action['joint_states']}")

# # Output actions
# for key, value in predicted_action.items():
#     print(f"{key}: {value.shape}")  # Shape: (B, T, D)

"""
joint_states: (1, 16, 6)
gripper_distance: (1, 16, 1)
"""