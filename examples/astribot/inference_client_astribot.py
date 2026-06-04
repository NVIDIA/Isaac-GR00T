import numpy as np
import time
from gr00t.policy.server_client import PolicyClient

# Connect to the policy server
policy = PolicyClient(
    host="localhost",  # or IP address of your GPU server
    port=5555,
    timeout_ms=15000,  # 15 second timeout for inference
    strict=False,      # leave the validation to the server
)

# Verify connection
if not policy.ping():
    raise RuntimeError("Cannot connect to policy server!")

observation = {
    "video": {
        "head": np.zeros((1, 1, 720, 1280, 3), dtype=np.uint8),
        "hand_left": np.zeros((1, 1, 360, 640, 3), dtype=np.uint8),
        "hand_right": np.zeros((1, 1, 360, 640, 3), dtype=np.uint8),
        },  # (1, T, H, W, 3)
    "state": {"arm_left": np.zeros((1, 1, 7), dtype=np.float32),
              "arm_right": np.zeros((1, 1, 7), dtype=np.float32),
              "gripper_left": np.zeros((1, 1, 1), dtype=np.float32),
              "gripper_right": np.zeros((1, 1, 1), dtype=np.float32),
              "head": np.zeros((1, 1, 2), dtype=np.float32),
              "torso": np.zeros((1, 1, 4), dtype=np.float32),
              },  # (1, T, D)
    "language": {"annotation.human.task_description": [["pick up the cube"]]},  # List of length 1
}
ts = time.time()
action, info = policy.get_action(observation)
te = time.time()
print("time:", te - ts)
arm_left_action = action["arm_left"][0,:,:]   #(30, 7)
arm_right_action = action["arm_right"][0,:,:] #(30, 7)
gripper_left_action = action["gripper_left"][0,:,:] #(30, 1)
gripper_right_action = action["gripper_right"][0,:,:] #(30, 1)
head_action = action["head"][0,:,:] #(30, 2)
torso_action = action["torso"][0,:,:] #(30, 4)
print("arm_left_action:", arm_left_action.shape)
print("arm_right_action:", arm_right_action.shape)
print("gripper_left_action:", gripper_left_action.shape)
print("gripper_right_action:", gripper_right_action.shape)
print("head_action:", head_action.shape)
print("torso_action:", torso_action.shape)
print("info:", info)

# # Get modality configs for your embodiment
# modality_configs = policy.get_modality_config()

# # Check what camera keys are expected
# video_keys = modality_configs["video"].modality_keys
# print(f"Expected cameras: {video_keys}")

# # Check video temporal horizon
# video_horizon = len(modality_configs["video"].delta_indices)
# print(f"Video frames needed: {video_horizon}")

# # Check state keys and horizon
# state_keys = modality_configs["state"].modality_keys
# state_horizon = len(modality_configs["state"].delta_indices)
# print(f"Expected states: {state_keys}, horizon: {state_horizon}")

# # Check action keys and horizon
# action_keys = modality_configs["action"].modality_keys
# action_horizon = len(modality_configs["action"].delta_indices)
# print(f"Action outputs: {action_keys}, horizon: {action_horizon}")