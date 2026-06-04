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
        "head": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
        "left_wrist": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
        "right_wrist": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
        },  # (1, T, H, W, 3)
    "state": {"arm_left": np.zeros((1, 1, 6), dtype=np.float32),
              "gripper_left": np.zeros((1, 1, 1), dtype=np.float32),
              "arm_right": np.zeros((1, 1, 6), dtype=np.float32),
              "gripper_right": np.zeros((1, 1, 1), dtype=np.float32),
              },  # (1, T, D)
    "language": {"annotation.human.task_description": [["Fold the cloth."]]},  # List of length 1
}
ts = time.time()
action, info = policy.get_action(observation)
te = time.time()
print("time:", te - ts)
arm_left_action = action["arm_left"][0,:,:]   #(30, 7)
arm_right_action = action["arm_right"][0,:,:] #(30, 7)
gripper_left_action = action["gripper_left"][0,:,:] #(30, 1)
gripper_right_action = action["gripper_right"][0,:,:] #(30, 1)
print("arm_left_action:", arm_left_action.shape)
print("arm_right_action:", arm_right_action.shape)
print("gripper_left_action:", gripper_left_action.shape)
print("gripper_right_action:", gripper_right_action.shape)
print("info:", info)
