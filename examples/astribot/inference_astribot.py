import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

# Load your trained model
policy = Gr00tPolicy(
    model_path="outputs/astribot/checkpoint-30000",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # or other embodiment tags
    device="cuda:0",  # or "cpu", or device index like 0
    strict=True  # Enable input/output validation (recommended during development)
)

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

action, info = policy.get_action(observation)
print("action:", action)
print("info:", info)