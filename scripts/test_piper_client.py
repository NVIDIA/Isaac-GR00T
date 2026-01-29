import numpy as np
from gr00t.policy.server_client import PolicyClient


client = PolicyClient(host="127.0.0.1", port=5555, strict=False)

client.reset()

for step in range(10):
    observation = {
        "video": {
            "hand_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
            "third_camera": np.random.randint(0, 256, (1, 1, 480, 640, 3), dtype=np.uint8),
        },
        "state": {
            "joint_states": np.random.rand(1, 1, 6).astype(np.float32),
            "gripper_distance": np.random.rand(1, 1, 1).astype(np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [["pick and place task"]],
        }
    }
    
    action, info = client.get_action(observation)
    print(f"Step {step}: action={action}")
