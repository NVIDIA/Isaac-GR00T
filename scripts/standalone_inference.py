import torch
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

np.set_printoptions(precision=4, suppress=False)

# 加载模型
policy = Gr00tPolicy(
    model_path="/home/lancel/Projects/2024-05-21-Robotics/VLA/Isaac-GR00T/exps/piper_joint_relative_action16_1202_n16/checkpoint-20000",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 查看模型需要的输入格式
modality_config = policy.get_modality_config()
print("Modality config:", modality_config)

# 推理获取动作
for i in range(5):

    joint_states = np.random.rand(1, 1, 6).astype(np.float32)
    gripper_distance = np.random.rand(1, 1, 1).astype(np.float32)

    # 构造 observation（需要根据你的数据格式调整）
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

    predicted_action, info = policy.get_action(observation)
    print(f"Step {i}: action[joint_states]=\n{predicted_action['joint_states']}")

# # 输出动作
# for key, value in predicted_action.items():
#     print(f"{key}: {value.shape}")  # 形状: (B, T, D)

"""
joint_states: (1, 16, 6)
gripper_distance: (1, 16, 1)
"""