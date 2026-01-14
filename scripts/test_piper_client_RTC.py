import numpy as np
from gr00t.policy.server_client import PolicyClient


client = PolicyClient(host="127.0.0.1", port=5555, strict=False)

client.reset()


# RTC 参数配置
inference_rtc_frozen_steps = 4
inference_rtc_overlap_steps = 8
pridict_horizon = 16
next_state_index = pridict_horizon - inference_rtc_overlap_steps - 1  # state should be the last state of the previous action.

predicted_action = None

for step in range(10):
    
    if step == 0:
        joint_states = np.random.rand(1, 1, 6).astype(np.float32)
        gripper_distance = np.random.rand(1, 1, 1).astype(np.float32)
    else:
        joint_states = predicted_action['joint_states'][:, next_state_index: next_state_index + 1, :]
        gripper_distance = predicted_action['gripper_distance'][:, next_state_index: next_state_index + 1, :]

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

    if step > 0:
        # padding 到和原始 predicted_action 一样的 shape
        original_joint_shape = predicted_action['joint_states'].shape
        original_gripper_shape = predicted_action['gripper_distance'].shape

        pad_joint = np.zeros(original_joint_shape, dtype=predicted_action['joint_states'].dtype)
        pad_gripper = np.zeros(original_gripper_shape, dtype=predicted_action['gripper_distance'].dtype)

        # 取 overlap，pad 到后面
        valid_joint = predicted_action['joint_states'][:, pridict_horizon - inference_rtc_overlap_steps : pridict_horizon, :]
        valid_gripper = predicted_action['gripper_distance'][:, pridict_horizon - inference_rtc_overlap_steps : pridict_horizon, :]

        pad_joint[:, :inference_rtc_overlap_steps, :] = valid_joint
        pad_gripper[:, :inference_rtc_overlap_steps, :] = valid_gripper

        observation['action'] = {
            "joint_states": pad_joint,
            "gripper_distance": pad_gripper,
        }
    
    predicted_action, info = client.get_action(observation)
    print(f"Step {step}: action[joint_states]=\n{predicted_action['joint_states']}")