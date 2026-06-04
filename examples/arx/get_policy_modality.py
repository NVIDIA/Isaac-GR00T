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


# Get modality configs for your embodiment
modality_configs = policy.get_modality_config()

# Check what camera keys are expected
video_keys = modality_configs["video"].modality_keys
print(f"Expected cameras: {video_keys}")

# Check video temporal horizon
video_horizon = len(modality_configs["video"].delta_indices)
print(f"Video frames needed: {video_horizon}")

# Check state keys and horizon
state_keys = modality_configs["state"].modality_keys
state_horizon = len(modality_configs["state"].delta_indices)
print(f"Expected states: {state_keys}, horizon: {state_horizon}")

# Check action keys and horizon
action_keys = modality_configs["action"].modality_keys
action_horizon = len(modality_configs["action"].delta_indices)
print(f"Action outputs: {action_keys}, horizon: {action_horizon}")


print('modality:', modality_configs)


# {'video': 
# ModalityConfig(delta_indices=[-15, 0], 
#                 modality_keys=['exterior_image_1_left', 'wrist_image_left'], 
#                 sin_cos_embedding_keys=None, 
#                 mean_std_embedding_keys=None, 
#                 action_configs=None), 
# 'state': ModalityConfig(delta_indices=[0], modality_keys=['eef_9d', 'gripper_position', 'joint_position'], sin_cos_embedding_keys=None, mean_std_embedding_keys=None, action_configs=None), 
# 'action': ModalityConfig(delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], 
#                         modality_keys=['eef_9d', 'gripper_position', 'joint_position'], 
#                         sin_cos_embedding_keys=None, 
#                         mean_std_embedding_keys=None, 
#                         action_configs=[
#                             ActionConfig(
#                                 rep=<ActionRepresentation.RELATIVE: 'relative'>, 
#                                 type=<ActionType.EEF: 'eef'>, 
#                                 format=<ActionFormat.XYZ_ROT6D: 'xyz+rot6d'>, 
#                                 state_key='eef_9d'), 
#                             ActionConfig(
#                                 rep=<ActionRepresentation.ABSOLUTE: 'absolute'>, 
#                                 type=<ActionType.NON_EEF: 'non_eef'>, 
#                                 ormat=<ActionFormat.DEFAULT: 'default'>, 
#                                 state_key='gripper_position'), 
#                             ActionConfig(
#                                 rep=<ActionRepresentation.RELATIVE: 'relative'>, 
#                                 type=<ActionType.NON_EEF: 'non_eef'>, 
#                                 format=<ActionFormat.DEFAULT: 'default'>, 
#                                 state_key='joint_position')]), 
# 'language': ModalityConfig(delta_indices=[0], 
#                             modality_keys=['annotation.language.language_instruction'], 
#                             sin_cos_embedding_keys=None, 
#                             mean_std_embedding_keys=None, 
#                             action_configs=None)}