from gr00t.eval.service import ExternalRobotInferenceClient
import numpy as np


print("Creating client...")
client = ExternalRobotInferenceClient(host="localhost", port=8000)

print("Preparing observation data...")
fake_observations = {
    "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
    "state.right_arm": np.expand_dims(np.random.rand(7), axis=0),
    "state.left_arm": np.expand_dims(np.random.rand(7), axis=0),
    "state.left_hand": np.expand_dims(np.random.rand(6), axis=0),
    "state.right_hand": np.expand_dims(np.random.rand(6), axis=0),
    "annotation.human.task_description": ["{'Make the chicken dance.'}"],
}

print("Calling inference...")
print(f"Input shapes: video.ego_view={fake_observations['video.ego_view'].shape}")
print(f"State shapes: right_arm={fake_observations['state.right_arm'].shape}, left_arm={fake_observations['state.left_arm'].shape}, left_hand={fake_observations['state.left_hand'].shape}, right_hand={fake_observations['state.right_hand'].shape}")

try:
    action = client.get_action(fake_observations)
    print("Success! Got action:")
    print(f"Action type: {type(action)}")
    if hasattr(action, 'shape'):
        print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
except Exception as e:
    print(f"Error during inference: {e}")
    import traceback
    traceback.print_exc()
