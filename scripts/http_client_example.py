import time

import json_numpy
import numpy as np
import requests

json_numpy.patch()

obs = {
    "video.ego_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
    "state.left_arm": np.random.rand(1, 7),
    "state.right_arm": np.random.rand(1, 7),
    "state.left_hand": np.random.rand(1, 6),
    "state.right_hand": np.random.rand(1, 6),
    "state.waist": np.random.rand(1, 3),
    "annotation.human.action.task_description": ["do your thing!"],
}


t = time.time()
response = requests.post(
    "http://0.0.0.0:8000/act",
    # "http://159.223.171.199:44989/act",   # Bore tunnel
    json={"observation": obs},
)
print(f"used time {time.time() - t}")
action = response.json()
print(action)
