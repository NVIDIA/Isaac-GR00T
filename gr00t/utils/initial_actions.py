from pathlib import Path

import numpy as np


INITIAL_ACTIONS_FILENAME = "initial_actions.npz"


def save_initial_actions(
    initial_actions: dict[str, dict[str, np.ndarray]], initial_actions_path: str | Path
):
    np.savez(str(initial_actions_path), initial_actions)


def load_initial_actions(initial_actions_path: str | Path):
    """
    initial_actions: list[dict[str, dict[str, np.ndarray]]]
    0: (the first dataset)
        trajectory_name:
          action_key:
            action: np.ndarray
    1: (the second dataset)
        ...
    """
    initial_actions_npz = np.load(str(initial_actions_path), allow_pickle=True)
    initial_actions = []
    initial_actions_array = initial_actions_npz[
        "arr_0"
    ]  # This is the default key when np.savez saves a list
    for dataset_initial_actions in initial_actions_array:
        initial_actions_for_this_dataset = {}
        for trajectory_name, action_dict in dataset_initial_actions.items():
            initial_actions_for_this_dataset[trajectory_name] = action_dict
        initial_actions.append(initial_actions_for_this_dataset)
    return initial_actions
