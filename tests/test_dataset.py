from pathlib import Path

import numpy as np
import pytest

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.misc import any_describe


@pytest.fixture
def dataset_path():
    import importlib.util

    package_spec = importlib.util.find_spec("gr00t", "")
    assert package_spec is not None
    package_root = package_spec.origin
    assert package_root is not None
    return Path(package_root).parents[1] / "demo_data/robot_sim.PickNPlace"


@pytest.fixture
def modality_configs():
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "state.left_arm",
                "state.right_arm",
                "state.left_hand",
                "state.right_hand",
                "state.waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "action.left_arm",
                "action.right_arm",
                "action.left_hand",
                "action.right_hand",
                "action.waist",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }


@pytest.fixture
def embodiment_tag():
    return EmbodimentTag.GR1


def test_dataset(dataset_path, modality_configs, embodiment_tag):
    # 5. load dataset
    dataset = LeRobotSingleDataset(
        dataset_path,
        modality_configs,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
    )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the first data point
    resp = dataset[0]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
