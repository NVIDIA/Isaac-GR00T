# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a replication of the notebook `getting_started/load_dataset.ipynb`
"""

import argparse
import json
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.misc import any_describe


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


def load_dataset(dataset_path: str, embodiment_tag: str, video_backend: str = "decord"):
    # 1. get modality keys
    dataset_path = pathlib.Path(dataset_path)
    modality_keys_dict = get_modality_keys(dataset_path)
    video_modality_keys = modality_keys_dict["video"]
    language_modality_keys = modality_keys_dict["annotation"]
    state_modality_keys = modality_keys_dict["state"]
    action_modality_keys = modality_keys_dict["action"]

    pprint(f"Valid modality_keys for debugging:: {modality_keys_dict} \n")

    print(f"state_modality_keys: {state_modality_keys}")
    print(f"action_modality_keys: {action_modality_keys}")

    # 2. modality configs
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=video_modality_keys,  # we will include all video modalities
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_modality_keys,
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=action_modality_keys,
        ),
    }

    # 3. language modality config (if exists)
    if language_modality_keys:
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=language_modality_keys,
        )

    # 4. gr00t embodiment tag
    embodiment_tag: EmbodimentTag = EmbodimentTag(embodiment_tag)

    # 5. load dataset
    dataset = LeRobotSingleDataset(
        dataset_path,
        modality_configs,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
    )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the 7th data point
    resp = dataset[7]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 6. plot the first 100 images
    images_list = []
    video_key = video_modality_keys[0]  # we will use the first video modality

    for i in range(100):
        if i % 10 == 0:
            resp = dataset[i]
            img = resp[video_key][0]
            images_list.append(img)

    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i}")
    plt.tight_layout()  # adjust the subplots to fit into the figure area.
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Robot Dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data/robot_sim.PickNPlace",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="gr1",
        help="Full list of embodiment tags can be found in gr00t.data.schema.EmbodimentTag",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="decord",
        choices=["decord", "torchvision_av"],
        help="Backend to use for video loading, use torchvision_av for av encoded videos",
    )
    args = parser.parse_args()
    load_dataset(args.data_path, args.embodiment_tag, args.video_backend)
