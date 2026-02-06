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

import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import ModalityConfig

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ['joint_position'])
    """Modality keys to evaluate."""

    data_config: str = "none"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: int = 150
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory to evaluate."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = './plot.png'
    """Path to save the plot."""

    plot_state: bool = False
    """Whether to plot the state."""


def main(args: ArgsConfig):

    video_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.camera"],
    )

    state_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=['state.joint_state_position',
                       'state.joint_state_velocity',
                       'state.insertion_position',
                       'state.insertion_rotation'],
    )

    action_modality = ModalityConfig(
        delta_indices=[0],  # 8 future action steps
        modality_keys=["action.joint_position"],
    )

    language_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    )

    modality_configs = {
        "video": video_modality,
        "state": state_modality,
        "action": action_modality,
        "language": language_modality,
    }

    
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.model.transforms import GR00TTransform


    # select the transforms you want to apply to the data
    transforms = ComposedModalityTransform(
        transforms=[
            # video transforms
            VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
            VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
            VideoResize(apply_to=video_modality.modality_keys, height=224, width=224,
                        interpolation="linear", backend="torchvision" ),
            VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3,
                             contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
            VideoToNumpy(apply_to=video_modality.modality_keys),

            # state transforms
            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                "state.joint_state_position": "min_max",
                "state.joint_state_velocity": "min_max",
                "state.insertion_position": "min_max",
                "state.insertion_rotation": "min_max",
            }),

            # action transforms
            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                "action.joint_position": "min_max",
            }),

            # ConcatTransform
            ConcatTransform(
                video_concat_order=video_modality.modality_keys,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(state_modality.delta_indices),
                action_horizon=len(action_modality.delta_indices),
                max_state_dim=64,
                max_action_dim=6,
            ),
        ]
    )
    
    args.action_horizon = len(action_modality.delta_indices)
    
    if args.model_path is not None:
        import torch

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_configs,
            modality_transform=transforms,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        print("Running trajectory:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=args.save_plot_path,
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
