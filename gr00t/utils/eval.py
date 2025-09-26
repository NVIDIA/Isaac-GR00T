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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    execution_horizon=16,
    action_horizon=16,
    inference_latency_steps=0,
    plot=False,
    plot_state=False,
    save_plot_path=None,
    rtc_enabled=False,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    prev_action = None

    chunk_pred_action = []
    # For example, we have action_horizon of 16, execution_horizon of 10, and inference_latency_steps of 4
    # then the intermediate_overlap_steps is 2, which is the overlap between the first and second chunk
    intermediate_overlap_steps = action_horizon - execution_horizon - inference_latency_steps

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % execution_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            # TODO: hack all actions in the data_point to be the same as the previous action
            # remove action.*** from data_point to new_data_point
            new_data_point = {k: v for k, v in data_point.items() if not k.startswith("action.")}

            # print("inferencing at step: ", step_count)
            # This is used by RTC
            if prev_action is not None:
                # combine dict of prev_action and new_data_point
                # add one dimension to prev_action
                new_data_point = {**prev_action, **new_data_point}

            action_chunk = policy.get_action(new_data_point)
            action_horizon = len(action_chunk[f"action.{modality_keys[0]}"])

            prev_action = action_chunk
            for j in range(execution_horizon):
                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in modality_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)

                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                if rtc_enabled or inference_latency_steps == 0:
                    pred_action_across_time.append(concat_pred_action)

            for j in range(action_horizon):
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                chunk_pred_action.append(concat_pred_action)

                # when without rtc and we wanna visualize how latency affects the actual action,
                # we will still run the action_steps during the t+1 inference latency steps
                if not rtc_enabled and inference_latency_steps > 0:
                    if step_count < execution_horizon:
                        if j < execution_horizon + inference_latency_steps:
                            pred_action_across_time.append(concat_pred_action)
                    # only append the action_horizon - action_horizon steps
                    elif inference_latency_steps <= j < action_horizon - intermediate_overlap_steps:
                        pred_action_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, print(
        gt_action_across_time.shape, pred_action_across_time.shape
    )

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "mse": mse,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "execution_horizon": execution_horizon,
            "inference_latency_steps": inference_latency_steps,
            "steps": steps,
            "chunk_pred_action": chunk_pred_action,
            "rtc_enabled": rtc_enabled,
        }
        plot_trajectory(info, save_plot_path)

    return mse


def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    mse = info["mse"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]
    execution_horizon = info["execution_horizon"]
    inference_latency_steps = info["inference_latency_steps"]
    rtc_enabled = info["rtc_enabled"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {mse:.6f}"
    title_text += f"\nlatency steps: {inference_latency_steps} | execution horizon: {execution_horizon} | RTC: {rtc_enabled}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.96)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # Colorize overlap regions where multiple chunks predict for the same time steps
        intermediate_overlap_steps = action_horizon - execution_horizon - inference_latency_steps
        for step_idx, inference_start in enumerate(
            range(execution_horizon, steps, execution_horizon)
        ):
            if inference_start < steps:
                inference_end = inference_start + inference_latency_steps
                ax.axvspan(
                    inference_start,
                    inference_end,
                    alpha=0.2,
                    color="lightcoral",
                    label="inference latency" if step_idx == 0 else "",
                )
                ax.axvspan(
                    inference_end,
                    inference_end + intermediate_overlap_steps,
                    alpha=0.2,
                    color="lightblue",
                    label="intermediate overlap" if step_idx == 0 else "",
                )

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for j in range(0, steps, execution_horizon):
            ax.plot(
                j,
                gt_action_across_time[j, i],
                "ro",
                markersize=4,
                label="inference point" if j == 0 else "",
            )

        # plot chunk_pred_action with alternating colors between chunks
        chunk_pred_action_array = np.array(info["chunk_pred_action"])
        if len(chunk_pred_action_array) > 0:
            colors = ["green", "lightgreen"]
            for idx, step in enumerate(range(0, steps, execution_horizon)):
                chunk_start = idx * action_horizon
                chunk_end = min(chunk_start + action_horizon, len(chunk_pred_action_array))
                chunk_data = chunk_pred_action_array[chunk_start:chunk_end]
                if len(chunk_data) == 0:
                    continue
                chunk_time_steps = np.arange(step, step + len(chunk_data))
                color = colors[idx % len(colors)]
                label = "chunk pred action" if idx == 0 else None
                ax.plot(
                    chunk_time_steps,
                    chunk_data[:, i],
                    "o",
                    color=color,
                    label=label,
                    markersize=1,
                    alpha=0.8,
                )

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
