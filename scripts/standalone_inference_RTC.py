"""
Standalone inference script for GR00T policy with RTC (Real-Time Control) mode.

This script demonstrates how to:
1. Load a trained GR00T policy model
2. Perform sequential inference with RTC overlap
3. Visualize joint trajectories across inference steps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


# ==================== Configuration ====================
MODEL_PATH = "Your_absolute_path_to_the_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RTC parameters
INFERENCE_RTC_FROZEN_STEPS = 6
INFERENCE_RTC_OVERLAP_STEPS = 16
PREDICT_HORIZON = 32
NUM_INFERENCE_STEPS = 5

# Plotting parameters
PLOT_COLORS = [
    (0.8, 0.2, 0.2),   # Red
    (0.2, 0.6, 0.2),   # Green
    (0.2, 0.4, 0.8),   # Blue
    (0.8, 0.5, 0.1),   # Orange
    (0.6, 0.2, 0.8),   # Purple
]
OUTPUT_FIGURE_PATH = "joint_trajectories_RTC.png"


def load_policy(model_path: str, device: str) -> Gr00tPolicy:
    """Load and configure the GR00T policy model."""
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device=device,
    )
    
    # Disable outlier clipping to avoid RTC fusion inconsistency
    policy.processor.clip_outliers = False
    policy.processor.state_action_processor.clip_outliers = False
    
    return policy


def create_observation(
    joint_states: np.ndarray,
    gripper_distance: np.ndarray,
    predicted_action: dict = None,
    predict_horizon: int = PREDICT_HORIZON,
    overlap_steps: int = INFERENCE_RTC_OVERLAP_STEPS,
) -> dict:
    """
    Create observation dictionary for policy inference.
    
    Args:
        joint_states: Current joint states, shape (1, 1, 6)
        gripper_distance: Current gripper distance, shape (1, 1, 1)
        predicted_action: Previous predicted action for RTC overlap (optional)
        predict_horizon: Prediction horizon length
        overlap_steps: Number of overlap steps for RTC
    
    Returns:
        Observation dictionary ready for policy inference
    """
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
    
    # Add RTC overlap action if available
    if predicted_action is not None:
        original_joint_shape = predicted_action['joint_states'].shape
        original_gripper_shape = predicted_action['gripper_distance'].shape

        pad_joint = np.zeros(original_joint_shape, dtype=predicted_action['joint_states'].dtype)
        pad_gripper = np.zeros(original_gripper_shape, dtype=predicted_action['gripper_distance'].dtype)

        # Extract overlap portion and pad to the beginning
        valid_joint = predicted_action['joint_states'][:, predict_horizon - overlap_steps:predict_horizon, :]
        valid_gripper = predicted_action['gripper_distance'][:, predict_horizon - overlap_steps:predict_horizon, :]

        pad_joint[:, :overlap_steps, :] = valid_joint
        pad_gripper[:, :overlap_steps, :] = valid_gripper

        observation['action'] = {
            "joint_states": pad_joint,
            "gripper_distance": pad_gripper,
        }
    
    return observation


def run_inference(
    policy: Gr00tPolicy,
    num_steps: int,
    predict_horizon: int,
    overlap_steps: int,
) -> list:
    """
    Run sequential inference with RTC overlap.
    
    Args:
        policy: Loaded GR00T policy
        num_steps: Number of inference steps to run
        predict_horizon: Prediction horizon length
        overlap_steps: Number of overlap steps for RTC
    
    Returns:
        List of predicted joint trajectories
    """
    all_trajectories = []
    next_state_index = predict_horizon - overlap_steps - 1
    predicted_action = None
    
    for step in range(num_steps):
        # Initialize or get state from previous prediction
        if step == 0:
            joint_states = np.random.rand(1, 1, 6).astype(np.float32)
            gripper_distance = np.random.rand(1, 1, 1).astype(np.float32)
        else:
            joint_states = predicted_action['joint_states'][:, next_state_index:next_state_index + 1, :]
            gripper_distance = predicted_action['gripper_distance'][:, next_state_index:next_state_index + 1, :]

        # Create observation
        observation = create_observation(
            joint_states=joint_states,
            gripper_distance=gripper_distance,
            predicted_action=predicted_action if step > 0 else None,
            predict_horizon=predict_horizon,
            overlap_steps=overlap_steps,
        )

        # Run inference
        predicted_action, info = policy.get_action(observation)
        print(f"Step {step}: action[joint_states]=\n{predicted_action['joint_states']}")
        
        # Store trajectory
        all_trajectories.append(predicted_action['joint_states'].copy())
    
    return all_trajectories


def plot_trajectories(
    trajectories: list,
    predict_horizon: int,
    overlap_steps: int,
    colors: list,
    output_path: str,
) -> None:
    """
    Plot joint trajectories across inference steps.
    
    Args:
        trajectories: List of joint trajectories from each inference step
        predict_horizon: Prediction horizon length
        overlap_steps: Number of overlap steps for RTC
        colors: List of colors for each step
        output_path: Path to save the figure
    """
    num_joints = 6
    num_steps = len(trajectories)
    joint_names = [f'Joint {j+1}' for j in range(num_joints)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        for step_idx, traj in enumerate(trajectories):
            joint_traj = traj[0, :, joint_idx]
            T = len(joint_traj)
            
            # Calculate time offset
            time_offset = 0 if step_idx == 0 else step_idx * (predict_horizon - overlap_steps)
            time_points = np.arange(T) + time_offset
            
            base_color = colors[step_idx % len(colors)]
            
            # Plot trajectory
            ax.plot(
                time_points, joint_traj,
                color=base_color, linewidth=2, linestyle='--', alpha=0.8,
                label=f'Step {step_idx}' if joint_idx == 0 else None
            )
            
            # Mark start (circle) and end (triangle)
            ax.scatter(time_points[0], joint_traj[0], color=base_color, s=80,
                      marker='o', zorder=6, edgecolors='black', linewidths=1)
            ax.scatter(time_points[-1], joint_traj[-1], color=base_color, s=80,
                      marker='^', zorder=6, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Joint Value', fontsize=10)
        ax.set_title(joint_names[joint_idx], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], linewidth=2, linestyle='--', alpha=0.8, label=f'Step {i}')
        for i in range(num_steps)
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle(
        'Joint Trajectories Across Inference Steps\n(Circle: start, Triangle: end)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to '{output_path}'")


def main():
    """Main entry point for the inference script."""
    np.set_printoptions(precision=4, suppress=False)
    
    # Load model
    print("Loading policy model...")
    policy = load_policy(MODEL_PATH, DEVICE)
    
    # Print modality config
    modality_config = policy.get_modality_config()
    print("Modality config:", modality_config)
    
    # Run inference
    print(f"\nRunning {NUM_INFERENCE_STEPS} inference steps...")
    trajectories = run_inference(
        policy=policy,
        num_steps=NUM_INFERENCE_STEPS,
        predict_horizon=PREDICT_HORIZON,
        overlap_steps=INFERENCE_RTC_OVERLAP_STEPS,
    )
    
    # Plot results
    print("\nPlotting trajectories...")
    plot_trajectories(
        trajectories=trajectories,
        predict_horizon=PREDICT_HORIZON,
        overlap_steps=INFERENCE_RTC_OVERLAP_STEPS,
        colors=PLOT_COLORS,
        output_path=OUTPUT_FIGURE_PATH,
    )


if __name__ == "__main__":
    main()
