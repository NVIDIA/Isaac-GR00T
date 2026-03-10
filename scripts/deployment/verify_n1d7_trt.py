#!/usr/bin/env python3
"""Quick verification: compare PyTorch vs TRT action head outputs for N1.7."""

import os
import sys

import torch
from torch.nn.functional import cosine_similarity


# Make sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_onnx_n1d7 import prepare_observation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--dataset_path", type=str, default="demo_data/gr1.PickNPlace")
    parser.add_argument("--engine_dir", type=str, default="./gr00t_n1d7_engines")
    parser.add_argument(
        "--mode",
        type=str,
        default="action_head",
        choices=["action_head", "n17_full_pipeline"],
        help="TRT setup mode",
    )
    parser.add_argument("--embodiment_tag", type=EmbodimentTag, default=EmbodimentTag.GR1)
    args = parser.parse_args()

    print("=" * 60)
    print("N1.7 TRT Action Head Verification")
    print("=" * 60)

    # Step 1: Load policy and get PyTorch reference output
    print("\n[1] Loading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )

    print("[2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    print("[3] Running PyTorch inference...")
    obs = prepare_observation(policy, dataset, traj_idx=0)
    torch.manual_seed(42)
    with torch.inference_mode():
        result = policy.get_action(obs)

    # get_action returns (action_dict, info_dict)
    action_dict = result[0] if isinstance(result, tuple) else result
    print(f"  Action keys: {list(action_dict.keys())}")

    # Concatenate all action arrays into a single tensor for comparison
    pt_arrays = []
    for k in sorted(action_dict.keys()):
        v = action_dict[k]
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        pt_arrays.append(t.float().flatten())
        print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else len(v)}")
    pt_action = torch.cat(pt_arrays)

    # Step 2: Setup TRT engines and run
    print("\n[4] Loading TRT engines...")
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy, args.engine_dir, mode=args.mode)

    print("[5] Running TRT inference...")
    obs2 = prepare_observation(policy, dataset, traj_idx=0)
    torch.manual_seed(42)
    with torch.inference_mode():
        result2 = policy.get_action(obs2)

    action_dict2 = result2[0] if isinstance(result2, tuple) else result2
    trt_arrays = []
    for k in sorted(action_dict2.keys()):
        v = action_dict2[k]
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        trt_arrays.append(t.float().flatten())
    trt_act = torch.cat(trt_arrays)

    # Step 3: Compare
    print("\n[6] Comparing outputs...")
    pt_flat = pt_action.float().flatten()
    trt_flat = trt_act.float().flatten()

    cosine = cosine_similarity(pt_flat.unsqueeze(0), trt_flat.unsqueeze(0)).item()
    l1 = (pt_flat - trt_flat).abs().mean().item()
    linf = (pt_flat - trt_flat).abs().max().item()

    print(f"\n  Cosine Similarity: {cosine:.6f}")
    print(f"  L1 Mean Error:     {l1:.6f}")
    print(f"  L∞ Max Error:      {linf:.6f}")

    if cosine > 0.999:
        print("\n  ✓ PASS — TRT action head matches PyTorch")
    elif cosine > 0.99:
        print("\n  ~ WARN — Minor drift detected")
    else:
        print("\n  ✗ FAIL — Significant divergence")


if __name__ == "__main__":
    main()
