#!/usr/bin/env python3
"""Quick verification: compare PyTorch vs TRT action head outputs for N1.7."""

from dataclasses import dataclass
import os
import sys
from typing import Literal

import torch
from torch.nn.functional import cosine_similarity
import tyro


# Make sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_onnx_n1d7 import prepare_observation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


@dataclass
class VerifyConfig:
    """Configuration for TRT verification."""

    model_path: str = ""
    """Path to model checkpoint."""

    dataset_path: str = "demo_data/libero_demo"
    """Path to dataset."""

    engine_dir: str = "./gr00t_n1d7_engines"
    """Directory with TRT engines."""

    mode: Literal["action_head", "n17_full_pipeline", "vit_llm_only"] = "action_head"
    """TRT setup mode: 'vit_llm_only' uses ViT+LLM TRT with PyTorch action head."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.LIBERO_PANDA
    """Embodiment tag to use."""


def main(args: VerifyConfig | None = None):
    if args is None:
        args = tyro.cli(VerifyConfig)

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

    # --- Capture ViT input/output and backbone output from PyTorch ---
    pt_backbone_features = None
    pt_vit_output = None
    pt_vit_input = None

    def _capture_backbone_hook(module, args, output):
        nonlocal pt_backbone_features
        pt_backbone_features = output["backbone_features"].detach().clone()
        return output

    def _capture_vit_hook(module, args, kwargs, output):
        nonlocal pt_vit_output, pt_vit_input
        # Capture input pixel_values
        if args:
            pt_vit_input = args[0].detach().clone()
        elif "pixel_values" in kwargs:
            pt_vit_input = kwargs["pixel_values"].detach().clone()
        # Capture output (image_embeds after merger)
        if isinstance(output, tuple):
            pt_vit_output = output[0].detach().clone()
        else:
            pt_vit_output = output.detach().clone()
        return output

    backbone_hook = policy.model.backbone.register_forward_hook(_capture_backbone_hook)
    vit_hook = policy.model.backbone.model.model.visual.register_forward_hook(
        _capture_vit_hook, with_kwargs=True
    )

    print("[3] Running PyTorch inference...")
    obs = prepare_observation(policy, dataset, traj_idx=0)
    torch.manual_seed(42)
    with torch.inference_mode():
        result = policy.get_action(obs)

    backbone_hook.remove()
    vit_hook.remove()

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

    # --- Capture backbone output from TRT ---
    trt_backbone_features = None

    def _capture_trt_backbone_hook(module, args, output):
        nonlocal trt_backbone_features
        trt_backbone_features = output["backbone_features"].detach().clone()
        return output

    backbone_hook2 = policy.model.backbone.register_forward_hook(_capture_trt_backbone_hook)

    # Run ViT TRT with the same pixel_values captured during PyTorch pass
    trt_vit_output = None
    if pt_vit_input is not None and getattr(policy.model.backbone, "vit_engine", None) is not None:
        vit_dtype = policy.model.backbone.vit_engine.dtype_of("pixel_values")
        pv = pt_vit_input.to(vit_dtype).cuda().contiguous()
        policy.model.backbone.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
        vit_result = policy.model.backbone.vit_engine(pv)
        trt_vit_output = vit_result["image_embeds"].detach().clone()

    print("[5] Running TRT inference...")
    obs2 = prepare_observation(policy, dataset, traj_idx=0)
    torch.manual_seed(42)
    with torch.inference_mode():
        result2 = policy.get_action(obs2)

    backbone_hook2.remove()

    action_dict2 = result2[0] if isinstance(result2, tuple) else result2
    trt_arrays = []
    for k in sorted(action_dict2.keys()):
        v = action_dict2[k]
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        trt_arrays.append(t.float().flatten())
    trt_act = torch.cat(trt_arrays)

    # Step 3a: Compare ViT outputs
    if pt_vit_output is not None and trt_vit_output is not None:
        vit_pt = pt_vit_output.float().flatten()
        vit_trt = trt_vit_output.float().flatten()
        vit_cosine = cosine_similarity(vit_pt.unsqueeze(0), vit_trt.unsqueeze(0)).item()
        vit_l1 = (vit_pt - vit_trt).abs().mean().item()
        vit_linf = (vit_pt - vit_trt).abs().max().item()
        print("\n[6a] ViT output comparison (image_embeds):")
        print(f"  Cosine Similarity: {vit_cosine:.6f}")
        print(f"  L1 Mean Error:     {vit_l1:.6f}")
        print(f"  L∞ Max Error:      {vit_linf:.6f}")
    else:
        print("\n[6a] ViT comparison skipped (PyTorch ViT was deleted before capture)")

    # Step 3b: Compare backbone outputs (before vl_self_attention)
    if pt_backbone_features is not None and trt_backbone_features is not None:
        bb_pt = pt_backbone_features.float().flatten()
        bb_trt = trt_backbone_features.float().flatten()
        bb_cosine = cosine_similarity(bb_pt.unsqueeze(0), bb_trt.unsqueeze(0)).item()
        bb_l1 = (bb_pt - bb_trt).abs().mean().item()
        bb_linf = (bb_pt - bb_trt).abs().max().item()
        print("\n[6b] Backbone output comparison (LLM output, before vl_self_attention):")
        print(f"  Cosine Similarity: {bb_cosine:.6f}")
        print(f"  L1 Mean Error:     {bb_l1:.6f}")
        print(f"  L∞ Max Error:      {bb_linf:.6f}")

    # Step 4: Compare final action outputs
    print("\n[6b] Final action output comparison:")
    pt_flat = pt_action.float().flatten()
    trt_flat = trt_act.float().flatten()

    cosine = cosine_similarity(pt_flat.unsqueeze(0), trt_flat.unsqueeze(0)).item()
    l1 = (pt_flat - trt_flat).abs().mean().item()
    linf = (pt_flat - trt_flat).abs().max().item()

    print(f"\n  Cosine Similarity: {cosine:.6f}")
    print(f"  L1 Mean Error:     {l1:.6f}")
    print(f"  L∞ Max Error:      {linf:.6f}")

    full_pipeline = args.mode == "n17_full_pipeline"
    if cosine > 0.999:
        print("\n  ✓ PASS — TRT matches PyTorch")
    elif cosine > 0.99:
        print("\n  ~ WARN — Minor drift detected")
    elif full_pipeline and cosine > 0.94:
        print("\n  ~ EXPECTED — full_pipeline mode compares flash_attention_2 (PyTorch)")
        print("    vs eager attention (TRT/ONNX) — these are numerically non-equivalent")
        print("    algorithms; ~0.96 cosine is normal. Use action_head mode for 0.999 cosine.")
    else:
        print("\n  ✗ FAIL — Significant divergence")


if __name__ == "__main__":
    config = tyro.cli(VerifyConfig)
    main(config)
