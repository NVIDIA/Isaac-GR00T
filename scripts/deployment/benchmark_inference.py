#!/usr/bin/env python3
"""
Benchmark script for GR00T inference timing.

Measures component-wise timing for:
- Data Processing: VLAStepData preparation and collation
- Backbone (VLM): Eagle VLM forward pass
- Action Head (DiT): Flow-matching diffusion model
- E2E: Full end-to-end inference

Supports three inference modes:
1. PyTorch Eager: Standard PyTorch execution
2. torch.compile: PyTorch 2.0+ JIT compilation with max-autotune
3. TensorRT: Optimized DiT action head using TensorRT engine

Usage:
    python scripts/deployment/benchmark_inference.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path /path/to/dataset \
        --trt_engine_path ./groot_n1d6_onnx/dit_model_bf16.trt
"""

import argparse
import os
import sys
import time

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn.functional as F


# Ensure scripts/deployment/ is on sys.path for sibling module imports
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from accuracy_thresholds import (  # noqa: E402
    BF16_ACTION_PRED,
    BF16_BACKBONE,
    BF16_COMPONENT_PASS,
    BF16_COMPONENT_WARN,
    BF16_E2E_ACTION,
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rec_to_dtype(x, dtype):
    """Recursively convert all floating point tensors to the given dtype."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def prepare_model_inputs(policy, observation, return_states=False):
    """
    Prepare inputs for the model, mimicking what happens inside _get_action.
    Returns collated_inputs that can be passed to model.get_action()

    Args:
        policy: The Gr00tPolicy instance
        observation: Dict with "video", "state", "language" keys
        return_states: If True, also return the states list (for action denormalization)

    Returns:
        collated_inputs if return_states=False, else (collated_inputs, states)
    """
    unbatched_obs = []
    batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
    for i in range(batch_size):
        unbatched_value = {
            "video": {k: v[i] for k, v in observation["video"].items()},
            "state": {k: v[i] for k, v in observation["state"].items()},
            "language": {k: v[i] for k, v in observation["language"].items()},
        }
        unbatched_obs.append(unbatched_value)

    processed_inputs = []
    states = []
    for obs in unbatched_obs:
        vla_step_data = VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        states.append(vla_step_data.states)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = collated_inputs["inputs"]
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    if return_states:
        return collated_inputs, states
    return collated_inputs


def get_device_name():
    """Get short device name for table."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # Shorten common names
        if "H100" in name:
            return "H100"
        elif "A100" in name:
            return "A100"
        elif "RTX 5090" in name:
            return "RTX 5090"
        elif "RTX 4090" in name:
            return "RTX 4090"
        elif "RTX 3090" in name:
            return "RTX 3090"
        elif "Orin" in name:
            return "Jetson Orin"
        else:
            # Return first meaningful part
            return name.split()[1] if len(name.split()) > 1 else name
    return "CPU"


def compute_e2e_from_components(components):
    """Compute E2E timing as sum of components (more stable than separate measurement)."""
    return components["data_processing"] + components["backbone"] + components["action_head"]


def benchmark_data_processing(policy, observation, num_iterations=20, warmup=10):
    """
    Benchmark data processing separately with proper warmup.
    Data processing is CPU-bound and needs more warmup iterations.

    Args:
        policy: The Gr00tPolicy instance
        observation: Either a single observation dict OR a list of observation dicts (trajectory)
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    If observation is a list (trajectory), cycles through observations during benchmarking.
    """
    import gc

    # Handle both single observation and trajectory (list of observations)
    if isinstance(observation, list):
        observations = observation
    else:
        observations = [observation]

    num_obs = len(observations)

    # Force GC before warmup to reduce variance
    gc.collect()

    # Warmup - helps with CPU caching and JIT for consistent benchmarks
    # For trajectory mode, warmup benefit is reduced since each observation is different
    if warmup > 0:
        for i in range(warmup):
            obs = observations[i % num_obs]
            _ = prepare_model_inputs(policy, obs)
        # Force GC after warmup
        gc.collect()

    # Benchmark
    times = []
    for i in range(num_iterations):
        obs = observations[i % num_obs]
        start = time.perf_counter()
        _ = prepare_model_inputs(policy, obs)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_components(policy, observation, num_iterations=20, warmup=3):
    """
    Benchmark component-wise timing.
    Returns dict with times for: data_processing, backbone, action_head

    Args:
        policy: The Gr00tPolicy instance
        observation: Either a single observation dict OR a list of observation dicts (trajectory)
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    If observation is a list (trajectory), cycles through observations during benchmarking.
    """
    import gc

    # Handle both single observation and trajectory (list of observations)
    if isinstance(observation, list):
        observations = observation
    else:
        observations = [observation]

    num_obs = len(observations)

    # Prepare inputs once for backbone/action_head warmup
    collated_inputs = prepare_model_inputs(policy, observations[0])

    # Warmup backbone + action head
    for i in range(warmup):
        obs = observations[i % num_obs]
        collated_inputs = prepare_model_inputs(policy, obs)
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
            _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
    torch.cuda.synchronize()

    # Force GC before timing
    gc.collect()

    # Benchmark backbone and action head (GPU-bound)
    backbone_times = []
    action_head_times = []

    for i in range(num_iterations):
        obs = observations[i % num_obs]
        collated_inputs = prepare_model_inputs(policy, obs)

        # Backbone timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
            backbone_outputs = policy.model.backbone(backbone_inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        backbone_times.append(end - start)

        # Action head timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.action_head.get_action(backbone_outputs, action_inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        action_head_times.append(end - start)

    # Benchmark data processing separately with proper warmup
    data_processing_times = benchmark_data_processing(
        policy, observation, num_iterations, warmup=10
    )

    return {
        "data_processing": data_processing_times,
        "backbone": np.array(backbone_times) * 1000,
        "action_head": np.array(action_head_times) * 1000,
    }


def print_markdown_table(results, device_name, denoising_steps):
    """Print results as a markdown table using median for robustness."""
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE (copy/paste into README)")
    print("=" * 100)
    print(f"\nGR00T-N1.6-3B Inference Timing ({denoising_steps} denoising steps):\n")

    # Component breakdown table (using median for robustness against outliers)
    print("### Component-wise Breakdown\n")
    print("| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency |")
    print("|--------|------|-----------------|----------|-------------|-----|-----------|")

    for mode, data in results.items():
        dp_median = np.median(data["data_processing"])
        bb_median = np.median(data["backbone"])
        ah_median = np.median(data["action_head"])
        e2e_median = np.median(data["e2e"])
        freq = 1000 / e2e_median
        print(
            f"| {device_name} | {mode} | {dp_median:.0f} ms | {bb_median:.0f} ms | {ah_median:.0f} ms | {e2e_median:.0f} ms | {freq:.1f} Hz |"
        )

    # Speedup table
    if "PyTorch Eager" in results and len(results) > 1:
        print("\n### Speedup vs PyTorch Eager\n")
        print("| Device | Mode | E2E Speedup | Action Head Speedup |")
        print("|--------|------|-------------|---------------------|")

        baseline_e2e = np.median(results["PyTorch Eager"]["e2e"])
        baseline_ah = np.median(results["PyTorch Eager"]["action_head"])

        for mode, data in results.items():
            e2e_median = np.median(data["e2e"])
            ah_median = np.median(data["action_head"])
            e2e_speedup = baseline_e2e / e2e_median
            ah_speedup = baseline_ah / ah_median
            print(f"| {device_name} | {mode} | {e2e_speedup:.2f}x | {ah_speedup:.2f}x |")

    print("\n" + "=" * 100)


# ============================================================
# Accuracy Comparison: TRT vs PyTorch
# ============================================================


def compare_tensors(name, ref, test):
    """Compare two tensors and print accuracy metrics.

    Args:
        name: Label for this comparison
        ref: Reference tensor (PyTorch)
        test: Test tensor (TRT)

    Returns:
        dict with cosine_sim, l1_mean, l1_max, relative_error
    """
    ref_f = ref.float().cpu().flatten()
    test_f = test.float().cpu().flatten()

    # Cosine similarity
    cos_sim = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()

    # L1 distance
    diff = (ref_f - test_f).abs()
    l1_mean = diff.mean().item()
    l1_max = diff.max().item()

    # Relative error (avoid division by zero)
    ref_abs = ref_f.abs()
    nonzero = ref_abs > 1e-8
    if nonzero.any():
        rel_err = (diff[nonzero] / ref_abs[nonzero]).mean().item()
    else:
        rel_err = 0.0

    # Value range
    ref_min, ref_max, ref_mean = ref_f.min().item(), ref_f.max().item(), ref_f.mean().item()
    test_min, test_max, test_mean = test_f.min().item(), test_f.max().item(), test_f.mean().item()

    print(f"\n  [{name}]")
    print(f"    Cosine Similarity:  {cos_sim:.6f}")
    print(f"    L1 Mean / Max:      {l1_mean:.6f} / {l1_max:.6f}")
    print(f"    Relative Error:     {rel_err:.6f}")
    print(f"    Ref  range: [{ref_min:.4f}, {ref_max:.4f}] mean={ref_mean:.4f}")
    print(f"    Test range: [{test_min:.4f}, {test_max:.4f}] mean={test_mean:.4f}")

    return {
        "cosine_sim": cos_sim,
        "l1_mean": l1_mean,
        "l1_max": l1_max,
        "relative_error": rel_err,
    }


def compare_numpy_arrays(name, ref, test):
    """Compare two numpy arrays and print accuracy metrics.

    Args:
        name: Label for this comparison
        ref: Reference numpy array (PyTorch output)
        test: Test numpy array (TRT output)

    Returns:
        dict with cosine_sim, l1_mean, l1_max, relative_error
    """
    return compare_tensors(
        name,
        torch.from_numpy(ref.astype(np.float32)),
        torch.from_numpy(test.astype(np.float32)),
    )


def run_compare_mode(args, policy_pytorch, policy_trt, observation, device):
    """Run accuracy comparison between PyTorch and TRT policies.

    Both policies run the same observation with fixed random noise for the
    denoising process. Compares outputs at backbone, action head, and
    final denormalized action levels.

    Args:
        args: CLI arguments
        policy_pytorch: Gr00tPolicy with PyTorch backend
        policy_trt: Gr00tPolicy with TRT engines loaded
        observation: Single observation dict
        device: CUDA device
    """
    print("\n" + "=" * 100)
    print("ACCURACY COMPARISON: PyTorch vs TensorRT")
    print("=" * 100)

    # Fix random noise for reproducible comparison
    set_seed(args.seed)
    init_actions = torch.randn(
        1,
        policy_pytorch.model.action_head.config.action_horizon,
        policy_pytorch.model.action_head.action_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # Set init_actions on both policies so they use the same starting noise
    policy_pytorch.model.action_head.init_actions = init_actions
    policy_trt.model.action_head.init_actions = init_actions

    # --- Level 1: Backbone output comparison ---
    print("\n--- Level 1: Backbone Features ---")
    collated_inputs = prepare_model_inputs(policy_pytorch, observation)
    with torch.inference_mode():
        backbone_inputs_pt, action_inputs_pt = policy_pytorch.model.prepare_input(collated_inputs)
        backbone_out_pt = policy_pytorch.model.backbone(backbone_inputs_pt)

    collated_inputs_trt = prepare_model_inputs(policy_trt, observation)
    with torch.inference_mode():
        backbone_inputs_trt, action_inputs_trt = policy_trt.model.prepare_input(collated_inputs_trt)
        backbone_out_trt = policy_trt.model.backbone(backbone_inputs_trt)

    backbone_metrics = compare_tensors(
        "backbone_features",
        backbone_out_pt.backbone_features,
        backbone_out_trt.backbone_features,
    )

    # --- Level 2: Action head output comparison ---
    print("\n--- Level 2: Action Prediction (normalized) ---")
    with torch.inference_mode():
        action_out_pt = policy_pytorch.model.action_head.get_action(
            backbone_out_pt, action_inputs_pt
        )
        action_out_trt = policy_trt.model.action_head.get_action(
            backbone_out_trt, action_inputs_trt
        )

    action_metrics = compare_tensors(
        "action_pred", action_out_pt.action_pred, action_out_trt.action_pred
    )

    # --- Level 3: Full end-to-end get_action (denormalized) ---
    print("\n--- Level 3: Denormalized Action (end-to-end) ---")
    set_seed(args.seed)
    policy_pytorch.model.action_head.init_actions = init_actions.clone()
    policy_trt.model.action_head.init_actions = init_actions.clone()

    action_pt, _ = policy_pytorch.get_action(observation)
    action_trt, _ = policy_trt.get_action(observation)

    e2e_metrics = {}
    for key in action_pt:
        if key in action_trt:
            m = compare_numpy_arrays(f"action.{key}", action_pt[key], action_trt[key])
            e2e_metrics[key] = m

    # --- Summary ---
    print("\n" + "=" * 100)
    print("ACCURACY SUMMARY")
    print("=" * 100)

    all_pass = True
    checks = [
        ("Backbone Features", backbone_metrics, BF16_BACKBONE),
        ("Action Pred", action_metrics, BF16_ACTION_PRED),
    ]
    for name, metrics, thresh in checks:
        cos_ok = metrics["cosine_sim"] >= thresh.cosine_min
        l1_ok = thresh.l1_max is None or metrics["l1_mean"] <= thresh.l1_max
        status = "PASS" if (cos_ok and l1_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  {name:25s}: cosine={metrics['cosine_sim']:.6f} "
            f"(>={thresh.cosine_min}? {cos_ok})  "
            f"L1_mean={metrics['l1_mean']:.6f} (<={thresh.l1_max}? {l1_ok})  "
            f"=> {status}"
        )

    for key, metrics in e2e_metrics.items():
        cos_ok = metrics["cosine_sim"] >= BF16_E2E_ACTION.cosine_min
        status = "PASS" if cos_ok else "FAIL"
        if not cos_ok:
            all_pass = False
        print(
            f"  action.{key:19s}: cosine={metrics['cosine_sim']:.6f}  "
            f"L1_mean={metrics['l1_mean']:.6f}  => {status}"
        )

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 100)

    # Clean up init_actions
    if hasattr(policy_pytorch.model.action_head, "init_actions"):
        del policy_pytorch.model.action_head.init_actions
    if hasattr(policy_trt.model.action_head, "init_actions"):
        del policy_trt.model.action_head.init_actions


def run_detailed_compare_mode(args, policy_pytorch, policy_trt, observation, device):
    """Run detailed per-component accuracy tests between PyTorch and TRT.

    Tests each of the 6 TRT engines in isolation (identical input → compare output),
    then runs integration tests to isolate action head drift from backbone drift
    and per-step denoising drift from compounding.

    Part A (Tests 1-6): Per-component isolation
    Part B (Tests 7-8): Integration tests

    Args:
        args: CLI arguments
        policy_pytorch: Gr00tPolicy with PyTorch backend (all modules intact)
        policy_trt: Gr00tPolicy with TRT engines loaded (PyTorch modules deleted)
        observation: Single observation dict
        device: CUDA device
    """
    print("\n" + "=" * 72)
    print("  DETAILED PER-COMPONENT ACCURACY TEST")
    print("=" * 72)

    # Access models
    backbone_pt = policy_pytorch.model.backbone
    backbone_trt = policy_trt.model.backbone
    action_head_pt = policy_pytorch.model.action_head
    action_head_trt = policy_trt.model.action_head
    eagle_model = backbone_pt.model  # Eagle3_VLForConditionalGeneration
    engine_dtype = torch.bfloat16

    # Prepare common inputs (using PyTorch policy's processor)
    collated_inputs = prepare_model_inputs(policy_pytorch, observation)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy_pytorch.model.prepare_input(collated_inputs)

    pixel_values = backbone_inputs["pixel_values"]
    input_ids = backbone_inputs["input_ids"]
    attention_mask = backbone_inputs["attention_mask"]
    if attention_mask.dtype != torch.int64:
        attention_mask = attention_mask.to(torch.int64)

    state = action_inputs["state"]
    if state.dtype != engine_dtype:
        state = state.to(engine_dtype)
    embodiment_id = action_inputs["embodiment_id"]
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    batch_size = state.shape[0]
    results = {}

    # ===================================================================
    # Part A: Component Isolation (same input → compare output)
    # ===================================================================
    print("\n--- Part A: Component Isolation (same input -> compare output) ---")

    # ------ Test 1: ViT (Siglip2) ------
    has_vit_engine = hasattr(backbone_trt, "vit_engine") and backbone_trt.vit_engine is not None
    if has_vit_engine:
        if isinstance(pixel_values, (list, tuple)):
            pv = torch.cat(pixel_values, dim=0)
        else:
            pv = pixel_values
        if pv.dtype != engine_dtype:
            pv = pv.to(engine_dtype)

        print("\n  [1/6] ViT (Siglip2)")
        print(f"    Input: pixel_values {tuple(pv.shape)}")

        with torch.inference_mode():
            original_vit = eagle_model.vision_model.vision_model
            vit_out_pt = original_vit([pv]).last_hidden_state

            backbone_trt.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
            vit_out_trt = backbone_trt.vit_engine(pv)["vit_embeds"]

        results["ViT"] = compare_tensors("1/6 ViT (Siglip2)", vit_out_pt, vit_out_trt)
    else:
        print("\n  [1/6] ViT (Siglip2) — SKIPPED (no vit_engine found)")

    # ------ Test 2: LLM (Qwen3, truncated) ------
    # Build input_embeds: ViT → pixel_shuffle → mlp1 → embed scatter
    # Note: PyTorch LLM may use flash attention while TRT uses eager attention,
    # so small numerical differences from attention implementation are expected.
    print(f"\n  [2/6] LLM (Qwen3, {len(backbone_pt.model.language_model.model.layers)} layers)")

    with torch.inference_mode():
        # Compute ViT embeddings (using full PyTorch pipeline including pixel_shuffle + mlp1)
        vit_embeds = eagle_model.extract_feature(pixel_values)  # [B*N, C]

        # Create input embeddings and scatter vision tokens
        embedding_layer = eagle_model.language_model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids).to(engine_dtype)
        vit_embeds = vit_embeds.to(engine_dtype)

        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = input_ids_flat == eagle_model.config.image_token_index
        input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds_flat.reshape(B, N, C)

        print(f"    Input: input_embeds {tuple(input_embeds.shape)}")

        # PyTorch LLM: run Qwen3Model (already truncated to select_layer layers)
        qwen3_model = eagle_model.language_model.model  # Qwen3Model
        llm_out_pt = qwen3_model(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        ).last_hidden_state

        # TRT LLM
        backbone_trt.llm_engine.set_runtime_tensor_shape("inputs_embeds", input_embeds.shape)
        backbone_trt.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
        llm_out_trt = backbone_trt.llm_engine(input_embeds, attention_mask)["embeddings"]

    results["LLM"] = compare_tensors("2/6 LLM (Qwen3)", llm_out_pt, llm_out_trt)

    # ------ Test 3: State Encoder ------
    print("\n  [3/6] State Encoder")
    print(f"    Input: state {tuple(state.shape)}, embodiment_id {tuple(embodiment_id.shape)}")

    with torch.inference_mode():
        state_out_pt = action_head_pt.state_encoder(state, embodiment_id)

        action_head_trt.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
        action_head_trt.state_encoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        state_out_trt = action_head_trt.state_encoder_engine(state, embodiment_id)["output"]

    results["State Encoder"] = compare_tensors("3/6 State Encoder", state_out_pt, state_out_trt)

    # ------ Test 4: Action Encoder ------
    print("\n  [4/6] Action Encoder")
    set_seed(args.seed)
    actions_noise = torch.randn(
        batch_size,
        action_head_pt.config.action_horizon,
        action_head_pt.action_dim,
        dtype=engine_dtype,
        device=device,
    )
    t_val = 0  # first denoising step
    timesteps = torch.full((batch_size,), fill_value=t_val, device=device, dtype=torch.int64)
    print(
        f"    Input: actions {tuple(actions_noise.shape)}, "
        f"timesteps [t={t_val}], embodiment_id {tuple(embodiment_id.shape)}"
    )

    with torch.inference_mode():
        action_enc_out_pt = action_head_pt.action_encoder(actions_noise, timesteps, embodiment_id)

        action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
            "actions", actions_noise.shape
        )
        action_head_trt.action_encoder_engine.set_runtime_tensor_shape("timesteps", timesteps.shape)
        action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        action_enc_out_trt = action_head_trt.action_encoder_engine(
            actions_noise, timesteps, embodiment_id
        )["output"]

    results["Action Encoder"] = compare_tensors(
        "4/6 Action Encoder", action_enc_out_pt, action_enc_out_trt
    )

    # ------ Test 5: DiT (single forward, t=0) ------
    print("\n  [5/6] DiT (single forward, t=0)")

    # Build sa_embs and vl_embs using PyTorch outputs
    with torch.inference_mode():
        # Get backbone output for vl_embs
        backbone_out_pt = backbone_pt(backbone_inputs)
        vl_embs = action_head_pt.vlln(backbone_out_pt.backbone_features).to(engine_dtype)
        image_mask = backbone_out_pt.image_mask
        bb_attn_mask = backbone_out_pt.backbone_attention_mask

        # Build sa_embs: state encoder output + action encoder output (with pos embed)
        action_features = action_enc_out_pt.clone()
        if action_head_pt.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head_pt.position_embedding(pos_ids).unsqueeze(0).to(engine_dtype)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_out_pt, action_features), dim=1).to(engine_dtype)

    print(f"    Input: sa_embs {tuple(sa_embs.shape)}, vl_embs {tuple(vl_embs.shape)}")

    with torch.inference_mode():
        # PyTorch DiT
        if action_head_pt.config.use_alternate_vl_dit:
            dit_out_pt = action_head_pt.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
                image_mask=image_mask,
                backbone_attention_mask=bb_attn_mask,
            )
        else:
            dit_out_pt = action_head_pt.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
            )

        # TRT DiT
        action_head_trt.dit_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        action_head_trt.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        action_head_trt.dit_engine.set_runtime_tensor_shape("timestep", timesteps.shape)

        dit_kwargs = {}
        if image_mask is not None:
            action_head_trt.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
            dit_kwargs["image_mask"] = image_mask
        if bb_attn_mask is not None:
            action_head_trt.dit_engine.set_runtime_tensor_shape(
                "backbone_attention_mask", bb_attn_mask.shape
            )
            dit_kwargs["backbone_attention_mask"] = bb_attn_mask

        dit_out_trt = action_head_trt.dit_engine(sa_embs, vl_embs, timesteps, **dit_kwargs)[
            "output"
        ]

    results["DiT"] = compare_tensors("5/6 DiT (single step)", dit_out_pt, dit_out_trt)

    # ------ Test 6: Action Decoder ------
    print("\n  [6/6] Action Decoder")
    print(f"    Input: model_output {tuple(dit_out_pt.shape)}")

    with torch.inference_mode():
        decoder_out_pt = action_head_pt.action_decoder(dit_out_pt, embodiment_id)

        action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
            "model_output", dit_out_pt.shape
        )
        action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        decoder_out_trt = action_head_trt.action_decoder_engine(dit_out_pt, embodiment_id)["output"]

    results["Action Decoder"] = compare_tensors(
        "6/6 Action Decoder", decoder_out_pt, decoder_out_trt
    )

    # ===================================================================
    # Part B: Integration Tests
    # ===================================================================
    print("\n--- Part B: Integration Tests ---")

    # ------ Test 7: Action Head with SAME backbone output ------
    # Both PyTorch and TRT action heads receive the SAME backbone output
    # (from PyTorch). Isolates action head TRT drift from backbone drift.
    print("\n  [7] Action Head (same backbone output)")

    set_seed(args.seed)
    init_actions = torch.randn(
        batch_size,
        action_head_pt.config.action_horizon,
        action_head_pt.action_dim,
        dtype=engine_dtype,
        device=device,
    )

    num_steps = action_head_pt.num_inference_timesteps
    dt = 1.0 / num_steps

    with torch.inference_mode():
        # PyTorch denoising loop (manual — for explicit noise control)
        state_features_pt = action_head_pt.state_encoder(state, embodiment_id)
        actions_pt = init_actions.clone()

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_disc = int(t_cont * action_head_pt.num_timestep_buckets)
            ts = torch.full((batch_size,), fill_value=t_disc, device=device, dtype=torch.int64)

            af = action_head_pt.action_encoder(actions_pt, ts, embodiment_id)
            if action_head_pt.config.add_pos_embed:
                pos_ids = torch.arange(af.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head_pt.position_embedding(pos_ids).unsqueeze(0)
                af = af + pos_embs

            sa = torch.cat((state_features_pt, af), dim=1)

            if action_head_pt.config.use_alternate_vl_dit:
                mo = action_head_pt.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl_embs,
                    timestep=ts,
                    image_mask=image_mask,
                    backbone_attention_mask=bb_attn_mask,
                )
            else:
                mo = action_head_pt.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl_embs,
                    timestep=ts,
                )

            pred = action_head_pt.action_decoder(mo, embodiment_id)
            actions_pt = actions_pt + dt * pred[:, -action_head_pt.action_horizon :]

        # TRT denoising loop (using TRT engines, same backbone features + noise)
        action_head_trt.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
        action_head_trt.state_encoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        state_features_trt = action_head_trt.state_encoder_engine(state, embodiment_id)["output"]

        actions_trt = init_actions.clone()

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_disc = int(t_cont * action_head_pt.num_timestep_buckets)
            ts = torch.full((batch_size,), fill_value=t_disc, device=device, dtype=torch.int64)

            action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
                "actions", actions_trt.shape
            )
            action_head_trt.action_encoder_engine.set_runtime_tensor_shape("timesteps", ts.shape)
            action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            af_trt = action_head_trt.action_encoder_engine(
                actions_trt.to(engine_dtype), ts, embodiment_id
            )["output"]

            if action_head_pt.config.add_pos_embed:
                pos_ids = torch.arange(af_trt.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head_pt.position_embedding(pos_ids).unsqueeze(0).to(engine_dtype)
                af_trt = af_trt + pos_embs

            sa_trt = torch.cat((state_features_trt, af_trt), dim=1).to(engine_dtype)

            action_head_trt.dit_engine.set_runtime_tensor_shape("sa_embs", sa_trt.shape)
            action_head_trt.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
            action_head_trt.dit_engine.set_runtime_tensor_shape("timestep", ts.shape)

            dit_kw = {}
            if image_mask is not None:
                action_head_trt.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
                dit_kw["image_mask"] = image_mask
            if bb_attn_mask is not None:
                action_head_trt.dit_engine.set_runtime_tensor_shape(
                    "backbone_attention_mask", bb_attn_mask.shape
                )
                dit_kw["backbone_attention_mask"] = bb_attn_mask

            mo_trt = action_head_trt.dit_engine(sa_trt, vl_embs, ts, **dit_kw)["output"]

            action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
                "model_output", mo_trt.shape
            )
            action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            pred_trt = action_head_trt.action_decoder_engine(mo_trt, embodiment_id)["output"]
            actions_trt = actions_trt + dt * pred_trt[:, -action_head_pt.action_horizon :]

    results["Action Head (same backbone)"] = compare_tensors(
        "7 Action Head (same backbone)", actions_pt, actions_trt
    )

    # ------ Test 8: Per-step Denoising (lockstep) ------
    # At each step, TRT receives the SAME inputs as PyTorch (not its own
    # previous output). Shows isolated per-step drift vs compounding.
    print("\n  [8] Per-step Denoising (lockstep)")

    set_seed(args.seed)
    actions_lock = torch.randn(
        batch_size,
        action_head_pt.config.action_horizon,
        action_head_pt.action_dim,
        dtype=engine_dtype,
        device=device,
    )

    step_results = []
    with torch.inference_mode():
        state_features_lock = action_head_pt.state_encoder(state, embodiment_id)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_disc = int(t_cont * action_head_pt.num_timestep_buckets)
            ts = torch.full((batch_size,), fill_value=t_disc, device=device, dtype=torch.int64)

            # --- PyTorch path ---
            af_pt = action_head_pt.action_encoder(actions_lock, ts, embodiment_id)
            if action_head_pt.config.add_pos_embed:
                pos_ids = torch.arange(af_pt.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head_pt.position_embedding(pos_ids).unsqueeze(0)
                af_pt = af_pt + pos_embs

            sa_pt = torch.cat((state_features_lock, af_pt), dim=1)

            if action_head_pt.config.use_alternate_vl_dit:
                dit_pt = action_head_pt.model(
                    hidden_states=sa_pt,
                    encoder_hidden_states=vl_embs,
                    timestep=ts,
                    image_mask=image_mask,
                    backbone_attention_mask=bb_attn_mask,
                )
            else:
                dit_pt = action_head_pt.model(
                    hidden_states=sa_pt,
                    encoder_hidden_states=vl_embs,
                    timestep=ts,
                )

            dec_pt = action_head_pt.action_decoder(dit_pt, embodiment_id)

            # --- TRT path (same inputs as PyTorch) ---
            action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
                "actions", actions_lock.shape
            )
            action_head_trt.action_encoder_engine.set_runtime_tensor_shape("timesteps", ts.shape)
            action_head_trt.action_encoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            af_trt_lock = action_head_trt.action_encoder_engine(
                actions_lock.to(engine_dtype), ts, embodiment_id
            )["output"]

            if action_head_pt.config.add_pos_embed:
                pos_ids = torch.arange(af_trt_lock.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head_pt.position_embedding(pos_ids).unsqueeze(0).to(engine_dtype)
                af_trt_lock = af_trt_lock + pos_embs

            # Use PyTorch state encoder output for lockstep
            sa_trt_lock = torch.cat((state_features_lock, af_trt_lock), dim=1).to(engine_dtype)

            action_head_trt.dit_engine.set_runtime_tensor_shape("sa_embs", sa_trt_lock.shape)
            action_head_trt.dit_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
            action_head_trt.dit_engine.set_runtime_tensor_shape("timestep", ts.shape)

            dit_kw = {}
            if image_mask is not None:
                action_head_trt.dit_engine.set_runtime_tensor_shape("image_mask", image_mask.shape)
                dit_kw["image_mask"] = image_mask
            if bb_attn_mask is not None:
                action_head_trt.dit_engine.set_runtime_tensor_shape(
                    "backbone_attention_mask", bb_attn_mask.shape
                )
                dit_kw["backbone_attention_mask"] = bb_attn_mask

            dit_trt_lock = action_head_trt.dit_engine(sa_trt_lock, vl_embs, ts, **dit_kw)["output"]

            action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
                "model_output", dit_trt_lock.shape
            )
            action_head_trt.action_decoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            dec_trt_lock = action_head_trt.action_decoder_engine(dit_trt_lock, embodiment_id)[
                "output"
            ]

            # Compare at this step
            dit_cos = F.cosine_similarity(
                dit_pt.float().flatten().unsqueeze(0),
                dit_trt_lock.float().flatten().unsqueeze(0),
            ).item()
            dec_cos = F.cosine_similarity(
                dec_pt.float().flatten().unsqueeze(0),
                dec_trt_lock.float().flatten().unsqueeze(0),
            ).item()
            step_results.append({"dit_cosine": dit_cos, "dec_cosine": dec_cos})
            print(f"    Step {t}: DiT cosine={dit_cos:.6f}, Decoder cosine={dec_cos:.6f}")

            # Advance using PyTorch output (lockstep — TRT follows PyTorch's trajectory)
            pred_velocity = dec_pt[:, -action_head_pt.action_horizon :]
            actions_lock = actions_lock + dt * pred_velocity

    results["Per-step Lockstep"] = step_results

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Part A summary
    worst_name = None
    worst_cosine = 1.0
    part_a_names = [
        "ViT",
        "LLM",
        "State Encoder",
        "Action Encoder",
        "DiT",
        "Action Decoder",
    ]
    for name in part_a_names:
        if name not in results:
            continue
        m = results[name]
        cos = m["cosine_sim"]
        status = (
            "PASS"
            if cos >= BF16_COMPONENT_PASS
            else ("WARN" if cos >= BF16_COMPONENT_WARN else "FAIL")
        )
        print(f"  {name:25s}: cosine={cos:.6f}  L1_mean={m['l1_mean']:.6f}  => {status}")
        if cos < worst_cosine:
            worst_cosine = cos
            worst_name = name

    # Part B summary
    print()
    if "Action Head (same backbone)" in results:
        m = results["Action Head (same backbone)"]
        cos = m["cosine_sim"]
        status = "PASS" if cos >= BF16_E2E_ACTION.cosine_min else "FAIL"
        print(
            f"  {'Action Head (same bb)':25s}: cosine={cos:.6f}  "
            f"L1_mean={m['l1_mean']:.6f}  => {status}"
        )
        if cos < worst_cosine:
            worst_cosine = cos
            worst_name = "Action Head (same backbone)"

    if "Per-step Lockstep" in results:
        steps = results["Per-step Lockstep"]
        avg_dit = sum(s["dit_cosine"] for s in steps) / len(steps)
        avg_dec = sum(s["dec_cosine"] for s in steps) / len(steps)
        print(f"  {'Lockstep avg DiT':25s}: cosine={avg_dit:.6f}")
        print(f"  {'Lockstep avg Decoder':25s}: cosine={avg_dec:.6f}")

    print(f"\n  Component with most drift: {worst_name} (cosine={worst_cosine:.6f})")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GR00T inference timing")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset. Defaults to demo_data/gr1.PickNPlace",
    )
    parser.add_argument("--embodiment_tag", type=str, default="gr1")
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        default=None,
        help="Path to TensorRT engine. If not provided, TensorRT benchmark is skipped.",
    )
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_compile",
        action="store_true",
        help="Skip torch.compile benchmark (can take a while due to JIT compilation)",
    )
    parser.add_argument(
        "--trt_mode",
        type=str,
        default="full_pipeline",
        choices=["dit_only", "full_pipeline"],
        help="TensorRT mode: 'dit_only' or 'full_pipeline' (default: full_pipeline)",
    )
    parser.add_argument(
        "--use_trajectory",
        action="store_true",
        help="Benchmark on full trajectory instead of single data point. "
        "This cycles through all steps in an episode for more realistic benchmarking.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run accuracy comparison between PyTorch and TensorRT. "
        "Requires --trt_engine_path. Compares backbone, action head, and "
        "denormalized action outputs with cosine similarity and L1 metrics.",
    )
    parser.add_argument(
        "--compare-detailed",
        action="store_true",
        dest="compare_detailed",
        help="Run detailed per-component accuracy tests between PyTorch and TensorRT. "
        "Tests each of the 6 TRT engines in isolation, action head with identical "
        "backbone output, and per-step denoising lockstep. Requires --trt_engine_path.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = get_device_name()

    # Default dataset path
    if args.dataset_path is None:
        repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
        args.dataset_path = os.path.join(repo_path, "demo_data/gr1.PickNPlace")

    print("=" * 100)
    print("GR00T INFERENCE BENCHMARK")
    print("=" * 100)
    print(
        f"Device: {device_name} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Use Trajectory: {args.use_trajectory}")
    print()

    # Load dataset and prepare observation
    print("Loading policy...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device=device,
        strict=True,
    )

    denoising_steps = policy.model.action_head.num_inference_timesteps
    action_horizon = policy.model.action_head.action_horizon
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )

    episode_data = dataset[0]

    if args.use_trajectory:
        # Load all steps from the episode for trajectory-based benchmarking
        # episode_data is a pandas DataFrame, so len() gives the number of steps
        trajectory_length = len(episode_data)

        observations = []
        for step_idx in range(trajectory_length):
            try:
                step_data = extract_step_data(
                    episode_data,
                    step_index=step_idx,
                    modality_configs=modality_config,
                    embodiment_tag=EmbodimentTag(args.embodiment_tag),
                    allow_padding=False,
                )
                obs = {
                    "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
                    "state": {k: step_data.states[k][None] for k in step_data.states},
                    "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
                }
                observations.append(obs)
            except Exception:
                # Stop if we can't extract more steps (e.g., due to video frame requirements)
                break

        print(f"Loaded trajectory with {len(observations)} steps")
        observation = observations  # Pass list to benchmark functions
    else:
        step_data = extract_step_data(
            episode_data,
            step_index=0,
            modality_configs=modality_config,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            allow_padding=False,
        )

        observation = {
            "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
            "state": {k: step_data.states[k][None] for k in step_data.states},
            "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
        }

    # ========================================
    # Compare mode: accuracy validation only (skip benchmarking)
    # ========================================
    if args.compare:
        if not args.trt_engine_path or not os.path.exists(args.trt_engine_path):
            print("ERROR: --compare requires --trt_engine_path pointing to built TRT engines.")
            return

        from trt_model_forward import setup_tensorrt_engines

        print("Loading TRT policy for comparison...")
        policy_trt = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
            strict=True,
        )
        setup_tensorrt_engines(policy_trt, args.trt_engine_path, mode=args.trt_mode)

        # For compare mode, use a single observation (not trajectory)
        compare_obs = observation if not isinstance(observation, list) else observation[0]

        # Warmup both policies
        print("Warming up policies...")
        for _ in range(3):
            with torch.inference_mode():
                _ = policy.get_action(compare_obs)
                _ = policy_trt.get_action(compare_obs)

        run_compare_mode(args, policy, policy_trt, compare_obs, device)
        return

    # ========================================
    # Detailed compare mode: per-component accuracy tests (skip benchmarking)
    # ========================================
    if args.compare_detailed:
        if not args.trt_engine_path or not os.path.exists(args.trt_engine_path):
            print(
                "ERROR: --compare-detailed requires --trt_engine_path "
                "pointing to built TRT engines."
            )
            return

        from trt_model_forward import setup_tensorrt_engines

        print("Loading TRT policy for detailed comparison...")
        policy_trt = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
            strict=True,
        )
        setup_tensorrt_engines(policy_trt, args.trt_engine_path, mode=args.trt_mode)

        # Use a single observation (not trajectory)
        compare_obs = observation if not isinstance(observation, list) else observation[0]

        # Warmup both policies
        print("Warming up policies...")
        for _ in range(3):
            with torch.inference_mode():
                _ = policy.get_action(compare_obs)
                _ = policy_trt.get_action(compare_obs)

        run_detailed_compare_mode(args, policy, policy_trt, compare_obs, device)
        return

    results = {}

    # ========================================
    # 0. Benchmark Data Processing (shared across all modes)
    # ========================================
    # Data processing is the same regardless of inference mode (PyTorch/compile/TensorRT)
    # so we benchmark it once with proper warmup to get consistent measurements
    print("\n" + "-" * 50)
    print("Benchmarking Data Processing (shared across all modes)...")
    print("-" * 50)

    shared_data_processing_times = benchmark_data_processing(
        policy, observation, args.num_iterations, warmup=10
    )
    print(
        f"  Data Processing: {np.mean(shared_data_processing_times):.2f} ± {np.std(shared_data_processing_times):.2f} ms"
    )

    # ========================================
    # 1. PyTorch Eager
    # ========================================
    print("\n" + "-" * 50)
    print("Benchmarking PyTorch Eager...")
    print("-" * 50)

    times_components = benchmark_components(policy, observation, args.num_iterations, args.warmup)

    components = {
        "data_processing": shared_data_processing_times,
        "backbone": times_components["backbone"],
        "action_head": times_components["action_head"],
    }
    components["e2e"] = compute_e2e_from_components(components)
    results["PyTorch Eager"] = components

    e2e_median = np.median(components["e2e"])
    print(f"  E2E:             {e2e_median:.0f} ms ({1000 / e2e_median:.1f} Hz)")
    print(f"  Data Processing: {np.median(components['data_processing']):.0f} ms")
    print(f"  Backbone:        {np.median(components['backbone']):.0f} ms")
    print(f"  Action Head:     {np.median(components['action_head']):.0f} ms")

    # ========================================
    # 2. torch.compile
    # ========================================
    if not args.skip_compile:
        print("\n" + "-" * 50)
        print("Benchmarking torch.compile (mode='max-autotune')...")
        print("(This may take a while due to JIT compilation on first run)")
        print("-" * 50)

        policy_compiled = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
            strict=True,
        )
        policy_compiled.model.action_head.model.forward = torch.compile(
            policy_compiled.model.action_head.model.forward, mode="max-autotune"
        )

        # Enable cuDNN benchmark for additional optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Extra warmup for torch.compile JIT
        times_components = benchmark_components(
            policy_compiled, observation, args.num_iterations, warmup=args.warmup + 2
        )

        components = {
            "data_processing": shared_data_processing_times,
            "backbone": times_components["backbone"],
            "action_head": times_components["action_head"],
        }
        components["e2e"] = compute_e2e_from_components(components)
        results["torch.compile"] = components

        e2e_median = np.median(components["e2e"])
        print(f"  E2E:             {e2e_median:.0f} ms ({1000 / e2e_median:.1f} Hz)")
        print(f"  Data Processing: {np.median(components['data_processing']):.0f} ms")
        print(f"  Backbone:        {np.median(components['backbone']):.0f} ms")
        print(f"  Action Head:     {np.median(components['action_head']):.0f} ms")

    # ========================================
    # 3. TensorRT (if available)
    # ========================================
    if args.trt_engine_path and os.path.exists(args.trt_engine_path):
        print("\n" + "-" * 50)
        print(f"Benchmarking TensorRT ({args.trt_mode})...")
        print("-" * 50)

        from trt_model_forward import setup_tensorrt_engines

        policy_trt = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
            strict=True,
        )
        setup_tensorrt_engines(policy_trt, args.trt_engine_path, mode=args.trt_mode)

        # TensorRT needs extra warmup for engine initialization and CUDA context setup
        trt_warmup = max(args.warmup + 5, 10)
        times_components = benchmark_components(
            policy_trt, observation, args.num_iterations, warmup=trt_warmup
        )

        trt_label = f"TensorRT ({args.trt_mode})"
        components = {
            "data_processing": shared_data_processing_times,
            "backbone": times_components["backbone"],
            "action_head": times_components["action_head"],
        }
        components["e2e"] = compute_e2e_from_components(components)
        results[trt_label] = components

        e2e_median = np.median(components["e2e"])
        print(f"  E2E:             {e2e_median:.0f} ms ({1000 / e2e_median:.1f} Hz)")
        print(f"  Data Processing: {np.median(components['data_processing']):.0f} ms")
        print(f"  Backbone:        {np.median(components['backbone']):.0f} ms")
        print(f"  Action Head:     {np.median(components['action_head']):.0f} ms")
    elif args.trt_engine_path:
        print(f"\nTensorRT engine path not found: {args.trt_engine_path}")
        print("To build engines for full pipeline, run:")
        print(
            "  python scripts/deployment/export_onnx_n1d6.py "
            "--model_path nvidia/GR00T-N1.6-3B "
            "--dataset_path demo_data/gr1.PickNPlace "
            "--output_dir ./groot_n1d6_onnx --export_mode full_pipeline"
        )
        print(
            "  python scripts/deployment/build_tensorrt_engine.py "
            "--mode full_pipeline --onnx_dir ./groot_n1d6_onnx "
            "--engine_dir ./groot_n1d6_engines --precision bf16"
        )

    # ========================================
    # Print Summary Tables
    # ========================================
    print_markdown_table(results, device_name, denoising_steps)

    # Detailed summary
    print("\n" + "=" * 100)
    print("DETAILED SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {args.model_path}")
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    for mode, data in results.items():
        print(f"\n{mode}:")
        e2e = data["e2e"]
        print(
            f"  E2E:             median={np.median(e2e):.1f} ms, mean={np.mean(e2e):.1f} ± {np.std(e2e):.1f} ms, "
            f"min={np.min(e2e):.1f}, max={np.max(e2e):.1f} ({1000 / np.median(e2e):.1f} Hz)"
        )
        print(f"  Data Processing: {np.median(data['data_processing']):.2f} ms (median)")
        print(f"  Backbone:        {np.median(data['backbone']):.2f} ms (median)")
        print(f"  Action Head:     {np.median(data['action_head']):.2f} ms (median)")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
