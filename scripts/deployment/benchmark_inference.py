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
import time

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch


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
        "--use_trajectory",
        action="store_true",
        help="Benchmark on full trajectory instead of single data point. "
        "This cycles through all steps in an episode for more realistic benchmarking.",
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
        print("Benchmarking TensorRT...")
        print("-" * 50)

        from standalone_inference_script import replace_dit_with_tensorrt

        policy_trt = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            device=device,
            strict=True,
        )
        replace_dit_with_tensorrt(policy_trt, args.trt_engine_path)

        # TensorRT needs extra warmup for engine initialization and CUDA context setup
        trt_warmup = max(args.warmup + 5, 10)
        times_components = benchmark_components(
            policy_trt, observation, args.num_iterations, warmup=trt_warmup
        )

        components = {
            "data_processing": shared_data_processing_times,
            "backbone": times_components["backbone"],
            "action_head": times_components["action_head"],
        }
        components["e2e"] = compute_e2e_from_components(components)
        results["TensorRT"] = components

        e2e_median = np.median(components["e2e"])
        print(f"  E2E:             {e2e_median:.0f} ms ({1000 / e2e_median:.1f} Hz)")
        print(f"  Data Processing: {np.median(components['data_processing']):.0f} ms")
        print(f"  Backbone:        {np.median(components['backbone']):.0f} ms")
        print(f"  Action Head:     {np.median(components['action_head']):.0f} ms")
    elif args.trt_engine_path:
        print(f"\nTensorRT engine not found: {args.trt_engine_path}")
        print("To build the engine, run:")
        print(
            "  python scripts/deployment/export_onnx_n1d6.py --model_path nvidia/GR00T-N1.6-3B --output_dir ./groot_n1d6_onnx"
        )
        print(
            "  python scripts/deployment/build_tensorrt_engine.py --onnx ./groot_n1d6_onnx/dit_model.onnx --engine <path>.trt --precision bf16"
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
