#!/usr/bin/env python3
"""
ViT FP8 Per-Layer Cosine Profiling for GR00T N1.6.

Compares BF16 vs FP8-quantized Siglip2VisionTransformerOpt at each of the 27
transformer layers to identify which layers contribute most to FP8 drift.

This enables **partial quantization** — keeping high-drift layers in BF16
while quantizing the rest, recovering accuracy with minimal speed loss.

Both models use the SDPA attention path (no ONNX export), so the comparison
isolates purely the FP8 quantization drift per layer.

Usage:
    uv run python scripts/deployment/profile_vit_fp8_layers.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path demo_data/gr1.PickNPlace \
        --num_samples 16 \
        --test_partial_quant
"""

import argparse
from collections import defaultdict
import copy
import logging

# Reuse ViT wrappers and calibration helpers from export_onnx_n1d6.py
from export_onnx_n1d6 import (
    Siglip2VisionTransformerOpt,
    ViTCalibrationDataset,
    _load_calibration_observations,
    parse_observation_gr00t,
)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Hook-based Layer Output Capture
# ============================================================


class LayerOutputCapture:
    """Registers forward hooks on encoder layers to capture their outputs."""

    def __init__(self, model):
        """
        Args:
            model: Siglip2VisionTransformerOpt instance
        """
        self.outputs = defaultdict(list)  # layer_idx -> list of tensors
        self._hooks = []

        for idx, layer in enumerate(model.encoder.layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is a tensor (B, N, C) from Siglip2EncoderLayerOpt.forward
            self.outputs[layer_idx].append(output.detach().clone())

        return hook_fn

    def clear(self):
        """Clear captured outputs for next sample."""
        self.outputs.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Per-Layer Metrics Computation
# ============================================================


def compute_layer_metrics(ref_tensor, test_tensor):
    """Compute accuracy metrics between two tensors.

    Args:
        ref_tensor: Reference (BF16) output tensor
        test_tensor: Test (FP8) output tensor

    Returns:
        dict with cosine_sim, l1_mean, relative_error
    """
    ref_f = ref_tensor.float().cpu().flatten()
    test_f = test_tensor.float().cpu().flatten()

    cos_sim = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()

    diff = (ref_f - test_f).abs()
    l1_mean = diff.mean().item()

    ref_norm = ref_f.norm(p=2).item()
    diff_norm = (ref_f - test_f).norm(p=2).item()
    rel_err = diff_norm / ref_norm if ref_norm > 1e-8 else 0.0

    return {"cosine_sim": cos_sim, "l1_mean": l1_mean, "relative_error": rel_err}


def compute_full_model_cosine(ref_output, test_output):
    """Compute cosine similarity between full model outputs."""
    ref_f = ref_output.float().cpu().flatten()
    test_f = test_output.float().cpu().flatten()
    return F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()


# ============================================================
# FP8 Quantization with Optional Layer Exclusion
# ============================================================


def quantize_vit_fp8(model, calib_ds, skip_layers=None):
    """Apply FP8 quantization to a Siglip2VisionTransformerOpt model.

    Args:
        model: Siglip2VisionTransformerOpt instance (will be modified in-place)
        calib_ds: ViTCalibrationDataset
        skip_layers: Optional list of layer indices to exclude from FP8.
                     These layers keep BF16 precision.

    Returns:
        The quantized model (same object, modified in-place)
    """
    import modelopt.torch.quantization as mtq

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # Disable Conv2d quantization (patch embedding) — same as production config
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    # Disable quantization for specific encoder layers
    if skip_layers:
        for layer_idx in skip_layers:
            # Disable all quantizers in this layer's subtree
            layer_prefix = f"encoder.layers.{layer_idx}"
            quant_cfg["quant_cfg"][f"*{layer_prefix}*"] = {"*": {"enable": False}}
        logger.info(f"  Excluding layers {skip_layers} from FP8 quantization")

    def calibrate_loop(model):
        with torch.inference_mode():
            for i in range(min(len(calib_ds), 64)):
                model(calib_ds[i])

    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    return model


# ============================================================
# Profiling Orchestration
# ============================================================


def capture_image_size(policy, dataset):
    """Run one inference to capture the actual image size after Eagle's smart_resize."""
    captured_shape = [None]

    def hook_fn(module, args, kwargs):
        if captured_shape[0] is None:
            pv = args[0] if len(args) > 0 else kwargs.get("pixel_values")
            if pv is not None:
                if isinstance(pv, (list, tuple)) and len(pv) > 0:
                    captured_shape[0] = pv[0].shape
                elif isinstance(pv, torch.Tensor):
                    captured_shape[0] = pv.shape

    eagle_model = policy.model.backbone.model
    hook = eagle_model.vision_model.vision_model.register_forward_pre_hook(
        hook_fn, with_kwargs=True
    )

    modality_configs = policy.get_modality_config()
    traj = dataset[0]
    data_point = extract_step_data(
        traj, 0, modality_configs=modality_configs, embodiment_tag=policy.embodiment_tag
    )
    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in modality_configs["language"].modality_keys:
        observation[key] = data_point.text
    parsed_obs = parse_observation_gr00t(observation, modality_configs)

    with torch.inference_mode():
        _ = policy.get_action(parsed_obs)

    hook.remove()

    if captured_shape[0] is not None:
        image_size = captured_shape[0][-1]
        logger.info(f"  Captured image size: {image_size}x{image_size}")
        return image_size
    else:
        logger.warning("  Could not capture image size, defaulting to 252")
        return 252


def run_profiling(policy, dataset, observations, image_size, num_samples, cosine_threshold):
    """Run per-layer FP8 profiling and print results.

    Args:
        policy: Loaded Gr00tPolicy
        dataset: Dataset loader (not used directly, image_size already captured)
        observations: Calibration observations for FP8 quantization
        image_size: Actual image resolution
        num_samples: Number of samples to average over
        cosine_threshold: Threshold below which a layer is marked as high-drift

    Returns:
        List of high-drift layer indices
    """
    eagle_model = policy.model.backbone.model
    original_vit = eagle_model.vision_model.vision_model

    # 1. Create BF16 baseline model
    logger.info("\n[Step 1] Creating BF16 baseline model...")
    model_bf16 = (
        Siglip2VisionTransformerOpt(original_vit, image_size=image_size)
        .eval()
        .cuda()
        .to(torch.bfloat16)
    )
    num_layers = len(model_bf16.encoder.layers)
    logger.info(f"  {num_layers} encoder layers")

    # 2. Create FP8 quantized model (deep copy weights first)
    logger.info("\n[Step 2] Creating FP8 quantized model...")
    model_fp8 = copy.deepcopy(model_bf16)

    calib_ds = ViTCalibrationDataset(policy, observations)
    logger.info(f"  Calibration samples: {len(calib_ds)}")

    model_fp8 = quantize_vit_fp8(model_fp8, calib_ds)
    logger.info("  FP8 quantization complete")

    # 3. Register hooks on both models
    logger.info("\n[Step 3] Registering forward hooks...")
    capture_bf16 = LayerOutputCapture(model_bf16)
    capture_fp8 = LayerOutputCapture(model_fp8)

    # 4. Run samples through both models and accumulate metrics
    logger.info(f"\n[Step 4] Running {num_samples} samples through both models...")
    actual_samples = min(num_samples, len(calib_ds))

    # Accumulate per-layer metrics
    layer_metrics_accum = defaultdict(
        lambda: {"cosine_sim": [], "l1_mean": [], "relative_error": []}
    )
    full_model_cosines = []

    for i in range(actual_samples):
        pixel_values = calib_ds[i]  # (1, 3, H, W) on CUDA, BF16

        with torch.inference_mode():
            out_bf16 = model_bf16(pixel_values)
            out_fp8 = model_fp8(pixel_values)

        # Full model cosine
        full_model_cosines.append(compute_full_model_cosine(out_bf16, out_fp8))

        # Per-layer metrics
        for layer_idx in range(num_layers):
            bf16_out = capture_bf16.outputs[layer_idx][-1]  # last captured for this sample
            fp8_out = capture_fp8.outputs[layer_idx][-1]
            metrics = compute_layer_metrics(bf16_out, fp8_out)
            for k, v in metrics.items():
                layer_metrics_accum[layer_idx][k].append(v)

        capture_bf16.clear()
        capture_fp8.clear()

        if (i + 1) % 4 == 0:
            logger.info(f"  Processed {i + 1}/{actual_samples} samples")

    # Clean up hooks
    capture_bf16.remove_hooks()
    capture_fp8.remove_hooks()

    # 5. Compute averaged metrics
    logger.info("\n[Step 5] Computing averaged metrics...")
    layer_results = []
    for layer_idx in range(num_layers):
        avg_metrics = {}
        for k in ["cosine_sim", "l1_mean", "relative_error"]:
            values = layer_metrics_accum[layer_idx][k]
            avg_metrics[k] = sum(values) / len(values)
        layer_results.append(avg_metrics)

    avg_full_cosine = sum(full_model_cosines) / len(full_model_cosines)

    # 6. Print report
    print(f"\n{'=' * 72}")
    print(f" ViT FP8 Per-Layer Cosine Profiling ({num_layers} layers, {actual_samples} samples)")
    print(f"{'=' * 72}")
    print(f"\n Full model cosine (BF16 vs FP8, SDPA path): {avg_full_cosine:.6f}")
    print()
    print(f" {'Layer':>5} | {'Cosine Sim':>10} | {'L1 Mean':>10} | {'Rel Error':>10} | {'Status'}")
    print(f" {'-----':>5}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+--------")

    high_drift_layers = []
    for layer_idx, metrics in enumerate(layer_results):
        cos = metrics["cosine_sim"]
        l1 = metrics["l1_mean"]
        rel = metrics["relative_error"]

        if cos < cosine_threshold:
            status = "FAIL"
            high_drift_layers.append(layer_idx)
        elif cos < 0.995:
            status = "WARN"
        else:
            status = "PASS"

        marker = " <-- high drift" if status == "FAIL" else ""
        print(f" {layer_idx:>5} | {cos:>10.6f} | {l1:>10.6f} | {rel:>10.6f} | {status}{marker}")

    print()
    if high_drift_layers:
        print(f" High-drift layers (cosine < {cosine_threshold}): {high_drift_layers}")
    else:
        print(f" No high-drift layers found (all cosine >= {cosine_threshold})")

    print(f"{'=' * 72}")

    # Clean up models to free GPU memory
    del model_bf16, model_fp8
    torch.cuda.empty_cache()

    return high_drift_layers


def run_partial_quant_test(policy, observations, image_size, num_samples, skip_layers):
    """Test partial quantization by excluding high-drift layers from FP8.

    Args:
        policy: Loaded Gr00tPolicy
        observations: Calibration observations
        image_size: Actual image resolution
        num_samples: Number of samples to average over
        skip_layers: Layer indices to exclude from FP8
    """
    eagle_model = policy.model.backbone.model
    original_vit = eagle_model.vision_model.vision_model

    # BF16 baseline
    model_bf16 = (
        Siglip2VisionTransformerOpt(original_vit, image_size=image_size)
        .eval()
        .cuda()
        .to(torch.bfloat16)
    )

    calib_ds = ViTCalibrationDataset(policy, observations)

    # Full FP8 model
    logger.info("  Creating full FP8 model...")
    model_full_fp8 = copy.deepcopy(model_bf16)
    model_full_fp8 = quantize_vit_fp8(model_full_fp8, calib_ds)

    # Partial FP8 model (skip high-drift layers)
    logger.info(f"  Creating partial FP8 model (excluding layers {skip_layers})...")
    model_partial_fp8 = copy.deepcopy(model_bf16)
    model_partial_fp8 = quantize_vit_fp8(model_partial_fp8, calib_ds, skip_layers=skip_layers)

    # Compare both against BF16
    actual_samples = min(num_samples, len(calib_ds))
    full_cosines = []
    partial_cosines = []

    for i in range(actual_samples):
        pixel_values = calib_ds[i]
        with torch.inference_mode():
            out_bf16 = model_bf16(pixel_values)
            out_full = model_full_fp8(pixel_values)
            out_partial = model_partial_fp8(pixel_values)

        full_cosines.append(compute_full_model_cosine(out_bf16, out_full))
        partial_cosines.append(compute_full_model_cosine(out_bf16, out_partial))

    avg_full = sum(full_cosines) / len(full_cosines)
    avg_partial = sum(partial_cosines) / len(partial_cosines)
    improvement = avg_partial - avg_full

    print(f"\n{'=' * 72}")
    print(" Partial Quantization Test")
    print(f"{'=' * 72}")
    print(f" Excluding layers {skip_layers} from FP8...")
    print(f" Full FP8 model cosine:    {avg_full:.6f}")
    print(f" Partial FP8 model cosine: {avg_partial:.6f}  ({improvement:+.4f})")

    if avg_partial >= 0.995:
        print(" Result: PASS (>= 0.995 threshold)")
    elif avg_partial >= 0.990:
        print(" Result: WARN (>= 0.990 but < 0.995)")
    else:
        print(" Result: FAIL (< 0.990, may need to exclude more layers)")

    print(f"{'=' * 72}")

    # Clean up
    del model_bf16, model_full_fp8, model_partial_fp8
    torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================


def main(args):
    logger.info("=" * 72)
    logger.info("ViT FP8 Per-Layer Cosine Profiling")
    logger.info("=" * 72)
    logger.info(f"Model path:   {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment:   {args.embodiment_tag}")
    logger.info(f"Num samples:  {args.num_samples}")
    logger.info(f"Threshold:    {args.cosine_threshold}")
    logger.info(f"Test partial: {args.test_partial_quant}")

    # Step 1: Load policy
    logger.info("\nLoading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )

    # Step 2: Load dataset
    logger.info("Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
    )
    logger.info(f"  Dataset loaded ({len(dataset)} trajectories)")

    # Step 3: Capture actual image size
    logger.info("Capturing actual image size...")
    image_size = capture_image_size(policy, dataset)

    # Step 4: Load calibration observations
    logger.info("Loading calibration observations...")
    calib_path = args.calib_dataset_path or args.dataset_path
    observations = _load_calibration_observations(
        policy, calib_path, args.video_backend, args.calib_size
    )

    # Step 5: Run per-layer profiling
    high_drift_layers = run_profiling(
        policy=policy,
        dataset=dataset,
        observations=observations,
        image_size=image_size,
        num_samples=args.num_samples,
        cosine_threshold=args.cosine_threshold,
    )

    # Step 6: Optional partial quantization test
    if args.test_partial_quant and high_drift_layers:
        logger.info("\nRunning partial quantization test...")
        run_partial_quant_test(
            policy=policy,
            observations=observations,
            image_size=image_size,
            num_samples=args.num_samples,
            skip_layers=high_drift_layers,
        )
    elif args.test_partial_quant and not high_drift_layers:
        logger.info("\nSkipping partial quantization test — no high-drift layers found")

    logger.info("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ViT FP8 Per-Layer Cosine Profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--embodiment_tag",
        type=EmbodimentTag,
        default=EmbodimentTag.GR1,
        help="Embodiment tag",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to average over",
    )
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.990,
        help="Cosine threshold below which a layer is marked high-drift",
    )
    parser.add_argument(
        "--test_partial_quant",
        action="store_true",
        help="Test partial quantization by excluding high-drift layers",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Video decoding backend",
    )
    parser.add_argument(
        "--calib_dataset_path",
        type=str,
        default=None,
        help="Calibration dataset path (defaults to --dataset_path)",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=100,
        help="Max calibration observations to load",
    )

    args = parser.parse_args()
    main(args)
