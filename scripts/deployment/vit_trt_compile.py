#!/usr/bin/env python3
"""Compile Siglip2 ViT using torch_tensorrt with SDPA preserved.

This script compiles the ViT wrapper using torch_tensorrt's dynamo backend,
which can handle F.scaled_dot_product_attention natively — avoiding the
flash→manual attention numerical drift seen in the ONNX export path.

Prerequisites:
    - TensorRT >= 10.8 (earlier versions have SDPA correctness bugs)
    - torch_tensorrt: uv pip install torch_tensorrt

Usage:
    uv run python scripts/deployment/vit_trt_compile.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --dataset_path demo_data/gr1.PickNPlace \
        --output_dir ./groot_n1d6_engines

The compiled ViT is saved as `vit_trt_compiled.ep` (ExportedProgram format).
It can be loaded alongside ONNX-based TRT engines in the full pipeline.
"""

import argparse
import logging
import os

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check TRT and torch_tensorrt versions."""
    import tensorrt as trt

    trt_version = trt.__version__
    trt_major_minor = tuple(int(x) for x in trt_version.split(".")[:2])
    logger.info(f"TensorRT version: {trt_version}")

    if trt_major_minor < (10, 8):
        logger.error(
            f"TensorRT {trt_version} has known SDPA correctness bugs. "
            f"Requires >= 10.8. See https://github.com/NVIDIA/TensorRT/issues/4333"
        )
        return False

    try:
        import torch_tensorrt

        logger.info(f"torch_tensorrt version: {torch_tensorrt.__version__}")
    except ImportError:
        logger.error("torch_tensorrt not installed. Run: uv pip install torch_tensorrt")
        return False

    return True


def compile_vit(policy, output_dir, image_size=None):
    """Compile the ViT using torch_tensorrt with SDPA preserved.

    Args:
        policy: Loaded Gr00tPolicy
        output_dir: Directory to save compiled model
        image_size: Override image size (auto-detected if None)

    Returns:
        Path to the compiled model file, or None on failure
    """
    import torch_tensorrt

    from export_onnx_n1d6 import Siglip2VisionTransformerOpt

    eagle_model = policy.model.backbone.model
    original_vit = eagle_model.vision_model.vision_model

    # Auto-detect image size from a sample inference if not provided
    if image_size is None:
        from export_onnx_n1d6 import prepare_observation

        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        dataset = LeRobotEpisodeLoader(
            dataset_path=args.dataset_path,
            modality_configs=policy.get_modality_config(),
            video_backend="torchcodec",
        )

        captured_shape = [None]

        def capture_hook(module, args, kwargs):
            pv = args[0] if len(args) > 0 else kwargs.get("pixel_values")
            if pv is not None and captured_shape[0] is None:
                if isinstance(pv, (list, tuple)):
                    captured_shape[0] = pv[0].shape
                else:
                    captured_shape[0] = pv.shape

        hook = original_vit.register_forward_pre_hook(capture_hook, with_kwargs=True)
        obs = prepare_observation(policy, dataset, traj_idx=0)
        with torch.inference_mode():
            policy.get_action(obs)
        hook.remove()

        if captured_shape[0] is not None:
            image_size = captured_shape[0][-1]
        else:
            image_size = 252  # default for GR1
        logger.info(f"Detected image size: {image_size}")

    # Create SDPA-only wrapper (no manual matmul fallback needed for torch_tensorrt)
    logger.info("Creating Siglip2VisionTransformerOpt wrapper...")
    opt_vit = Siglip2VisionTransformerOpt(original_vit, image_size=image_size)
    opt_vit = opt_vit.eval().cuda().to(torch.bfloat16)

    # Verify wrapper accuracy vs original
    dummy_input = torch.randn(1, 3, image_size, image_size, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        orig_out = original_vit([dummy_input]).last_hidden_state
        wrapper_out = opt_vit(dummy_input)

    cos_sim = F.cosine_similarity(
        orig_out.float().flatten().unsqueeze(0),
        wrapper_out.float().flatten().unsqueeze(0),
    ).item()
    logger.info(f"Wrapper vs Original cosine: {cos_sim:.6f}")

    # Compile with torch_tensorrt
    logger.info("Compiling ViT with torch_tensorrt...")
    logger.info("  This uses SDPA natively (no matmul+softmax decomposition)")

    try:
        # Use dynamo backend for best SDPA support.
        # Use fixed batch_size=1 to avoid ConstraintViolationError from dynamo
        # inferring batch dim as constant during tracing. Inference is always batch=1.
        compiled_vit = torch_tensorrt.compile(
            opt_vit,
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    shape=[1, 3, image_size, image_size],
                    dtype=torch.bfloat16,
                )
            ],
            enabled_precisions={torch.bfloat16},
            truncate_double=True,
        )

        # Verify compiled output
        with torch.inference_mode():
            compiled_out = compiled_vit(dummy_input)

        cos_sim_compiled = F.cosine_similarity(
            orig_out.float().flatten().unsqueeze(0),
            compiled_out.float().flatten().unsqueeze(0),
        ).item()
        logger.info(f"Compiled vs Original cosine: {cos_sim_compiled:.6f}")
        cos_sim_wrapper = F.cosine_similarity(
            wrapper_out.float().flatten().unsqueeze(0),
            compiled_out.float().flatten().unsqueeze(0),
        ).item()
        logger.info(f"Compiled vs Wrapper cosine: {cos_sim_wrapper:.6f}")

        if cos_sim_compiled < 0.999:
            logger.warning(
                f"Compiled ViT cosine {cos_sim_compiled:.6f} is below 0.999 threshold!"
            )

        # Save compiled model
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "vit_trt_compiled.ep")
        torch_tensorrt.save(compiled_vit, output_path, output_format="exported_program")
        logger.info(f"Saved compiled ViT to {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        return output_path

    except Exception as e:
        logger.error(f"torch_tensorrt compilation failed: {e}")
        logger.info("Falling back to ONNX export path for ViT.")
        return None


def main():
    global args
    parser = argparse.ArgumentParser(description="Compile ViT with torch_tensorrt")
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/GR00T-N1.6-3B",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="demo_data/gr1.PickNPlace",
        help="Dataset for shape capture",
    )
    parser.add_argument("--output_dir", type=str, default="./groot_n1d6_engines", help="Output dir")
    parser.add_argument("--image_size", type=int, default=None, help="Override image size")
    parser.add_argument(
        "--check_only", action="store_true", help="Only check prerequisites, don't compile"
    )

    args = parser.parse_args()

    if not check_prerequisites():
        return

    if args.check_only:
        logger.info("Prerequisites check passed!")
        return

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logger.info("Loading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.GR1,
        model_path=args.model_path,
        device="cuda",
    )

    compile_vit(policy, args.output_dir, image_size=args.image_size)


if __name__ == "__main__":
    main()
