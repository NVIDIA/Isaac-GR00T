#!/usr/bin/env python3
"""
Export GrootN1d6 model components to ONNX for TensorRT optimization.

This script exports the DiT (Diffusion Transformer)
of the GrootN1d6 model to ONNX format for TensorRT conversion.

Usage:
    python export_onnx_n1d6.py \
        --model_path /path/to/checkpoint \
        --dataset_path /path/to/dataset \
        --output_dir ./groot_n1d6_onnx
"""

import argparse
import logging
import os
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import torch.onnx


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DiTInputCapture:
    """
    Helper class to capture DiT forward pass inputs during inference.
    """

    def __init__(self):
        self.captured = False
        self.sa_embs = None
        self.vl_embs = None
        self.timestep = None
        self.image_mask = None
        self.backbone_attention_mask = None

    def hook_fn(self, module, args, kwargs):
        """Pre-forward hook to capture inputs."""
        if not self.captured:
            self.sa_embs = kwargs["hidden_states"].detach().cpu().clone()
            self.vl_embs = kwargs["encoder_hidden_states"].detach().cpu().clone()
            self.timestep = kwargs["timestep"].detach().cpu().clone()
            i_mask = kwargs.get("image_mask")
            if i_mask is not None:
                self.image_mask = i_mask.detach().cpu().clone()
            bb_mask = kwargs.get("backbone_attention_mask")
            if bb_mask is not None:
                self.backbone_attention_mask = bb_mask.detach().cpu().clone()

            self.captured = True
            logger.info(" Captured DiT inputs:")
            logger.info(f"  sa_embs shape: {self.sa_embs.shape}")
            logger.info(f"  vl_embs shape: {self.vl_embs.shape}")
            logger.info(
                f"  timestep shape: {self.timestep.shape if self.timestep is not None else 'None'}"
            )
            logger.info(
                f"  image_mask shape: {self.image_mask.shape if self.image_mask is not None else 'None'}"
            )
            logger.info(
                f"  backbone_attention_mask shape: {self.backbone_attention_mask.shape if self.backbone_attention_mask is not None else 'None'}"
            )


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def prepare_observation(policy, dataset, traj_idx=0):
    """
    Prepare a single observation for inference.

    Args:
        policy: Loaded Gr00tPolicy
        dataset: Dataset loader
        traj_idx: Trajectory index to use

    Returns:
        Observation dictionary ready for policy.get_action()
    """
    logger.info(f"\nPreparing observation from trajectory {traj_idx}...")

    # Load trajectory
    traj = dataset[traj_idx]

    # Get modality configs
    modality_configs = policy.get_modality_config()

    # Extract first step
    data_point = extract_step_data(
        traj,
        0,  # First timestep
        modality_configs=modality_configs,
        embodiment_tag=policy.embodiment_tag,
    )

    # Build observation dict
    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value

    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.array(value)

    for key in modality_configs["language"].modality_keys:
        observation[key] = data_point.text

    # Parse observation to expected format
    parsed_obs = parse_observation_gr00t(observation, modality_configs)

    logger.info(" Observation prepared")
    return parsed_obs


def export_dit_to_onnx(
    policy: Gr00tPolicy, captured_inputs: DiTInputCapture, output_path: str, use_bf16: bool = True
):
    """
    Export the DiT model to ONNX.

    Args:
        policy: Loaded policy with model
        captured_inputs: Captured input tensors from actual inference
        output_path: Path to save ONNX model
        use_bf16: Whether to export in FP16 precision
    """
    logger.info("\n" + "=" * 80)
    logger.info("Exporting DiT to ONNX")
    logger.info("=" * 80)

    # Extract DiT model
    dit_model = policy.model.action_head.model
    dit_model.eval()

    # NOTE: Model is already in BF16 by default (from checkpoint)
    # We only need to set the dtype for dummy inputs
    if use_bf16:
        # Use BF16 to match model's native precision and avoid accuracy loss
        dtype = torch.bfloat16
        logger.info("Using BF16 precision (native model precision)")
    else:
        dtype = torch.float32

    dit_model = dit_model.cuda()

    sa_embs = torch.randn(captured_inputs.sa_embs.shape, dtype=dtype, device="cuda")
    vl_embs = torch.randn(captured_inputs.vl_embs.shape, dtype=dtype, device="cuda")
    timestep = torch.ones(captured_inputs.timestep.shape, dtype=torch.int64, device="cuda")

    export_inputs = [sa_embs, vl_embs, timestep]
    input_names = ["sa_embs", "vl_embs", "timestep"]
    # Establish the dynamix_axes:
    # For example:
    #   vl_embs dimensions are [B, vl_seq_len, vl_embed]
    #   The axes of B and vl_seq_len can vary i.e. batch sz and num tokens in input
    #   vl_embed is fixed at 2048 and hence is not in dynalic)axes
    dynamic_axes = {
        "sa_embs": {0: "batch_size", 1: "sa_seq_len"},
        "vl_embs": {0: "batch_size", 1: "vl_seq_len"},
        "timestep": {0: "batch_size"},
        "output": {0: "batch_size", 1: "sa_seq_len"},
    }

    image_mask = None
    if captured_inputs.image_mask is not None:
        image_mask = torch.ones(captured_inputs.image_mask.shape, dtype=torch.bool, device="cuda")
        export_inputs.append(image_mask)
        input_names.append("image_mask")
        dynamic_axes["image_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    backbone_attention_mask = None
    if captured_inputs.backbone_attention_mask is not None:
        backbone_attention_mask = torch.ones(
            captured_inputs.backbone_attention_mask.shape, dtype=torch.bool, device="cuda"
        )
        export_inputs.append(backbone_attention_mask)
        input_names.append("backbone_attention_mask")
        dynamic_axes["backbone_attention_mask"] = {0: "batch_size", 1: "vl_seq_len"}

    logger.info("Export input shapes:")
    logger.info(f"  sa_embs: {sa_embs.shape} ({sa_embs.dtype})")
    logger.info(f"  vl_embs: {vl_embs.shape} ({vl_embs.dtype})")
    logger.info(f"  timestep: {timestep.shape} ({timestep.dtype})")
    if image_mask is not None:
        logger.info(f"  image_mask: {image_mask.shape} ({image_mask.dtype})")
    if backbone_attention_mask is not None:
        logger.info(
            f"  backbone_attention_mask: {backbone_attention_mask.shape} ({backbone_attention_mask.dtype})"
        )

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    logger.info(f"Exporting to {output_path}...")

    # Create a wrapper to handle keyword arguments
    # torch.onnx.export uses positional args: `dit.forward(arg1, arg2...)`
    # DiT module uses keyword args: `dit.forward(hidden_states=....)`
    # The DiTWrapper handles this translation
    class DiTWrapper(torch.nn.Module):
        def __init__(self, dit_model, has_backbone_mask):
            super().__init__()
            self.dit_model = dit_model
            self.has_backbone_mask = has_backbone_mask

        def forward(self, sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask=None):
            # Call DiT with keyword arguments
            if self.has_backbone_mask:
                return self.dit_model(
                    sa_embs,
                    vl_embs,
                    timestep,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
            else:
                return self.dit_model(sa_embs, vl_embs, timestep, image_mask=image_mask)

    has_backbone_mask = backbone_attention_mask is not None
    wrapped_model = DiTWrapper(dit_model, has_backbone_mask)
    wrapped_model.eval()

    with torch.inference_mode():
        torch.onnx.export(
            wrapped_model,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    logger.info(" DiT exported successfully!")

    # Verify the export
    logger.info("\nVerifying ONNX export...")
    import onnx

    # Get file size first
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Model size on disk: {file_size_mb:.2f} MB")

    # Check if external data file exists (for large models)
    external_data_path = output_path.replace(".onnx", ".onnx.data")
    if os.path.exists(external_data_path):
        external_size_mb = os.path.getsize(external_data_path) / (1024 * 1024)
        logger.info(f"External data size: {external_size_mb:.2f} MB")
        logger.info(f"Total model size: {file_size_mb + external_size_mb:.2f} MB")

    # For large models, validate using file path instead of loading into memory
    try:
        # Use model path checking for large models
        onnx.checker.check_model(output_path)
        logger.info(" ONNX model is valid!")
    except ValueError as e:
        if "too large" in str(e):
            # Model is large, just verify it can be loaded
            logger.info("Model is very large, skipping full validation...")
            try:
                onnx.shape_inference.infer_shapes_path(output_path, output_path + ".tmp")
                os.remove(output_path + ".tmp")
                logger.info(" ONNX model structure verified!")
            except Exception as e2:
                logger.warning(f"Could not fully validate (this is OK): {e2}")
                logger.info(" ONNX model exported (validation skipped for large model)")
        else:
            raise


def main(args):
    logger.info("=" * 80)
    logger.info("GrootN1d6 ONNX Export Script")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Embodiment: {args.embodiment_tag}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Step 1: Load the policy
    logger.info("\n[Step 1] Loading policy...")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    logger.info(" Policy loaded")

    # Step 2: Load dataset
    logger.info("\n[Step 2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )
    logger.info(f" Dataset loaded ({len(dataset)} trajectories)")

    # Step 3: Capture DiT inputs
    logger.info("\n[Step 3] Capturing DiT inputs from actual inference...")

    # Set up hook to capture inputs
    capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(
        capture.hook_fn, with_kwargs=True
    )

    # Run one inference to capture shapes
    observation = prepare_observation(policy, dataset, traj_idx=0)
    logger.info("Running inference to capture shapes...")
    with torch.inference_mode():
        _ = policy.get_action(observation)

    # Remove hook
    hook.remove()

    if not capture.captured:
        logger.error(" Failed to capture DiT inputs!")
        return

    # Step 4: Export DiT
    logger.info("\n[Step 4] Exporting DiT to ONNX...")
    dit_output_path = os.path.join(args.output_dir, "dit_model.onnx")
    export_dit_to_onnx(
        policy=policy, captured_inputs=capture, output_path=dit_output_path, use_bf16=True
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nExported files in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GrootN1d6 model to ONNX")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset (used to capture input shapes)",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=EmbodimentTag,
        default=EmbodimentTag.GR1,
        help="Embodiment tag (default: GR1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./groot_n1d6_onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Options: ['decord', 'torchvision_av', 'torchcodec']",
    )

    args = parser.parse_args()
    main(args)
