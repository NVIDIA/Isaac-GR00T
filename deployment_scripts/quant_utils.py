import modelopt.torch.quantization as mtq
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.data.dataset import LeRobotSingleDataset
from torch.utils.data import Dataset, DataLoader
import torch
from gr00t.experiment.data_config import load_data_config
import time

def no_batch_collate_fn(batch):
    """Collate function that returns the first item without adding batch dimension."""
    return batch[0]


class ViTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for ViT calibration data.
    This provides realistic calibration data for the vision transformer.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
        use_position_ids: bool = False,
    ):
        """
        Initialize the ViT calibration dataset.

        Args:
            dataset_path: Path to the LeRobot dataset
            modality_configs: Modality configuration for the dataset
            embodiment_tag: Embodiment tag for the dataset
            policy: Gr00tPolicy instance for using apply_transforms()
            calib_size: Number of calibration samples to use
            video_backend: Video backend for loading videos
        """
        self.calib_size = calib_size
        self.policy = policy
        self.use_position_ids = use_position_ids
        # Initialize the LeRobot dataset
        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        # Use sequential indices for calibration
        self.dataset_size = len(self.lerobot_dataset)
        print(f"ViT Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        data = self.lerobot_dataset[idx]

        # Process the data to get pixel_values and position_ids for ViT
        processed_data = self._process_vit_data(data)
        return processed_data

    def _process_vit_data(self, data):
        """
        Process LeRobot data to extract pixel_values and position_ids for ViT calibration.
        """
        try:
            # Ensure data is in the correct format for apply_transforms
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            # Apply the same transforms as used in training/inference
            transformed_data = self.policy.apply_transforms(data)

            # Check if we have eagle pixel values
            if "eagle_pixel_values" in transformed_data:
                pixel_values = transformed_data["eagle_pixel_values"]
                batch_size = pixel_values.shape[0]
                # Generate position_ids for the patches
                num_patches = (
                    self.policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
                )
                position_ids = torch.arange(
                    num_patches, dtype=torch.long, device=pixel_values.device
                ).expand((batch_size, -1))
                if self.use_position_ids:
                    return {
                        "pixel_values": pixel_values,
                        "position_ids": position_ids,
                    }
                else:
                    return {
                        "pixel_values": pixel_values,
                    }
            else:
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )
        except Exception as e:
            print(f"Warning: ViT data processing failed: {e}, using dummy data")
            raise RuntimeError(f"apply_transforms() failed: {e}")

def _quantize_model(model, calib_dataloader, quant_cfg):
    """
    The calibration loop for the model can be setup using the modelopt API.
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            data = {k: v.to(next(model.parameters()).device) for k, v in data.items()}
            # breakpoint()
            model(**data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model

def quantize_vit(
    model,
    precision="fp8",
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    use_position_ids=False,
):
    """
    Quantize the ViT model using FP8 quantization.

    Args:
        model: The ViT model to quantize
        precision: Quantization precision (fp8, fp16, etc.)
        calib_size: Number of calibration samples
        batch_size: Batch size for calibration
        dataset_path: Path to LeRobot dataset
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        video_backend: Video backend
        policy: Gr00tPolicy instance
        compare_accuracy: Whether to compare accuracy before/after quantization

    Returns:
        Quantized model
    """
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in [
        "fp8",
        "fp16",
    ], f"Only fp8 and fp16 are supported for ViT. You passed: {precision}."

    # FP8 quantization configuration
    quant_cfg = mtq.FP8_DEFAULT_CFG

    # Disable Conv to avoid accuracy degradation.
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    # Create the dataset and dataloader
    if dataset_path is None or modality_configs is None or policy is None:
        raise ValueError(
            "ViT quantization requires valid dataset_path, modality_configs, and policy."
        )

    print(f"Using LeRobot dataset for ViT calibration: {dataset_path}")
    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()
    device = "cuda"
    policy_copy2 = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    dataset = ViTCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        policy=policy_copy2,
        calib_size=calib_size,
        video_backend=video_backend,
        use_position_ids=use_position_ids,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    # Quantize the model if quantization config is provided
    if quant_cfg is not None:
        quantized_model = _quantize_model(model, data_loader, quant_cfg)
        mtq.print_quant_summary(quantized_model)

        return quantized_model
    else:
        print("No quantization applied to ViT model")
        return model