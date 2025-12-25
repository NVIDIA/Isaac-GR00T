import warnings

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms


def apply_with_replay(transform, images, replay=None):
    """
    Apply albumentations transforms to multiple images with replay functionality.

    Args:
        transform: Albumentations ReplayCompose or Compose transform
        images: List of PIL Images to transform
        replay: Optional replay data for consistent transforms. If None, creates new replay.

    Returns:
        tuple: (transformed_tensors_list, replay_data)
            - transformed_tensors_list: List of transformed torch tensors (C, H, W) as uint8
            - replay_data: Replay data for consistent transforms across images (None for regular Compose)
    """
    transformed_tensors = []
    current_replay = replay

    # Check if transform supports replay (ReplayCompose)
    has_replay = hasattr(transform, "replay")

    for img in images:
        if has_replay:
            if current_replay is None:
                # First image - create replay data
                augmented_image = transform(image=np.array(img))
                current_replay = augmented_image["replay"]
            else:
                # Subsequent images - use replay for consistent transforms
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented_image = transform.replay(
                        image=np.array(img), saved_augmentations=current_replay
                    )
            img_array = augmented_image["image"]
        else:
            # Regular Compose transform - no replay functionality
            augmented_image = transform(image=np.array(img))
            img_array = augmented_image["image"]

        # Convert to uint8 if needed (albumentations may return float32 in [0,1])
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        # Convert to torch tensor (C, H, W) as uint8
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


class FractionalRandomCrop(A.DualTransform):
    """Crop a random part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)
        # Random position for crop
        max_y = height - crop_height
        max_x = width - crop_width

        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class FractionalCenterCrop(A.DualTransform):
    """Crop the center part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        # Center the crop
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    """
    Build albumentations-based image transformations equivalent to the torchvision version.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters (brightness, contrast, saturation, hue)

    Returns:
        tuple: (train_transform, eval_transform) - raw albumentations transforms
    """

    if crop_fraction is None:
        fraction_to_use = image_crop_size[0] / image_target_size[0]
    else:
        fraction_to_use = crop_fraction

    if shortest_image_edge is None:
        max_size = image_target_size[0]
    else:
        max_size = shortest_image_edge

    # Training transforms (using ReplayCompose for consistent augmentation across views)
    # Use SmallestMaxSize to preserve aspect ratios, with INTER_AREA for antialiasing
    train_transform_list = [
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_transform_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        # Map torchvision ColorJitter parameters to albumentations ColorJitter
        # Note: albumentations uses different parameter names and ranges
        train_transform_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    train_transform = A.ReplayCompose(train_transform_list, p=1.0)

    # Evaluation transforms (deterministic)
    # Use SmallestMaxSize to preserve aspect ratios, with INTER_AREA for antialiasing
    eval_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )

    return train_transform, eval_transform


class LetterBoxTransform:
    """Custom transform to pad non-square images to square by adding black bars.

    Works with any tensor shape where the last 3 dimensions are (C, H, W).
    Leading dimensions (batch, time, views, etc.) are preserved.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Pad image to square dimensions by adding black bars to the smaller dimension.

        Args:
            img: Image tensor of shape (..., C, H, W) where ... can be any leading dimensions
                 Examples: (C, H, W), (B, C, H, W), (B, T*V, C, H, W)

        Returns:
            Padded image tensor of shape (..., C, max(H,W), max(H,W))
        """
        # Get the height and width from the last 2 dimensions
        *leading_dims, c, h, w = img.shape

        if h == w:
            return img

        # Calculate padding needed
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        # Add padding to center the image (divide padding equally on both sides)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # If we have leading dimensions, we need to flatten them, pad, then unflatten
        if leading_dims:
            # Reshape to (batch, C, H, W) where batch includes all leading dimensions
            batch_size = torch.tensor(leading_dims).prod().item()
            img_reshaped = img.reshape(batch_size, c, h, w)

            # Apply padding to each image in the batch
            # torchvision padding format: (left, right, top, bottom)
            padded_img = transforms.functional.pad(
                img_reshaped, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )

            # Reshape back to original leading dimensions
            output_shape = leading_dims + [c, max_dim, max_dim]
            padded_img = padded_img.reshape(output_shape)
        else:
            # Simple case: just (C, H, W)
            padded_img = transforms.functional.pad(
                img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )

        return padded_img


def build_image_transformations(
    image_target_size, image_crop_size, random_rotation_angle, color_jitter_params
):
    """
    Build torchvision-based image transformations.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters (brightness, contrast, saturation, hue)

    Returns:
        tuple: (train_transform, eval_transform) - torchvision transforms
    """
    transform_list = [
        transforms.ToImage(),
        LetterBoxTransform(),
        # transforms.ToDtype(torch.get_default_dtype(), scale=True),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        transform_list.append(
            transforms.RandomRotation(degrees=[-random_rotation_angle, random_rotation_angle])
        )
    if color_jitter_params is not None:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    train_image_transform = transforms.Compose(transform_list)
    eval_image_transform = transforms.Compose(
        [
            # transforms.ToDtype(torch.get_default_dtype(), scale=True),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
    return train_image_transform, eval_image_transform
