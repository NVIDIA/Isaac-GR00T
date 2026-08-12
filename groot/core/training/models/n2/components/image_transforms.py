# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import random
import warnings

import albumentations as A
import av
import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Codec-augmentation transforms (JPEG / h264 / mp4v round-trip)
# ---------------------------------------------------------------------------
# Each transform is an albumentations ImageOnlyTransform so it composes with
# ReplayCompose. Single-frame application — matches the deployment-time
# in-policy codec block which runs frame-by-frame.
#
# No try/except: errors propagate. Inputs are expected to be uint8 RGB HWC.
# h264 and mp4v require even H and W (yuv420p constraint); JPEG has no
# dim restriction. Placing codec aug AFTER the A.Resize step satisfies the
# even-dim requirement (target_size is even by convention).
#
# Color convention: ALL three transforms preserve RGB channel order on input
# RGB uint8 HWC arrays. JPEG via cv2.imencode/imdecode is value-preserving;
# PyAV h264/mp4v use format='rgb24' end-to-end and need no cvtColor.


def _assert_uint8_hwc3(img: np.ndarray) -> None:
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, (
        f"codec aug requires HxWx3 uint8 RGB; got shape={img.shape} dtype={img.dtype}"
    )


def _assert_even(img: np.ndarray) -> None:
    _assert_uint8_hwc3(img)
    h, w = img.shape[:2]
    assert h % 2 == 0 and w % 2 == 0, (
        f"codec aug requires even H,W (yuv420p); got ({h},{w})"
    )


def _pyav_video_roundtrip(img: np.ndarray, codec: str, rate: int, options: dict) -> np.ndarray:
    """Single-frame PyAV encode → decode. format='rgb24' end-to-end.

    Encodes the frame twice. Some PyAV codec backends drop a single-frame
    stream during flush; encoding two identical frames is a robust workaround
    that doesn't change the artifact pattern of the returned frame[0].
    """
    _assert_even(img)
    h, w, _ = img.shape
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as cnt:
        st = cnt.add_stream(codec, rate=rate)
        st.width = w
        st.height = h
        st.pix_fmt = "yuv420p"
        if options:
            st.options = {k: str(v) for k, v in options.items()}
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in st.encode(frame):
            cnt.mux(pkt)
        for pkt in st.encode(frame):
            cnt.mux(pkt)
        for pkt in st.encode():
            cnt.mux(pkt)
    buf.seek(0)
    decoded = []
    with av.open(buf) as cnt:
        for f in cnt.decode(video=0):
            decoded.append(f.to_ndarray(format="rgb24"))
    assert len(decoded) >= 1, f"{codec} decode returned 0 frames"
    return decoded[0]


class _JPEGCodec(A.ImageOnlyTransform):
    """cv2.imencode/imdecode JPEG roundtrip. Mirrors the in-policy JPEG cycle."""

    def __init__(self, quality_range: tuple[int, int] = (60, 95), p: float = 1.0):
        super().__init__(p=p)
        self.quality_range = tuple(quality_range)

    def get_params(self) -> dict:
        return {"quality": random.randint(*self.quality_range)}

    def apply(self, img: np.ndarray, quality: int = 95, **kwargs) -> np.ndarray:
        _assert_uint8_hwc3(img)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        assert ok, "cv2.imencode jpg failed"
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert out is not None, "cv2.imdecode returned None"
        return out

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("quality_range",)


class _H264Codec(A.ImageOnlyTransform):
    """PyAV h264 single-frame encode-decode. Mirrors the in-policy h264 block."""

    def __init__(self, crf_range: tuple[int, int] = (18, 28), rate: int = 20, p: float = 1.0):
        super().__init__(p=p)
        self.crf_range = tuple(crf_range)
        self.rate = int(rate)

    def get_params(self) -> dict:
        return {"crf": random.randint(*self.crf_range)}

    def apply(self, img: np.ndarray, crf: int = 22, **kwargs) -> np.ndarray:
        return _pyav_video_roundtrip(img, codec="h264", rate=self.rate, options={"crf": int(crf)})

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crf_range", "rate")


class _Mp4vCodec(A.ImageOnlyTransform):
    """PyAV mpeg4 (MPEG-4 Part 2) single-frame encode-decode. Same codec family
    as cv2.VideoWriter's 'mp4v' fourcc. Uses libavcodec via PyAV so this works
    on every machine that has PyAV (cv2.VideoWriter's mp4v isn't always built
    into OpenCV)."""

    def __init__(self, rate: int = 30, p: float = 1.0):
        super().__init__(p=p)
        self.rate = int(rate)

    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        return _pyav_video_roundtrip(img, codec="mpeg4", rate=self.rate, options={})

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("rate",)


class RandomCodecMixture(A.ImageOnlyTransform):
    """Per-sample codec choice replayed across all frames of a sequence.

    Inside A.ReplayCompose, get_params is invoked once for the first frame; the
    returned dict (chosen codec + per-codec params) is stored in the replay
    record and reused for subsequent frames of the same sample. Different
    samples in a batch can pick different codecs.

    mixture_weights: dict of codec name -> weight (auto-normalized).
        Supported codecs: 'identity', 'jpeg', 'h264', 'mp4v'.
    """

    SUPPORTED = ("identity", "jpeg", "h264", "mp4v")

    def __init__(
        self,
        mixture_weights: dict,
        jpeg_quality_range: tuple[int, int] = (60, 95),
        h264_crf_range: tuple[int, int] = (18, 28),
        h264_rate: int = 20,
        mp4v_rate: int = 30,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        for k in mixture_weights:
            assert k in self.SUPPORTED, f"unsupported codec {k!r}; allowed: {self.SUPPORTED}"
        keys = list(mixture_weights.keys())
        weights = np.array(list(mixture_weights.values()), dtype=np.float64)
        assert (weights >= 0).all() and weights.sum() > 0, (
            "codec mixture weights must be non-negative and sum > 0"
        )
        self.mixture_weights = dict(mixture_weights)
        self._codecs = keys
        self._weights = weights / weights.sum()
        self.jpeg_quality_range = tuple(jpeg_quality_range)
        self.h264_crf_range = tuple(h264_crf_range)
        self.h264_rate = int(h264_rate)
        self.mp4v_rate = int(mp4v_rate)
        self._jpeg = _JPEGCodec(quality_range=self.jpeg_quality_range)
        self._h264 = _H264Codec(crf_range=self.h264_crf_range, rate=self.h264_rate)
        self._mp4v = _Mp4vCodec(rate=self.mp4v_rate)

    def get_params(self) -> dict:
        codec = str(np.random.choice(self._codecs, p=self._weights))
        params: dict = {"codec": codec}
        if codec == "jpeg":
            params["quality"] = random.randint(*self.jpeg_quality_range)
        elif codec == "h264":
            params["crf"] = random.randint(*self.h264_crf_range)
        return params

    def apply(
        self,
        img: np.ndarray,
        codec: str = "identity",
        quality: int = 95,
        crf: int = 22,
        **kwargs,
    ) -> np.ndarray:
        if codec == "identity":
            return img
        if codec == "jpeg":
            return self._jpeg.apply(img, quality=quality)
        if codec == "h264":
            return self._h264.apply(img, crf=crf)
        if codec == "mp4v":
            return self._mp4v.apply(img)
        raise ValueError(f"unhandled codec={codec!r}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("mixture_weights", "jpeg_quality_range", "h264_crf_range", "h264_rate", "mp4v_rate")


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
    """Crop a random part of the input based on a fraction of its dimensions.

    The crop size is computed dynamically from the input image, so no
    ``original_size`` parameter is needed.

    Args:
        crop_fraction: Fraction of each spatial dimension to keep (0.0, 1.0].
        p: probability of applying the transform.
    """

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class FractionalCenterCrop(A.DualTransform):
    """Center-crop the input based on a fraction of its dimensions.

    Args:
        crop_fraction: Fraction of each spatial dimension to keep (0.0, 1.0].
        p: probability of applying the transform.
    """

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


_INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
    "linear_exact": cv2.INTER_LINEAR_EXACT,
    "nearest_exact": cv2.INTER_NEAREST_EXACT,
    "max": cv2.INTER_MAX,
}


def build_image_transformations_albumentations(
    target_size: list[int],
    crop_fraction: float,
    interpolation: str,
    color_jitter_params: dict[str, float] | None,
    codec_aug_params: dict | None = None,
):
    """Build albumentations image transforms using fractional crops.

    The crop is computed as a fraction of the input image's actual dimensions,
    then resized to ``target_size``. This removes the need for an explicit
    ``original_size`` parameter.

    Args:
        target_size: Output size [height, width].
        crop_fraction: Fraction of each spatial dimension to keep before resizing.
        interpolation: Interpolation method name (key into _INTERPOLATION_MAP).
        color_jitter_params: Optional color jitter parameters.
        codec_aug_params: Optional codec-stack augmentation. Each sample picks
            one codec from {identity, jpeg, h264, mp4v} per the configured
            mixture weights, then applies it consistently across all frames of
            the sequence (via ReplayCompose).
            Mixture weights may be given either as flat keys (CLI-friendly:
            ``mixture_identity``, ``mixture_jpeg``, ``mixture_h264``,
            ``mixture_mp4v``) or as a nested ``mixture`` dict; flat keys win
            when both are present. Other keys: ``jpeg_quality_lower/upper``,
            ``h264_crf_lower/upper``, ``h264_rate``, ``mp4v_rate``, ``p``.

    Returns:
        (train_transform, eval_transform)
    """
    interp = _INTERPOLATION_MAP[interpolation]

    train_transform_list = [
        FractionalRandomCrop(crop_fraction=crop_fraction),
        A.Resize(height=target_size[0], width=target_size[1], interpolation=interp, p=1.0),
    ]

    if color_jitter_params is not None:
        train_transform_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    if codec_aug_params is not None:
        # Codec-stack augmentation must run after A.Resize so that the input to
        # the yuv420p-restricted codecs (h264, mp4v) has even H and W. Mixture
        # weights accepted in two forms for CLI flexibility:
        #   (a) flat per-codec keys: mixture_identity=0.7, mixture_jpeg=0.15, ...
        #       (tyro-friendly: ``dict[str, float]`` CLI works directly).
        #   (b) nested ``mixture`` sub-dict (loaded from YAML, not CLI).
        # Flat keys win when both are present.
        mixture = {}
        for _c in RandomCodecMixture.SUPPORTED:
            k = f"mixture_{_c}"
            if k in codec_aug_params:
                mixture[_c] = float(codec_aug_params[k])
        if not mixture:
            m = codec_aug_params.get("mixture")
            if isinstance(m, dict):
                mixture = m
            else:
                mixture = {"identity": 1.0}
        train_transform_list.append(
            RandomCodecMixture(
                mixture_weights=mixture,
                jpeg_quality_range=(
                    int(codec_aug_params.get("jpeg_quality_lower", 60)),
                    int(codec_aug_params.get("jpeg_quality_upper", 95)),
                ),
                h264_crf_range=(
                    int(codec_aug_params.get("h264_crf_lower", 18)),
                    int(codec_aug_params.get("h264_crf_upper", 28)),
                ),
                h264_rate=int(codec_aug_params.get("h264_rate", 20)),
                mp4v_rate=int(codec_aug_params.get("mp4v_rate", 30)),
                p=float(codec_aug_params.get("p", 1.0)),
            )
        )

    train_transform = A.ReplayCompose(train_transform_list, p=1.0)

    eval_transform = A.Compose([
        FractionalCenterCrop(crop_fraction=crop_fraction),
        A.Resize(height=target_size[0], width=target_size[1], interpolation=interp, p=1.0),
    ])

    return train_transform, eval_transform
