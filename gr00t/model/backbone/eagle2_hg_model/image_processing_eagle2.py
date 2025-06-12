# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for LLaVa-Onevision."""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    PaddingMode,
    convert_to_rgb,
    pad,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import IMAGENET_STANDARD_MEAN  # 0.5, 0.5, 0.5
from transformers.image_utils import IMAGENET_STANDARD_STD  # 0.5, 0.5, 0.5
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging

logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def crop(
    img: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    input_data_format: ChannelDimension,
) -> np.ndarray:
    """Crop the given numpy array.

    Args:
        img (np.ndarray): Image to be cropped. Format should be (H, W, C) or (H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.

    Returns:
        np.ndarray: Cropped image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img should be numpy array. Got {}".format(type(img)))

    if img.ndim not in [2, 3]:
        raise ValueError("Image should have 2 or 3 dimensions. Got {}".format(img.ndim))

    if input_data_format == ChannelDimension.LAST:
        img_height = img.shape[0]
        img_width = img.shape[1]
    else:
        img_height = img.shape[1]
        img_width = img.shape[2]

    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError("Crop coordinates out of bounds")

    if top >= bottom or left >= right:
        raise ValueError("Invalid crop coordinates")
    if input_data_format == ChannelDimension.LAST:
        return img[top:bottom, left:right, :]
    else:
        return img[:, top:bottom, left:right]


# Copied from transformers.models.llava_next.image_processing_llava_next.divide_to_patches
def divide_to_patches(image: np.array, patch_size: int, input_data_format) -> List[np.array]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i : i + patch_size, j : j + patch_size]
            else:
                patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


# Copied from transformers.models.llava_next.image_processing_llava_next.expand_to_square
def expand_to_square(image: np.array, background_color, input_data_format) -> np.array:
    """
    Expands an image to a square by adding a background color.
    """

    height, width = get_image_size(image, channel_dim=input_data_format)
    if width == height:
        return image
    elif width > height:
        result = np.ones((width, width, image.shape[2]), dtype=image.dtype) * background_color
        result[(width - height) // 2 : (width - height) // 2 + height, :] = image
        return result
    else:
        result = np.ones((height, height, image.shape[2]), dtype=image.dtype) * background_color
        result[:, (height - width) // 2 : (height - width) // 2 + width] = image
        return result


# Copied from transformers.models.llava_next.image_processing_llava_next._get_patch_output_size
def _get_patch_output_size(image, target_resolution, input_data_format):
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


class Eagle2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LLaVa-Onevision image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        image_grid_pinpoints (`List` *optional*, defaults to `[[672, 336], [336, 672], [672, 672], [336, 1008], [1008, 336]]`):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processinf videos.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values_videos"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        min_dynamic_tiles: int = 1,
        max_dynamic_tiles: int = 12,
        use_thumbnail: bool = True,
        pad_during_tiling: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.do_convert_rgb = do_convert_rgb
        self.min_dynamic_tiles = min_dynamic_tiles
        self.max_dynamic_tiles = max_dynamic_tiles
        self.use_thumbnail = use_thumbnail
        self.pad_during_tiling = pad_during_tiling

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor.pad
    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.

        Args:
            image (`np.ndarray`):
                The image to pad.
            padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
                Padding to apply to the edges of the height, width axes. Can be one of three formats:
                - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
                - `((before, after),)` yields same before and after pad for height and width.
                - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            `np.ndarray`: The padded image.

        """

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values, data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")
        image = (
            to_channel_dimension_format(image, data_format, input_data_format)
            if data_format is not None
            else image
        )
        return image

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._resize_for_patching
    def _resize_for_patching(
        self,
        image: np.array,
        target_resolution: tuple,
        resample,
        input_data_format: ChannelDimension,
    ) -> np.array:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """

        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)
        # Resize the image
        resized_image = resize(
            image, (new_height, new_width), resample=resample, input_data_format=input_data_format
        )

        return resized_image

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._pad_for_patching
    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = self.pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

        return padded_image

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        previous version mainly foucs on ratio.
        We also consider area ratio here.
        """
        best_factor = float("-inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            # ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            # area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
            """
            new area > 60% of original image area is enough.
            """
            factor_based_on_area_n_ratio = min(
                (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
            ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)

            if factor_based_on_area_n_ratio > best_factor:
                best_factor = factor_based_on_area_n_ratio
                best_ratio = ratio

        return best_ratio

    def get_image_patches(
        self,
        image: np.array,
        min_num: int,
        max_num: int,
        size: tuple,
        tile_size: int,
        use_thumbnail: bool,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ):
        image_size = get_image_size(image, channel_dim=input_data_format)
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        # calculate the target width and height
        target_width = tile_size * target_aspect_ratio[0]
        target_height = tile_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        if self.pad_during_tiling:
            resized_image = self._resize_for_patching(
                image,
                (target_height, target_width),
                resample=resample,
                input_data_format=input_data_format,
            )
            padded_image = self._pad_for_patching(
                resized_image, (target_height, target_width), input_data_format=input_data_format
            )
            image_used_to_split = padded_image
        else:
            image_used_to_split = resize(
                image,
                (target_height, target_width),
                resample=resample,
                input_data_format=input_data_format,
            )

        processed_tiles = []
        for i in range(blocks):
            box = (
                (i % (target_width // tile_size)) * tile_size,
                (i // (target_width // tile_size)) * tile_size,
                ((i % (target_width // tile_size)) + 1) * tile_size,
                ((i // (target_width // tile_size)) + 1) * tile_size,
            )
            # split the image
            split_img = crop(image_used_to_split, box[0], box[1], box[2], box[3], input_data_format)
            processed_tiles.append(split_img)
        assert len(processed_tiles) == blocks

        if use_thumbnail and len(processed_tiles) != 1:
            thumbnail_img = resize(
                image,
                (tile_size, tile_size),
                resample=resample,
                input_data_format=input_data_format,
            )
            processed_tiles.append(thumbnail_img)

        # make sure that all patches are in the input data format
        processed_tiles = [
            to_channel_dimension_format(
                tile, channel_dim=data_format, input_channel_dim=input_data_format
            )
            for tile in processed_tiles
        ]
        return processed_tiles

    # Copied from transformers.models.llava_next.image_processing_llava_next.LlavaNextImageProcessor._pad_for_batching
    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            List[`np.ndarray`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Image.Image:
        """
        Args:
            images (`ImageInput`):
                Batch of frames (one video) to preprocess. Expects a batch of frames with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        if do_resize:
            assert False, "do_resize is not supported"
            images = [
                resize(
                    image=image, size=size, resample=resample, input_data_format=input_data_format
                )
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        processed_images = []
        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        for image in images:
            # convert image into a list of patches
            # we intentially use the same data format as the input data format
            size_tuple = (
                (size["height"], size["width"])
                if "height" in size and "width" in size
                else (size["shortest_edge"], size["shortest_edge"])
            )
            image_patches = self.get_image_patches(
                image,
                min_num=self.min_dynamic_tiles,
                max_num=self.max_dynamic_tiles,
                size=size_tuple,
                tile_size=size["height"],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
                use_thumbnail=self.use_thumbnail,
            )

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size_tuple,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pixel_values = np.array(pixel_values)
            processed_images.append(pixel_values)

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )


__all__ = ["Eagle2ImageProcessor"]
