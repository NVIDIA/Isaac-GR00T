# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.experiment.data_config import FourierGr1ArmsOnlyDataConfig, So100DataConfig


class So101DataConfig(So100DataConfig):
    """
    Custom data config for SO-101 model that uses ['global1', 'wrist'] video keys
    instead of ['webcam'] video keys.
    """
    # Update video keys to match the SO-101 model's available keys
    video_keys = ["video.global1", "video.wrist"]

class LargeHorizonFourierGr1ArmsOnlyDataConfig(FourierGr1ArmsOnlyDataConfig):
    """
    Custom data config for Fourier-GR1 model that uses ['ego_view'] video keys
    instead of ['webcam'] video keys and a larger action horizon
    """
    action_indices = list(range(40))
    
