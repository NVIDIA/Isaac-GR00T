# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass

from transformers import PretrainedConfig

from groot.core.training.configs.model import register_model_config


@dataclass
class IdentityConfig(PretrainedConfig):
    """No-op model config used by the GPU data-loading dry-run.

    Selects ``IdentityPipeline`` from the model registry, which builds an
    ``IdentityModel`` + ``IdentityProcessor`` + ``IdentityCollator`` (all
    pass-through). Use this to isolate data-loading + preprocessing cost from
    model compute.
    """

    model_type: str = "Identity"


register_model_config("Identity", IdentityConfig)
