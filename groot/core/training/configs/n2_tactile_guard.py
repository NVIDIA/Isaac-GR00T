# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Lightweight refusal for runtimes that cannot carry N2 tactile tokens."""


def refuse_tactile_conditioning(config: object, *, context: str) -> None:
    """Check a resolved model config; only None disables a modality, including on stubs.

    Kept outside configs.model and models.n2.tactile: their package initializers
    import the model registry and model stack, respectively.
    """
    if any(getattr(config, field, None) is not None for field in ("tactile_cfg", "deform_cfg")):
        raise NotImplementedError(f"{context} does not support tactile/deform conditioning")
