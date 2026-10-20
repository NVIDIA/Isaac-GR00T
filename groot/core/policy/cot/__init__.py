# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Robot-agnostic chain-of-thought (System-2) policy wrappers.

A remote VLM planner translates a high-level goal plus current camera frames
into a simplified next-step instruction, which is written into the observation
before the wrapped (System-1) policy computes actions. Camera-frame extraction
and the instruction key are injected so any embodiment can reuse these wrappers.
"""
