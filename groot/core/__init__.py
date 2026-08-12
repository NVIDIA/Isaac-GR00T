# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pkgutil import extend_path as _extend_path

from .version import VERSION, VERSION_SHORT

__path__ = _extend_path(__path__, __name__)
__version__ = VERSION
__all__ = ["VERSION", "VERSION_SHORT", "__version__"]
