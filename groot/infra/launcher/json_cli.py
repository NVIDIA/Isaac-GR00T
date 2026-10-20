# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Serialize a JSON-valued CLI option after expanding launcher placeholders."""

from collections.abc import Callable
import json
from typing import Any


def serialize_json_value(value: Any, expand_string: Callable[[str], str]) -> str:
    """Keep mappings/lists/scalars in one CLI value and use the caller's path rules."""

    def expand(item: Any) -> Any:
        if isinstance(item, str):
            return expand_string(item)
        if isinstance(item, dict):
            return {key: expand(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [expand(nested) for nested in item]
        return item

    return json.dumps(expand(value))
