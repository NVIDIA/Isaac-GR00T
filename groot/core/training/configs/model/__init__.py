# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib
from pathlib import Path
import typing

import tyro

MODEL_CONFIG_TYPES: dict[str, type] = {}


def register_model_config(shortname: str, configtype: type):
    MODEL_CONFIG_TYPES[shortname] = configtype


for file in Path(__file__).parent.glob("*.py"):
    if file.stem.startswith("_"):
        continue
    try:
        importlib.import_module(f".{file.stem}", __name__)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error importing module groot.core.training.configs.model.{file.stem}: {e}")


def create_model_union_type():
    if not MODEL_CONFIG_TYPES:
        # A Union of no types is invalid, so just return None
        return None

    annotated_types = tuple(
        typing.Annotated[model_type, tyro.conf.subcommand(model_shortname)]
        for model_shortname, model_type in MODEL_CONFIG_TYPES.items()
    )

    # Create the Union dynamically
    return typing.Union.__getitem__(annotated_types)
