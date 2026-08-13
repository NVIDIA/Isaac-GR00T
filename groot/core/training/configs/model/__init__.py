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
import os
from pathlib import Path
import typing

import tyro

MODEL_CONFIG_TYPES: dict[str, type] = {}


def register_model_config(shortname: str, configtype: type):
    MODEL_CONFIG_TYPES[shortname] = configtype


# These dynamic import loops must be run before create_model_union_type is
# *called* (in base_config.py).
# They import the register_model_config function above and use it to add their
# models to the MODEL_CONFIG_TYPES dict.

# Add all model configs in groot/core/training/configs/model
for file in Path(__file__).parent.glob("*.py"):
    if file.stem.startswith("_"):
        continue
    try:
        importlib.import_module(f".{file.stem}", __name__)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error importing module groot.core.training.configs.model.{file.stem}: {e}")

# Add all model configs from GROOT_TRAINING_EXTENSIONS
for module_path in os.environ.get("GROOT_TRAINING_EXTENSIONS", "").strip().split(":"):
    if module_path:
        try:
            importlib.import_module(module_path)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error importing training extension {module_path}: {e}")


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
