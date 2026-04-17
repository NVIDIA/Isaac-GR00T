# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        print(f"Error importing module gr00t.configs.model.{file.stem}: {e}")


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
