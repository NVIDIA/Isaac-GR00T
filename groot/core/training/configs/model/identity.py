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
