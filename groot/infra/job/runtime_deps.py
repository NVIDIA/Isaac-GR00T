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
"""Runtime dependency lists for Ray jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import re
from typing import ClassVar


@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime pip dependencies for a Ray job."""

    model: str
    cluster_type: str | None = None
    package_overrides: Mapping[str, str] = field(default_factory=dict)

    STANDARD_PIP_PACKAGES: ClassVar[tuple[str, ...]] = (
        "lmdb",
        "wandb",
        "tianshou==0.5.1",
        "transformers==4.51.3",
        "decord2",
        "tyro",
        "loguru",
        "structlog",
        "albumentations==1.4.18",
        "tiktoken",
        "blobfile",
        "truststore",
        "polars>=1.0",
    )
    _TRANSFORMERS_PINS: ClassVar[dict[tuple[str, str], str]] = {
        ("n2", "gb300"): "4.57.6",
    }

    def get_cluster_type(self) -> str:
        return self.cluster_type or os.environ.get("CLUSTER_TYPE", "gb200")

    def get_pip_packages(self) -> list[str]:
        """Pip package list for the model family and cluster type."""
        cluster_type = self.get_cluster_type()
        pin = self._TRANSFORMERS_PINS.get((self.model, cluster_type))
        effective: dict[str, str] = {}
        if pin is not None:
            effective["transformers"] = pin
        effective.update(self.package_overrides)

        if not effective:
            return list(self.STANDARD_PIP_PACKAGES)

        out: list[str] = []
        seen: set[str] = set()
        for pkg in self.STANDARD_PIP_PACKAGES:
            name = re.split(r"==|>=|<=|<|>", pkg, maxsplit=1)[0].strip()
            if name in effective:
                out.append(f"{name}=={effective[name]}")
                seen.add(name)
            else:
                out.append(pkg)
        for name, version in effective.items():
            if name not in seen:
                out.append(f"{name}=={version}")
        return out


STANDARD_PIP_PACKAGES = list(RuntimeDeps.STANDARD_PIP_PACKAGES)
