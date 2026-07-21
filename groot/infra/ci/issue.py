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


@dataclass
class Issue:
    iid: int
    title: str
    description: str
    labels: list[str]
    state: str
    created_at: str
    updated_at: str
    web_url: str

    @classmethod
    def init_from_dict(cls, data: dict) -> "Issue":
        return cls(
            iid=data["iid"],
            title=data["title"],
            description=data.get("description") or "",
            labels=data.get("labels", []),
            state=data["state"],
            created_at=data.get("created_at", ""),
            updated_at=data["updated_at"],
            web_url=data["web_url"],
        )
