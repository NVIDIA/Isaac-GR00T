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

"""Single source of truth for the deployment CLIs' shared mode-flag value sets.

``scripts/deployment`` is not an importable package, so every value set
shared across the CLIs lives here and is imported via ``gr00t.*``. A CLI
never re-authors these strings — it imports the enum or fails on a name
that does not exist, so cross-file drift is not expressible.

Each mode flag is one enum here, imported by its CLI. The enums hold the
legitimate *subset* each tool supports (the tools really do run different
subsets — that is genuine capability, not duplication):

- :class:`ExportMode` — the two ``export_mode`` CLIs (``export_onnx_n1d7``,
  ``build_trt_pipeline``).
- :class:`VerifyMode` — ``verify_n1d7_trt`` ``--mode``.
- :class:`BenchmarkMode` — ``benchmark_inference`` ``--trt-mode``.
- :class:`BuildEngineMode` — ``build_tensorrt_engine`` ``--mode``.

Each member's value equals its name (via :func:`_generate_next_value_`), so
``tyro`` keeps the value-form CLI surface unchanged (``--mode full_pipeline``,
not ``--mode FULL_PIPELINE``) — no CLI/README/docstring edits needed when
switching from a ``Literal``.
"""

from __future__ import annotations

import enum


class _StrEnum(str, enum.Enum):
    """``enum.StrEnum`` stand-in for Python 3.10 (dGPU/Orin).

    Members are ``str`` subclasses whose value equals their name, so ``==``,
    ``in``, dict-keying, JSON, f-strings, and ``tyro`` choices all see the bare
    value. ``__str__`` is restored to ``str``'s so ``str()``/``%s`` yield the
    value rather than ``ClassName.member``.
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    __str__ = str.__str__


class ExportMode(_StrEnum):
    """Allowed values for the ``--export-mode`` flag, shared by the two
    ``export_mode`` CLIs."""

    dit_only = enum.auto()
    action_head = enum.auto()
    full_pipeline = enum.auto()


class VerifyMode(_StrEnum):
    """Allowed values for ``verify_n1d7_trt`` ``--mode``."""

    dit_only = enum.auto()
    action_head = enum.auto()
    n17_full_pipeline = enum.auto()
    vit_llm_only = enum.auto()


class BenchmarkMode(_StrEnum):
    """Allowed values for ``benchmark_inference`` ``--trt-mode``."""

    dit_only = enum.auto()
    n17_full_pipeline = enum.auto()
    vit_llm_only = enum.auto()


class BuildEngineMode(_StrEnum):
    """Allowed values for ``build_tensorrt_engine`` ``--mode``."""

    single = enum.auto()
    full_pipeline = enum.auto()
