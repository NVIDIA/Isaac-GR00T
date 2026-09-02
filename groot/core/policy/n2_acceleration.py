# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed N2 inference acceleration options shared by all policy packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class N2AccelerationOptions:
    """Resolve mutually dependent acceleration controls at one policy seam."""

    trt_engine_dir: Path | None
    use_cuda_graph: bool
    use_torch_compile: bool
    torch_compile_mode: str

    @classmethod
    def resolve(
        cls,
        *,
        trt_engine_dir: str | Path | None,
        use_cuda_graph: bool,
        use_torch_compile: bool,
        torch_compile_mode: str,
    ) -> N2AccelerationOptions:
        engine_dir = (
            Path(trt_engine_dir).expanduser().resolve(strict=True)
            if trt_engine_dir is not None
            else None
        )
        options = cls(
            trt_engine_dir=engine_dir,
            use_cuda_graph=use_cuda_graph,
            use_torch_compile=use_torch_compile,
            torch_compile_mode=torch_compile_mode,
        )
        options.validate()
        return options

    @property
    def use_tensorrt(self) -> bool:
        return self.trt_engine_dir is not None

    @property
    def backend(self) -> str:
        """Name the effective execution strategy without hiding composed modes."""
        if self.use_tensorrt:
            return "trt_graph" if self.use_cuda_graph else "trt"
        if self.use_torch_compile:
            return "compile_graph" if self.use_cuda_graph else "compile"
        return "graph" if self.use_cuda_graph else "eager"

    def runtime_state(self) -> dict[str, str | bool]:
        """Return the JSON-safe state consumed by serving and measurement contracts."""
        return {
            "backend": self.backend,
            "trt_engines": self.use_tensorrt,
            "cuda_graph": self.use_cuda_graph,
            "torch_compile": self.use_torch_compile,
        }

    def validate(self) -> None:
        if self.use_tensorrt and self.use_torch_compile:
            raise ValueError("TensorRT and torch.compile are mutually exclusive backends")
        if not self.torch_compile_mode:
            raise ValueError("torch_compile_mode cannot be empty")
        if self.use_cuda_graph and self.use_torch_compile and self.torch_compile_mode != "default":
            raise ValueError("manual CUDA graph capture with torch.compile requires mode='default'")
