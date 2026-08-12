# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CLI for OSS post-training open-loop evaluation."""

from __future__ import annotations

import tyro

from groot.infra.eval.oss_open_loop_eval_config import OssOpenLoopEvalConfig
from groot.infra.eval.oss_open_loop_evaluator import OssOpenLoopEvaluator


def main(config: OssOpenLoopEvalConfig) -> None:
    OssOpenLoopEvaluator(config).run()


if __name__ == "__main__":
    main(tyro.cli(OssOpenLoopEvalConfig))
