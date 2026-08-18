#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Rewrite the reusable Genie Sim data config for the selected OSMO dataset."""

from __future__ import annotations

import sys

import yaml


def main() -> None:
    source, dataset, output = sys.argv[1:]
    with open(source, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["data"]["datasets"][0]["dataset_paths"] = [dataset]

    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()
