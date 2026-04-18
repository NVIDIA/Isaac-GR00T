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

import logging
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import Config, get_default_config
from gr00t.experiment.experiment import run


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    config = tyro.cli(Config, default=get_default_config(), description=__doc__)
    # Load config from path if provided
    if config.load_config_path:
        assert Path(config.load_config_path).exists(), (
            f"Config path does not exist: {config.load_config_path}"
        )
        config = config.load(Path(config.load_config_path))  # inplace loading
        config.load_config_path = None
        logging.info(f"Loaded config from {config.load_config_path}")

        # Override with command-line.
        config = tyro.cli(Config, default=config, description=__doc__)
    run(config)
