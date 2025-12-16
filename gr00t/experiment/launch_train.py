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
