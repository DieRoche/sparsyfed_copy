"""Run federated learning experiments without Hydra configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import wandb

from config import get_config


def main() -> None:
    """Entry point for launching experiments.

    Configuration is obtained from :mod:`config.py` instead of Hydra. Only a minimal
    skeleton is provided here, which initialises Weights & Biases and prepares output
    directories. The actual training logic should be added where indicated.
    """

    args = get_config()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting run with configuration: %s", args)

    # Prepare output directories
    output_directory = Path("output")
    output_directory.mkdir(parents=True, exist_ok=True)
    results_dir = output_directory / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(
        project="compression_FL",
        config={k: v for k, v in vars(args).items()},
        settings=wandb.Settings(start_method="thread"),
    ):
        logging.info("Wandb run initialized")
        # Placeholder for the training/evaluation logic
        logging.info("No training logic is currently implemented.")


if __name__ == "__main__":
    main()

