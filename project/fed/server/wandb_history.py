"""History class which sends metrics to wandb.

Metrics are collected only at the central server, minimizing communication costs. Metric
collection only happens if wandb is turned on.
"""

from typing import Any, TypeAlias

from flwr.server.history import History

import wandb

Scalar: TypeAlias = Any

WANDB_METRIC_ALLOWLIST = {
    "compression_flops_clients",
    "decompression_flops_clients",
    "round_flops",
    "compression_flops_server",
    "decompression_flops_server",
    "acc_servers_highest",
    "overall_traffic",
    "upload_traffic",
    "download_traffic",
}


def filter_wandb_metrics(metrics: dict[str, Scalar]) -> dict[str, Scalar]:
    """Return a new W&B payload containing only allowlisted metrics."""
    return {
        key: value for key, value in metrics.items() if key in WANDB_METRIC_ALLOWLIST
    }


def log_wandb_metrics(metrics: dict[str, Scalar], server_round: int) -> None:
    """Log a filtered metric payload to W&B when at least one metric remains."""
    wandb_metrics = filter_wandb_metrics(metrics)
    if wandb_metrics:
        wandb.log(wandb_metrics, step=server_round)


class WandbHistory(History):
    """History class for training and/or evaluation metrics collection."""

    def __init__(self, use_wandb: bool = True) -> None:
        """Initialize the history.

        Parameters
        ----------
        use_wandb : bool
            Whether to use wandb.
            Turn off to avoid communication overhead.

        Returns
        -------
        None
        """
        super().__init__()

        self.use_wandb = use_wandb

    def add_loss_distributed(
        self,
        server_round: int,
        loss: float,
    ) -> None:
        """Add one loss entry (from distributed evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        loss : float
            The loss to add.

        Returns
        -------
        None
        """
        super().add_loss_distributed(server_round, loss)
        if self.use_wandb:
            log_wandb_metrics({"distributed_loss": loss}, server_round)

    def add_loss_centralized(
        self,
        server_round: int,
        loss: float,
    ) -> None:
        """Add one loss entry (from centralized evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        loss : float
            The loss to add.

        Returns
        -------
        None
        """
        super().add_loss_centralized(server_round, loss)
        if self.use_wandb:
            log_wandb_metrics({"training_loss_highest": loss}, server_round)

    def add_metrics_distributed_fit(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from distributed fit) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_distributed_fit(
            server_round,
            metrics,
        )
        if self.use_wandb:
            log_wandb_metrics(metrics, server_round)

    def add_metrics_distributed(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from distributed evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_distributed(
            server_round,
            metrics,
        )
        if self.use_wandb:
            wandb_metrics = {
                "distributed_test_accuracy" if key == "test_accuracy" else key: value
                for key, value in metrics.items()
            }
            log_wandb_metrics(wandb_metrics, server_round)

    def add_metrics_centralized(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from centralized evaluation) to history/wand.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_centralized(
            server_round,
            metrics,
        )
        if self.use_wandb:
            wandb_metrics = {
                "acc_servers_highest" if key == "test_accuracy" else key: value
                for key, value in metrics.items()
            }
            log_wandb_metrics(wandb_metrics, server_round)
