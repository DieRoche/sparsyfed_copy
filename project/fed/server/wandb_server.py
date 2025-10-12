"""Flower server accounting for Weights&Biases+file saving."""

import timeit
from collections.abc import Callable
from logging import INFO
from numbers import Number

from flwr.common import FitRes, Parameters
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy

from project.fed.utils.traffic import parameters_size_bytes
from project.utils.utils import cleanup_memory


class WandbServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Strategy | None = None,
        history: History | None = None,
        save_parameters_to_file: Callable[
            [Parameters],
            None,
        ],
        save_files_per_round: Callable[[int], None],
    ) -> None:
        """Flower server implementation.

        Parameters
        ----------
        client_manager : ClientManager
            Client manager implementation.
        strategy : Optional[Strategy]
            Strategy implementation.
        history : Optional[History]
            History implementation.
        save_parameters_to_file : Callable[[Parameters], None]
            Function to save the parameters to file.
        save_files_per_round : Callable[[int], None]
            Function to save files every round.

        Returns
        -------
        None
        """
        super().__init__(
            client_manager=client_manager,
            strategy=strategy,
        )

        self.history: History | None = history
        self.save_parameters_to_file = save_parameters_to_file
        self.save_files_per_round = save_files_per_round
        self._last_upload_size_bytes: float | None = None
        self._flop_totals: dict[str, float] = {
            "total_flops": 0.0,
            "total_flops_including_compression": 0.0,
            "total_flops_decompression": 0.0,
            "total_flops_compression": 0.0,
        }
        self._flop_metrics_available = False
        self._missing_flop_metrics_warned = False

    # pylint: disable=too-many-locals
    def fit(
        self,
        num_rounds: int,
        timeout: float | None,
    ) -> History:
        """Run federated averaging for a number of rounds.

        Parameters
        ----------
        num_rounds : int
            The number of rounds to run.
        timeout : Optional[float]
            Timeout in seconds.

        Returns
        -------
        History
            The history of the training.
            Potentially using a pre-defined history.
        """
        history = self.history if self.history is not None else History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(
            timeout=timeout,
        )
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(
            0,
            parameters=self.parameters,
        )
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(
                server_round=0,
                loss=res[0],
            )
            history.add_metrics_centralized(
                server_round=0,
                metrics=res[1],
            )

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        # Save initial parameters and files
        self.save_parameters_to_file(self.parameters)
        self.save_files_per_round(0)

        total_upload_traffic = 0.0
        total_download_traffic = 0.0
        last_logged_round: int | None = None

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            # prendere un timer sulla fit
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )

            if res_fit is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    fit_results_and_failures,
                ) = res_fit  # fit_metrics_aggregated

                fit_results, failures = fit_results_and_failures

                active_clients = len(fit_results) + len(failures)
                download_traffic = active_clients * parameters_size_bytes(
                    self.parameters
                )
                if fit_results:
                    on_wire_upload = parameters_size_bytes(
                        fit_results[0][1].parameters
                    )
                    self._last_upload_size_bytes = float(on_wire_upload)
                elif self._last_upload_size_bytes is not None:
                    on_wire_upload = self._last_upload_size_bytes
                else:
                    on_wire_upload = parameters_size_bytes(self.parameters)
                    self._last_upload_size_bytes = float(on_wire_upload)

                upload_traffic = active_clients * float(on_wire_upload)

                total_upload_traffic += upload_traffic
                total_download_traffic += download_traffic

                if fit_metrics is None:
                    fit_metrics = {}

                log(
                    INFO,
                    "Round %s upload size per client (bytes): %s",
                    current_round,
                    on_wire_upload,
                )

                fit_metrics.update({
                    "upload_traffic": float(upload_traffic),
                    "download_traffic": float(download_traffic),
                    "upload_traffic_per_client": float(on_wire_upload),
                    "overall_traffic": float(
                        total_upload_traffic + total_download_traffic
                    ),
                })

                self._update_flop_metrics(
                    fit_results,
                    fit_metrics,
                )

                if parameters_prime:
                    self.parameters = parameters_prime
                    # try to check the parameters sparsity here

                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=fit_metrics,
                )
                last_logged_round = current_round

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(
                current_round,
                parameters=self.parameters,
            )
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(
                    server_round=current_round,
                    loss=loss_cen,
                )
                history.add_metrics_centralized(
                    server_round=current_round,
                    metrics=metrics_cen,
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round,
                        loss=loss_fed,
                    )

                    history.add_metrics_distributed(
                        server_round=current_round,
                        metrics=evaluate_metrics_fed,
                    )

            # Saver round parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_files_per_round(current_round)
            cleanup_memory()

        if num_rounds > 0 and last_logged_round != num_rounds:
            fallback_metrics: dict[str, float] = {
                "upload_traffic": 0.0,
                "download_traffic": 0.0,
                "upload_traffic_per_client": float(
                    self._last_upload_size_bytes or 0.0
                ),
                "overall_traffic": float(
                    total_upload_traffic + total_download_traffic
                ),
            }

            if self._flop_metrics_available:
                fallback_metrics.update(
                    {
                        "round_flops": 0.0,
                        "round_flops_compression": 0.0,
                        "round_flops_decompression": 0.0,
                        "total_flops": self._flop_totals["total_flops"],
                        "total_flops_compression": self._flop_totals[
                            "total_flops_compression"
                        ],
                        "total_flops_decompression": self._flop_totals[
                            "total_flops_decompression"
                        ],
                        "total_flops_including_compression": self._flop_totals[
                            "total_flops_including_compression"
                        ],
                    }
                )

            history.add_metrics_distributed_fit(
                server_round=num_rounds,
                metrics=fallback_metrics,
            )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def _update_flop_metrics(
        self,
        fit_results: list[tuple[ClientProxy, FitRes]],
        fit_metrics: dict[str, float | int | bool | str],
    ) -> None:
        """Update per-round and cumulative FLOP metrics.

        Parameters
        ----------
        fit_results : list[tuple[ClientProxy, FitRes]]
            The successful fit results for the current round.
        fit_metrics : dict[str, float | int | bool | str]
            The aggregated metrics dictionary for the round.
        """

        flop_round_keys = (
            "round_flops",
            "round_flops_compression",
            "round_flops_decompression",
        )

        round_values = {key: 0.0 for key in flop_round_keys}
        values_found = False

        for _, fit_res in fit_results:
            metrics = getattr(fit_res, "metrics", None) or {}
            for key in flop_round_keys:
                value = metrics.get(key)
                if isinstance(value, Number):
                    round_values[key] += float(value)
                    values_found = True

        if not values_found:
            if not self._missing_flop_metrics_warned:
                log(
                    INFO,
                    (
                        "No client FLOP metrics were provided; reporting zero "
                        "values in WandB. Ensure clients populate 'round_flops', "
                        "'round_flops_compression', and 'round_flops_decompression'"
                    ),
                )
                self._missing_flop_metrics_warned = True

            zero_metrics: dict[str, float] = {key: 0.0 for key in flop_round_keys}
            zero_metrics.update(self._flop_totals)
            fit_metrics.update(zero_metrics)
            return

        self._flop_metrics_available = True
        self._missing_flop_metrics_warned = False

        fit_metrics.update(round_values)

        self._flop_totals["total_flops"] += round_values["round_flops"]
        self._flop_totals["total_flops_compression"] += round_values[
            "round_flops_compression"
        ]
        self._flop_totals["total_flops_decompression"] += round_values[
            "round_flops_decompression"
        ]
        self._flop_totals["total_flops_including_compression"] += (
            round_values["round_flops"]
            + round_values["round_flops_compression"]
            + round_values["round_flops_decompression"]
        )

        fit_metrics.update(self._flop_totals)
