"""Flower server accounting for Weights&Biases+file saving."""

import timeit
from collections.abc import Callable
from logging import INFO
from numbers import Number

from flwr.common import FitRes, Parameters
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy

from project.fed.utils.traffic import parameters_size_bytes
from project.fed.compression.sparse_transport import decode_sparse_transport_if_needed
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
        self._flop_totals: dict[str, float] = {
            "total_flops": 0.0,
            "total_flops_compression": 0.0,
            "total_serialization_flops": 0.0,
        }
        self._flop_metrics_available = False
        self._missing_flop_metrics_warned = False
        self._fit_stage_flops_by_round: dict[int, dict[str, float]] = {}

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

                active_clients = int(len(fit_results) + len(failures))
                upload_traffic = self._compute_upload_traffic_for_round(fit_results)
                download_traffic = self._compute_download_traffic_for_round(
                    server_payload=self.parameters,
                    active_clients=active_clients,
                )

                if fit_metrics is None:
                    fit_metrics = {}

                fit_metrics.update({
                    "upload_traffic": float(upload_traffic),
                    "download_traffic": float(download_traffic),
                    "overall_traffic": float(upload_traffic + download_traffic),
                })

                self._update_transport_metrics(
                    fit_results=fit_results,
                    fit_metrics=fit_metrics,
                    upload_traffic=float(upload_traffic),
                )

                self._update_fit_stage_flop_metrics(
                    server_round=current_round,
                    fit_results=fit_results,
                    fit_metrics=fit_metrics,
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
                if evaluate_metrics_fed is None:
                    evaluate_metrics_fed = {}
                self._finalize_round_flop_metrics(
                    server_round=current_round,
                    evaluate_metrics=evaluate_metrics_fed,
                )
                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=evaluate_metrics_fed,
                )
            elif current_round in self._fit_stage_flops_by_round:
                eval_only_metrics: dict[str, float] = {}
                self._finalize_round_flop_metrics(
                    server_round=current_round,
                    evaluate_metrics=eval_only_metrics,
                )
                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=eval_only_metrics,
                )

            # Saver round parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_files_per_round(current_round)
            cleanup_memory()

        if num_rounds > 0 and last_logged_round != num_rounds:
            fallback_metrics: dict[str, float] = {
                "upload_traffic": 0.0,
                "download_traffic": 0.0,
                "overall_traffic": 0.0,
            }

            if self._flop_metrics_available:
                fallback_metrics.update(
                    {
                        "round_flops": 0.0,
                        "training_flops": 0.0,
                        "aggregation_flops": 0.0,
                        "evaluation_flops": 0.0,
                        "compression_flops_clients": 0.0,
                        "compression_flops_server": 0.0,
                        "decompression_flops_clients": 0.0,
                        "decompression_flops_server": 0.0,
                        "serialization_flops": 0.0,
                        "round_flops_compression": 0.0,
                        "total_flops": self._flop_totals["total_flops"],
                        "total_flops_compression": self._flop_totals[
                            "total_flops_compression"
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

    def _update_transport_metrics(
        self,
        fit_results: list[tuple[ClientProxy, FitRes]],
        fit_metrics: dict[str, float | int | bool | str],
        upload_traffic: float,
    ) -> None:
        """Aggregate and inject sparse transport metrics from client fit results."""

        sums = {
            "upload_dense_bytes": 0.0,
            "upload_payload_bytes": 0.0,
            "upload_csr_tensors": 0.0,
            "upload_mask_value_tensors": 0.0,
            "upload_dense_tensors": 0.0,
            "upload_total_nnz": 0.0,
            "upload_total_numel": 0.0,
            "upload_csr_expected_but_low_sparsity_tensors": 0.0,
            "upload_csr_expected_but_low_sparsity_numel": 0.0,
        }
        clients_reporting_dense_reference = 0

        for _, fit_res in fit_results:
            metrics = getattr(fit_res, "metrics", None) or {}
            dense_bytes_value = metrics.get("upload_dense_bytes")
            if isinstance(dense_bytes_value, Number):
                clients_reporting_dense_reference += 1
            for key in sums:
                value = metrics.get(key, 0.0)
                if isinstance(value, Number):
                    sums[key] += float(value)

        dense_reference = sums["upload_dense_bytes"]
        if clients_reporting_dense_reference <= 0:
            dense_reference = float(upload_traffic)
        total_numel = sums["upload_total_numel"]
        total_nnz = sums["upload_total_nnz"]

        compression_ratio = (
            float(upload_traffic / dense_reference) if dense_reference > 0 else 1.0
        )

        fit_metrics.update({
            "upload_dense_reference_traffic": float(dense_reference),
            "upload_transport_payload_bytes_reported": float(sums["upload_payload_bytes"]),
            "upload_transport_compression_ratio": float(compression_ratio),
            "upload_transport_saving_bytes": float(dense_reference - upload_traffic),
            "upload_transport_saving_ratio": float(1.0 - compression_ratio),
            "upload_transport_csr_tensors": float(sums["upload_csr_tensors"]),
            "upload_transport_mask_value_tensors": float(sums["upload_mask_value_tensors"]),
            "upload_transport_dense_tensors": float(sums["upload_dense_tensors"]),
            "upload_transport_total_nnz": float(total_nnz),
            "upload_transport_total_numel": float(total_numel),
            "upload_transport_actual_sparsity": float(1.0 - (total_nnz / total_numel)) if total_numel > 0 else 0.0,
            "upload_csr_expected_but_low_sparsity_tensors": float(sums["upload_csr_expected_but_low_sparsity_tensors"]),
            "upload_csr_expected_but_low_sparsity_numel": float(sums["upload_csr_expected_but_low_sparsity_numel"]),
        })

    def _compute_upload_traffic_for_round(
        self,
        fit_results: list[tuple[ClientProxy, FitRes]],
    ) -> float:
        """Return the per-round upload traffic from successful active clients."""

        return float(
            sum(
                parameters_size_bytes(fit_res.parameters)
                for _, fit_res in fit_results
            )
        )

    def _compute_download_traffic_for_round(
        self,
        server_payload: Parameters | None,
        active_clients: int,
    ) -> float:
        """Return the per-round download traffic to active clients."""

        if active_clients <= 0:
            return 0.0
        return float(active_clients * parameters_size_bytes(server_payload))

    def _update_fit_stage_flop_metrics(
        self,
        server_round: int,
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

        round_values = {
            "training_flops": 0.0,
            "aggregation_flops": 0.0,
            "compression_flops_clients": 0.0,
            "compression_flops_server": 0.0,
            "decompression_flops_clients": 0.0,
            "decompression_flops_server": 0.0,
            "serialization_flops": 0.0,
            "round_flops_compression": 0.0,
        }

        def _metric_as_float(key: str) -> float:
            value = fit_metrics.get(key)
            return float(value) if isinstance(value, Number) else 0.0

        existing_values = {
            "training_flops": _metric_as_float("training_flops"),
            "aggregation_flops": _metric_as_float("aggregation_flops"),
            "compression_flops_clients": _metric_as_float(
                "compression_flops_clients"
            ),
            "compression_flops_server": _metric_as_float("compression_flops_server"),
            "decompression_flops_clients": _metric_as_float(
                "decompression_flops_clients"
            ),
            "decompression_flops_server": _metric_as_float(
                "decompression_flops_server"
            ),
            "serialization_flops": _metric_as_float("serialization_flops"),
            "round_flops_compression": _metric_as_float("round_flops_compression"),
        }
        values_found = False

        for _, fit_res in fit_results:
            metrics = getattr(fit_res, "metrics", None) or {}
            round_flops = metrics.get("round_flops")
            serialization_flops = metrics.get("serialization_flops")
            compression_flops_clients = metrics.get("compression_flops_clients")
            compression_flops_server = metrics.get("compression_flops_server")
            decompression_flops_clients = metrics.get("decompression_flops_clients")
            decompression_flops_server = metrics.get("decompression_flops_server")
            legacy_compression = metrics.get("round_flops_compression")
            legacy_decompression = metrics.get("round_flops_decompression")

            training_flops = metrics.get("training_flops")
            per_client_training_flops = (
                float(training_flops) if isinstance(training_flops, Number) else 0.0
            )
            if per_client_training_flops <= 0.0 and isinstance(round_flops, Number):
                per_client_training_flops = float(round_flops)
            aggregation_flops = metrics.get("aggregation_flops")
            # Intentionally ignore fit-stage evaluation_flops from client fit metrics.
            # Option B finalizes evaluation compute only after distributed evaluation
            # aggregation to avoid double counting and provisional round_flops logging.
            per_client_aggregation_flops = (
                float(aggregation_flops)
                if isinstance(aggregation_flops, Number)
                else 0.0
            )
            per_client_serialization_flops = (
                float(serialization_flops)
                if isinstance(serialization_flops, Number)
                else 0.0
            )
            has_split_compression_clients = isinstance(
                compression_flops_clients,
                Number,
            )
            per_client_compression_flops_clients = (
                float(compression_flops_clients)
                if has_split_compression_clients
                else 0.0
            )
            per_client_compression_flops_server = (
                float(compression_flops_server)
                if isinstance(compression_flops_server, Number)
                else 0.0
            )
            has_split_decompression_clients = isinstance(
                decompression_flops_clients,
                Number,
            )
            per_client_decompression_flops_clients = (
                float(decompression_flops_clients)
                if has_split_decompression_clients
                else 0.0
            )
            per_client_decompression_flops_server = (
                float(decompression_flops_server)
                if isinstance(decompression_flops_server, Number)
                else 0.0
            )

            if (
                not has_split_compression_clients
                and isinstance(legacy_compression, Number)
            ):
                per_client_compression_flops_clients = float(legacy_compression)

            if (
                not has_split_decompression_clients
                and isinstance(legacy_decompression, Number)
            ):
                per_client_decompression_flops_clients = float(legacy_decompression)

            per_client_round_compression = (
                per_client_compression_flops_clients
                + per_client_compression_flops_server
                + per_client_decompression_flops_clients
                + per_client_decompression_flops_server
                + per_client_serialization_flops
            )

            round_values["training_flops"] += per_client_training_flops
            round_values["aggregation_flops"] += per_client_aggregation_flops
            round_values["compression_flops_clients"] += (
                per_client_compression_flops_clients
            )
            round_values["compression_flops_server"] += (
                per_client_compression_flops_server
            )
            round_values["decompression_flops_clients"] += (
                per_client_decompression_flops_clients
            )
            round_values["decompression_flops_server"] += (
                per_client_decompression_flops_server
            )
            round_values["serialization_flops"] += per_client_serialization_flops
            round_values["round_flops_compression"] += per_client_round_compression

            values_found = values_found or any(
                value > 0.0
                for value in (
                    per_client_training_flops,
                    per_client_aggregation_flops,
                    per_client_compression_flops_clients,
                    per_client_compression_flops_server,
                    per_client_decompression_flops_clients,
                    per_client_decompression_flops_server,
                    per_client_serialization_flops,
                    per_client_round_compression,
                )
            )

        if not values_found and _metric_as_float("round_flops") <= 0.0:
            if not self._missing_flop_metrics_warned:
                log(
                    INFO,
                    (
                        "No client FLOP metrics were provided; reporting zero "
                        "values in WandB. Ensure clients populate 'round_flops', "
                        "compression/decompression split metrics, and "
                        "'serialization_flops' if available."
                    ),
                )
                self._missing_flop_metrics_warned = True

            zero_metrics: dict[str, float] = {
                "training_flops": 0.0,
                "aggregation_flops": 0.0,
                "evaluation_flops": 0.0,
                "fit_round_flops": 0.0,
                "round_flops_without_evaluation": 0.0,
                "compression_flops_clients": 0.0,
                "compression_flops_server": 0.0,
                "decompression_flops_clients": 0.0,
                "decompression_flops_server": 0.0,
                "serialization_flops": 0.0,
                "round_flops_compression": 0.0,
            }
            zero_metrics.update(self._flop_totals)
            fit_metrics.update(zero_metrics)
            return

        self._flop_metrics_available = True
        self._missing_flop_metrics_warned = False

        merged_values = round_values.copy()
        for key, existing_value in existing_values.items():
            if merged_values[key] <= 0.0 and existing_value > 0.0:
                merged_values[key] = existing_value

        total_parameters = 0
        if fit_results:
            first_params = parameters_to_ndarrays(fit_results[0][1].parameters)
            decoded_params = decode_sparse_transport_if_needed(first_params)
            total_parameters = sum(int(arr.size) for arr in decoded_params)
        server_aggregation_flops = float(total_parameters * len(fit_results))
        merged_values["aggregation_flops"] = max(
            merged_values["aggregation_flops"],
            server_aggregation_flops,
        )
        merged_values["fit_round_flops"] = (
            merged_values["training_flops"] + merged_values["aggregation_flops"]
        )
        merged_values["round_flops_without_evaluation"] = merged_values["fit_round_flops"]
        merged_values["evaluation_flops"] = 0.0
        merged_values["round_flops_compression"] = max(
            merged_values["compression_flops_clients"]
            + merged_values["compression_flops_server"]
            + merged_values["decompression_flops_clients"]
            + merged_values["decompression_flops_server"]
            + merged_values["serialization_flops"],
            merged_values["round_flops_compression"],
        )

        fit_metrics.update(merged_values)
        # Keep fit-stage logging free of provisional round_flops; final round_flops is
        # emitted only once in _finalize_round_flop_metrics after evaluation aggregation.
        fit_metrics.pop("round_flops", None)

        fit_metrics.pop("round_flops_decompression", None)
        fit_metrics.pop("total_flops_decompression", None)
        fit_metrics.pop("total_flops_including_compression", None)

        fit_metrics["total_flops"] = self._flop_totals["total_flops"]
        fit_metrics["total_flops_compression"] = self._flop_totals["total_flops_compression"]
        fit_metrics["total_serialization_flops"] = self._flop_totals["total_serialization_flops"]
        self._fit_stage_flops_by_round[server_round] = {
            key: float(fit_metrics.get(key, 0.0))
            for key in (
                "training_flops",
                "aggregation_flops",
                "fit_round_flops",
                "round_flops_without_evaluation",
                "compression_flops_clients",
                "compression_flops_server",
                "decompression_flops_clients",
                "decompression_flops_server",
                "serialization_flops",
                "round_flops_compression",
            )
        }

    def _finalize_round_flop_metrics(
        self,
        server_round: int,
        evaluate_metrics: dict[str, float | int | bool | str],
    ) -> None:
        fit_values = self._fit_stage_flops_by_round.pop(server_round, None)
        if fit_values is None:
            return
        evaluation_flops = evaluate_metrics.get("evaluation_flops", 0.0)
        evaluation_total = float(evaluation_flops) if isinstance(evaluation_flops, Number) else 0.0
        fit_round_flops = float(fit_values.get("fit_round_flops", 0.0))
        round_flops = fit_round_flops + evaluation_total
        evaluate_metrics["training_flops"] = float(fit_values["training_flops"])
        evaluate_metrics["aggregation_flops"] = float(fit_values["aggregation_flops"])
        evaluate_metrics["fit_round_flops"] = fit_round_flops
        evaluate_metrics["round_flops_without_evaluation"] = float(fit_values["round_flops_without_evaluation"])
        evaluate_metrics["evaluation_flops"] = evaluation_total
        evaluate_metrics["round_flops"] = round_flops
        evaluate_metrics["compression_flops_clients"] = float(fit_values["compression_flops_clients"])
        evaluate_metrics["compression_flops_server"] = float(fit_values["compression_flops_server"])
        evaluate_metrics["decompression_flops_clients"] = float(fit_values["decompression_flops_clients"])
        evaluate_metrics["decompression_flops_server"] = float(fit_values["decompression_flops_server"])
        evaluate_metrics["serialization_flops"] = float(fit_values["serialization_flops"])
        evaluate_metrics["round_flops_compression"] = float(fit_values["round_flops_compression"])
        self._flop_totals["total_flops"] += round_flops
        self._flop_totals["total_flops_compression"] += float(fit_values["round_flops_compression"])
        self._flop_totals["total_serialization_flops"] += float(fit_values["serialization_flops"])
        evaluate_metrics["total_flops"] = self._flop_totals["total_flops"]
        evaluate_metrics["total_flops_compression"] = self._flop_totals["total_flops_compression"]
        evaluate_metrics["total_serialization_flops"] = self._flop_totals["total_serialization_flops"]
