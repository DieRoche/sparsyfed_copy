"""Flower server accounting for Weights&Biases+file saving."""

import time
import timeit
from collections.abc import Callable
from logging import INFO

import numpy as np
import torch
import wandb
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from project.fed.utils.utils import tensor_dict_bytes


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
        total_upload_traffic = 0
        total_download_traffic = 0

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            # prendere un timer sulla fit
            timestamp = time.time()
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            fit_round_time = time.time() - timestamp

            participating_updates = []
            fit_metrics: dict[str, float] = {}
            parameters_prime = None

            if res_fit is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    _,
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                    participating_updates = [parameters_to_ndarrays(parameters_prime)]

                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=fit_metrics,
                )

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
                # mettere la metrica qui dentro centralized / round complition time
                metrics_cen["fit_round_time"] = fit_round_time
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
            else:
                evaluate_metrics_fed = {}

            acc_clients = fit_metrics.get("acc_clients", [])
            cos_mean = fit_metrics.get("cos_mean", 0.0)
            cos_std = fit_metrics.get("cos_std", 0.0)
            training_loss_mean = fit_metrics.get("training_loss_mean", 0.0)
            training_loss_std = fit_metrics.get("training_loss_std", 0.0)
            acc = 0.0
            if res_cen is not None:
                acc = metrics_cen.get("test_accuracy", acc)
            if evaluate_metrics_fed:
                acc_clients = acc_clients or [
                    evaluate_metrics_fed.get("test_accuracy", 0.0)
                ]
            acc_clients_mean = float(np.mean(acc_clients)) if acc_clients else 0.0
            acc_clients_std = float(np.std(acc_clients)) if acc_clients else 0.0
            acc_servers = [acc]
            acc_servers_mean = float(np.mean(acc_servers))
            acc_servers_std = float(np.std(acc_servers))

            report: dict[str, float] = {
                "cos_lowest": cos_mean - cos_std,
                "cos_highest": cos_mean + cos_std,
                "training_loss_lowest": training_loss_mean - training_loss_std,
                "training_loss_highest": training_loss_mean + training_loss_std,
                "acc_clients_lowest": acc_clients_mean - acc_clients_std,
                "acc_clients_highest": acc_clients_mean + acc_clients_std,
                "acc_servers_lowest": acc_servers_mean - acc_servers_std,
                "acc_servers_highest": acc_servers_mean + acc_servers_std,
            }

            global_state = parameters_to_ndarrays(self.parameters)
            download_traffic = tensor_dict_bytes(global_state)
            upload_traffic = sum(
                tensor_dict_bytes(update) for update in participating_updates
            )
            total_upload_traffic += upload_traffic
            total_download_traffic += download_traffic
            report["upload_traffic"] = upload_traffic
            report["download_traffic"] = download_traffic
            report["overall_traffic"] = total_upload_traffic + total_download_traffic

            if self.history is not None and getattr(self.history, "use_wandb", False):
                wandb.log(report, step=current_round)

            print(
                f"Round {current_round}, Clients Acc: {acc_clients}, Server Acc: {acc_servers}"
            )
            cleanup_memory()

            # Saver round parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_files_per_round(current_round)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


def cleanup_memory() -> None:
    """Empty CUDA cache if GPU is available."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
