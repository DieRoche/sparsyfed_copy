# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""


from logging import WARNING
from pathlib import Path
from typing import Optional, Union
from collections.abc import Callable

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from functools import reduce

import numpy as np

from project.fed.utils.sparse_update import (
    deserialise_sparse_update,
    reconstruct_dense_update,
)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


def original_aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def old_aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average for non-zero weights."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [
            (layer * num_examples).astype(bool) * layer for layer in weights
        ]  # Modify this line
        for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        np.divide(
            reduce(np.add, layer_updates),
            num_examples_total,
            where=(num_examples_total > 0),
        )  # Modify this line
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average for non-zero weights."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [(layer * num_examples, layer != 0) for layer in weights]
        for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        np.divide(
            np.sum([mask * layer for layer, mask in layer_updates], axis=0),
            num_examples_total,
            where=(num_examples_total > 0),
        )
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# pylint: disable=line-too-long
class FedAvgNZ(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float | dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        working_dir: Path,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.working_dir = working_dir
        self.current_weights: NDArrays | None = None
        self.param_shapes: list[tuple[int, ...]] | None = None
        self.param_dtypes: list[np.dtype] | None = None
        self.prev_mask: np.ndarray | None = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is not None:
            initial_ndarrays = parameters_to_ndarrays(initial_parameters)
            self.current_weights = [np.copy(layer) for layer in initial_ndarrays]
            self.param_shapes = [layer.shape for layer in initial_ndarrays]
            self.param_dtypes = [layer.dtype for layer in initial_ndarrays]
            self.prev_mask = None
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res  # type: ignore[misc]
        return loss, metrics  # type: ignore[has-type]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.current_weights is None:
            first_weights = parameters_to_ndarrays(results[0][1].parameters)
            self.current_weights = [np.zeros_like(layer) for layer in first_weights]
            self.param_shapes = [layer.shape for layer in first_weights]
            self.param_dtypes = [layer.dtype for layer in first_weights]
            self.prev_mask = None

        if self.param_shapes is None or self.param_dtypes is None:
            raise ValueError("Parameter metadata unavailable for aggregation")

        dense_updates: list[tuple[list[np.ndarray], int]] = []
        uplink_nonzeros = 0

        for _, fit_res in results:
            is_sparse = bool(fit_res.metrics.get("sparse_payload", 0.0))
            if is_sparse:
                serialization = deserialise_sparse_update(fit_res.parameters)
                client_update = reconstruct_dense_update(
                    serialization, self.param_shapes, self.param_dtypes
                )
                uplink_nonzeros += int(serialization.metadata[:, 1].sum())
            else:
                client_weights = parameters_to_ndarrays(fit_res.parameters)
                uplink_nonzeros += sum(layer.size for layer in client_weights)
                if self.current_weights is None:
                    raise ValueError("Current global weights unavailable for delta computation")
                client_update = [
                    layer - base
                    for layer, base in zip(
                        client_weights, self.current_weights, strict=True
                    )
                ]
                uplink_nonzeros += sum(np.count_nonzero(layer) for layer in client_update)

            dense_updates.append((client_update, fit_res.num_examples))

        total_examples = sum(num_examples for _, num_examples in dense_updates)
        if total_examples <= 0:
            total_examples = 1

        aggregated_update = [np.zeros_like(layer) for layer in self.current_weights]
        for update_layers, num_examples in dense_updates:
            weight = num_examples / total_examples
            for idx, delta in enumerate(update_layers):
                aggregated_update[idx] += delta * weight

        self.current_weights = [
            base + delta for base, delta in zip(self.current_weights, aggregated_update, strict=True)
        ]

        parameters_aggregated = ndarrays_to_parameters(self.current_weights)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        total_nonzero = 0
        total_elements = 0
        layer_sparsities: list[float] = []
        for layer in self.current_weights:
            nonzero = np.count_nonzero(layer)
            total = layer.size
            total_nonzero += nonzero
            total_elements += total
            sparsity = 1.0 - (nonzero / total) if total else 0.0
            layer_sparsities.append(float(sparsity))

        global_sparsity = 1.0 - (total_nonzero / total_elements) if total_elements else 0.0

        current_mask = np.concatenate(
            [layer != 0 for layer in self.current_weights]
        ).astype(bool)
        if self.prev_mask is None:
            mask_iou = 1.0
        else:
            union = np.logical_or(current_mask, self.prev_mask)
            intersection = np.logical_and(current_mask, self.prev_mask)
            mask_iou = float(intersection.sum() / union.sum()) if union.any() else 1.0
        self.prev_mask = current_mask

        metrics_aggregated["global_sparsity"] = float(global_sparsity)
        metrics_aggregated["mask_iou"] = float(mask_iou)
        for idx, sparsity in enumerate(layer_sparsities):
            metrics_aggregated[f"layer_sparsity_{idx}"] = float(sparsity)

        metrics_aggregated["uplink_nonzero_total"] = float(uplink_nonzeros)
        metrics_aggregated["downlink_nonzero_total"] = float(total_nonzero)
        metrics_aggregated["downlink_density"] = (
            float(total_nonzero) / float(total_elements) if total_elements else 0.0
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
