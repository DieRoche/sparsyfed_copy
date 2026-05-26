"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

import json
import math
from pathlib import Path


import flwr as fl
import numpy as np
from flwr.common import NDArrays
from pydantic import BaseModel
import torch
from torch import nn

from project.fed.compression.sparse_transport import encode_sparse_transport
from project.fed.utils.utils import count_nonzero_elements, generic_get_parameters, generic_set_parameters, get_nonzeros

from project.types.common import (
    ClientDataloaderGen,
    ClientGen,
    EvalRes,
    FedDataloaderGen,
    FitRes,
    NetGen,
    TestFunc,
    TrainFunc,
)
from project.utils.utils import cleanup_memory, obtain_device


class ClientConfig(BaseModel):
    """Fit/eval config, allows '.' member access and static checking.

    Used to check weather each component has its own independent config present. Each
    component should then use its own Pydantic model to validate its config. For
    anything extra, use the extra field as a simple dict.
    """

    # Instantiate model
    net_config: dict
    # Instantiate dataloader
    dataloader_config: dict
    # For train/test
    run_config: dict
    # Additional params used like a Dict
    extra: dict

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    _PRUNE_EVENT_COST = 1.0
    _REGROW_EVENT_COST = 6.0
    _SIGN_FLIP_COST = 1.0
    _LAYER_OSCILLATION_COST = 4.0


    def _estimate_sparse_dynamics_flops(
        self,
        before_arrays: NDArrays,
        after_arrays: NDArrays,
    ) -> tuple[float, dict[str, float]]:
        """Estimate dynamic sparsity FLOPs from prune/regrow oscillations."""
        if not before_arrays or not after_arrays:
            return 0.0, {}

        paired_len = min(len(before_arrays), len(after_arrays))
        total = 0.0
        per_layer: dict[str, float] = {}

        for idx in range(paired_len):
            before = before_arrays[idx]
            after = after_arrays[idx]
            if (
                not isinstance(before, np.ndarray)
                or not isinstance(after, np.ndarray)
                or before.shape != after.shape
                or not np.issubdtype(before.dtype, np.number)
                or not np.issubdtype(after.dtype, np.number)
            ):
                continue

            b = before.reshape(-1)
            a = after.reshape(-1)
            if b.size == 0:
                continue

            b_nz = b != 0.0
            a_nz = a != 0.0

            pruned = int(np.count_nonzero(b_nz & (~a_nz)))
            regrown = int(np.count_nonzero((~b_nz) & a_nz))
            sign_flips = int(
                np.count_nonzero(
                    (b_nz & a_nz) & (np.signbit(b) != np.signbit(a))
                )
            )

            before_density = float(np.count_nonzero(b_nz)) / float(b.size)
            after_density = float(np.count_nonzero(a_nz)) / float(a.size)
            density_delta = abs(after_density - before_density)
            oscillation = density_delta * float(a.size)

            layer_flops = (
                self._PRUNE_EVENT_COST * float(pruned)
                + self._REGROW_EVENT_COST * float(regrown)
                + self._SIGN_FLIP_COST * float(sign_flips)
                + self._LAYER_OSCILLATION_COST * float(oscillation)
            )
            if layer_flops <= 0.0:
                continue

            per_layer[f"layer_{idx}_sparsity_dynamics_flops"] = float(layer_flops)
            total += layer_flops

        return float(total), per_layer

    def _should_count_sparsity_dynamics_flops(self, config: ClientConfig) -> bool:
        """Return True when dynamic sparsity FLOP accounting is explicitly enabled."""
        return bool(config.extra.get("count_sparsity_dynamics_flops", False))

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
        train: TrainFunc,
        test: TestFunc,
        fed_dataloader_gen: FedDataloaderGen,
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str
            The client's ID.
        working_dir : Path
            The path to the working directory.
        net_generator : NetGen
            The network generator.
        dataloader_gen : ClientDataloaderGen
            The dataloader generator.
            Uses the client id to determine partition.

        Returns
        -------
        None
        """
        super().__init__()
        self.cid = cid
        self.net_generator = net_generator
        self.working_dir = working_dir
        self.net: nn.Module | None = None
        self.dataloader_gen = dataloader_gen
        self.train = train
        self.test = test
        self.fed_dataloader_gen = fed_dataloader_gen

    def fit(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> FitRes:
        """Fit the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside fit, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for training.
        _config : Dict
            The configuration for the training.
            Uses the pydantic model for static checking.

        Returns
        -------
        FitRes
            The parameters after training, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()
        config.run_config["curr_round"] = config.extra["curr_round"]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )

        server_nonzero_count, server_total_count = count_nonzero_elements(parameters)

        trainloader = self.dataloader_gen(
            self.cid,
            False,
            config.dataloader_config,
        )

        try:
            config.run_config["cid"] = self.cid

            num_samples, metrics = self.train(
                self.net,
                trainloader,
                config.run_config,
                self.working_dir,
            )

            metrics["learning_rate"] = config.run_config["learning_rate"]

            updated_parameters = generic_get_parameters(self.net)
            client_nonzero_count, client_total_count = count_nonzero_elements(
                updated_parameters
            )

            try:
                num_batches = len(trainloader)
            except TypeError:
                num_batches = 0

            loader_batch_size = getattr(trainloader, "batch_size", None)
            if not isinstance(loader_batch_size, int) or loader_batch_size <= 0:
                loader_batch_size = config.dataloader_config.get("batch_size")
            if not isinstance(loader_batch_size, int) or loader_batch_size <= 0:
                if isinstance(num_batches, int) and num_batches > 0:
                    loader_batch_size = max(
                        math.ceil(float(num_samples) / float(num_batches)),
                        1,
                    )
                else:
                    loader_batch_size = max(int(num_samples), 1)

            if not isinstance(num_batches, int) or num_batches <= 0:
                num_batches = max(
                    math.ceil(float(num_samples) / float(loader_batch_size)),
                    1,
                )

            training_flops = float(metrics.get("training_flops", 0.0))
            evaluation_flops = float(metrics.get("evaluation_flops", 0.0))
            aggregation_flops = float(metrics.get("aggregation_flops", 0.0))
            round_flops = training_flops + evaluation_flops + aggregation_flops
            if round_flops <= 0.0:
                legacy_round_flops = metrics.get("round_flops", 0.0)
                if isinstance(legacy_round_flops, (float, int)):
                    round_flops = float(legacy_round_flops)

            metrics["server_to_client_nonzero"] = float(server_nonzero_count)
            metrics["server_to_client_density"] = (
                float(server_nonzero_count) / float(server_total_count)
                if server_total_count
                else 0.0
            )
            metrics["client_to_server_nonzero"] = float(client_nonzero_count)
            metrics["client_to_server_density"] = (
                float(client_nonzero_count) / float(client_total_count)
                if client_total_count
                else 0.0
            )
            metrics["nonzero_communication_total"] = float(
                server_nonzero_count + client_nonzero_count
            )
            metrics["round_flops"] = round_flops
            if self._should_count_sparsity_dynamics_flops(config):
                dynamics_flops, per_layer_dynamics_flops = (
                    self._estimate_sparse_dynamics_flops(
                        parameters, updated_parameters
                    )
                )
                metrics["sparsity_dynamics_flops"] = dynamics_flops
                metrics.update(per_layer_dynamics_flops)
            else:
                metrics["sparsity_dynamics_flops"] = 0.0
            transport_cfg = config.extra.get("transport_compression", {})
            compress_uplink = bool(
                transport_cfg.get("enabled", False)
                and transport_cfg.get("compress_uplink", False)
            )

            updates_dir_raw = config.extra.get("client_updates_dir")
            if updates_dir_raw:
                updates_dir = Path(updates_dir_raw)
                updates_dir.mkdir(parents=True, exist_ok=True)
                base_name = (
                    f"client_{self.cid}_round_{config.extra['curr_round']}"
                )
                np.savez_compressed(
                    updates_dir / f"{base_name}.npz",
                    *updated_parameters,
                )

                metrics_to_store: dict[str, float | int | str | bool] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.floating, float)):
                        metrics_to_store[key] = float(value)
                    elif isinstance(value, (np.integer, int)):
                        metrics_to_store[key] = int(value)
                    elif isinstance(value, (np.bool_, bool)):
                        metrics_to_store[key] = bool(value)
                    else:
                        metrics_to_store[key] = value

                with open(
                    updates_dir / f"{base_name}.json",
                    "w",
                    encoding="utf-8",
                ) as meta_file:
                    json.dump(
                        {
                            "num_samples": int(num_samples),
                            "metrics": metrics_to_store,
                        },
                        meta_file,
                    )

            if (
                compress_uplink
            ):
                updated_parameters, transport_metrics = encode_sparse_transport(
                    arrays=updated_parameters,
                    cfg=transport_cfg,
                    curr_round=int(config.extra["curr_round"]),
                    target_sparsity=float(
                        transport_cfg.get(
                            "target_sparsity",
                            transport_cfg.get("sparsity", 0.0),
                        )
                    ),
                )
                metrics.update(transport_metrics)
            metrics["compression_flops_clients"] = float(
                metrics.get("compression_flops_clients", 0.0)
            )
            metrics["compression_flops_server"] = float(
                metrics.get("compression_flops_server", 0.0)
            )
            metrics["decompression_flops_clients"] = float(
                metrics.get("decompression_flops_clients", 0.0)
            )
            metrics["decompression_flops_server"] = float(
                metrics.get("decompression_flops_server", 0.0)
            )
            metrics["serialization_flops"] = float(metrics.get("serialization_flops", 0.0))
            metrics["round_flops_compression"] = (
                metrics["compression_flops_clients"]
                + metrics["compression_flops_server"]
                + metrics["decompression_flops_clients"]
                + metrics["decompression_flops_server"]
                + metrics["serialization_flops"]
            )
            metrics["round_flops_decompression"] = (
                metrics["decompression_flops_clients"]
                + metrics["decompression_flops_server"]
            )

            return (
                updated_parameters,
                num_samples,
                metrics,
            )
        finally:
            parameters = None
            trainloader = None
            if self.net is not None:
                self.net.to("cpu")
                self.net = None
            cleanup_memory()

    def evaluate(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> EvalRes:
        """Evaluate the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside eval, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for evaluation.
        _config : Dict
            The configuration for the evaluation.
            Uses the pydantic model for static checking.

        Returns
        -------
        EvalRes
            The loss, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()
        config.run_config["curr_round"] = config.extra["curr_round"]

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        sparsity = get_nonzeros(self.net)

        testloader = self.dataloader_gen(
            self.cid,
            True,
            config.dataloader_config,
        )

        try:
            loss, num_samples, metrics = self.test(
                self.net,
                testloader,
                config.run_config,
                self.working_dir,
            )

            self.net = self.set_parameters(
                parameters,
                config.net_config,
            )

            metrics["sparsity"] = sparsity
            metrics["cid"] = self.cid

            return loss, num_samples, metrics
        finally:
            testloader = None
            if self.net is not None:
                self.net.to("cpu")
                self.net = None
            cleanup_memory()

    def get_parameters(self, config: dict) -> NDArrays:
        """Obtain client parameters.

        If the network is currently none,generate a network using the net_generator.

        Parameters
        ----------
        config : Dict
            The configuration for the training.

        Returns
        -------
        NDArrays
            The parameters of the network.
        """
        if self.net is None:
            except_str: str = """Network is None.
                Call set_parameters first and
                do not use this template without a get_initial_parameters function.
            """
            raise ValueError(
                except_str,
            )

        return generic_get_parameters(self.net)

    def set_parameters(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> nn.Module:
        """Set client parameters.

        First generated the network. Only call this in fit/eval.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to set.
        config : Dict
            The configuration for the network generator.

        Returns
        -------
        nn.Module
            The network with the new parameters.
        """
        net = self.net_generator(config)
        generic_set_parameters(
            net,
            parameters,
            to_copy=False,
        )
        return net

    def __repr__(self) -> str:
        """Implement the string representation based on cid."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: dict) -> dict:
        """Implement how to get properties."""
        return {}


def get_client_generator(
    working_dir: Path,
    net_generator: NetGen,
    dataloader_gen: ClientDataloaderGen,
    train: TrainFunc,
    test: TestFunc,
    fed_dataloader_gen: FedDataloaderGen,
) -> ClientGen:
    """Return a function which creates a new Client.

    Client has access to the working dir,
    can generate a network and can generate a dataloader.
    The client receives train and test functions with pre-defined APIs.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
        Please respect the pydantic schema.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.
        Please respect the pydantic schema.
    train : TrainFunc
        The train function.
        Please respect the interface and pydantic schema.
    test : TestFunc
        The test function.
        Please respect the interface and pydantic schema.

    Returns
    -------
    ClientGen
        The function which creates a new Client.
    """

    def client_generator(cid: int | str) -> Client:
        """Return a new Client.

        Parameters
        ----------
        cid : int | str
            The client's ID.

        Returns
        -------
        Client
            The new Client.
        """
        return Client(
            cid,
            working_dir,
            net_generator,
            dataloader_gen,
            train,
            test,
            fed_dataloader_gen,
        )

    return client_generator
