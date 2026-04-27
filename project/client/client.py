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

from project.fed.utils.utils import (
    count_nonzero_elements,
    estimate_forward_flops,
    generic_get_parameters,
    generic_set_parameters,
    get_nonzeros,
)

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

    _BACKWARD_MULTIPLIER = 2.0
    _OPTIMIZER_COST = 2.0
    _FLOP_SAMPLE_SHAPE = (128, 3, 32, 32)
    _SPARSE_SCAN_COST = 1.0
    _SPARSE_INDEX_WRITE_COST = 1.0
    _QUANTIZE_COST = 4.0
    _DEQUANTIZE_COST = 2.0
    _SPARSE_RECONSTRUCTION_COST = 1.0
    _SERIALIZATION_FLOPS_PER_BIT = 1.0

    def _get_dense_forward_flops(
        self,
        device: torch.device,
    ) -> float:
        """Return (and cache) per-sample dense forward FLOPs for the model."""

        if self.net is None:
            return 0.0

        if (
            self._dense_forward_flops_per_sample is not None
            and self._dense_forward_flops_device == device.type
        ):
            return self._dense_forward_flops_per_sample

        first_param = next(self.net.parameters(), None)
        if first_param is None:
            return 0.0

        dtype = first_param.dtype
        sample_input = torch.randn(
            self._FLOP_SAMPLE_SHAPE,
            device=device,
            dtype=dtype,
        )
        flops = estimate_forward_flops(self.net, sample_input, device)
        if flops <= 0.0:
            return 0.0

        self._dense_forward_flops_per_sample = flops
        self._dense_forward_flops_device = device.type
        return flops

    def _estimate_round_flops(
        self,
        num_samples: int,
        num_batches: int,
        batch_size: int,
        epochs: int,
        server_nonzero_count: int,
        client_nonzero_count: int,
        total_param_count: int,
        device: torch.device,
    ) -> float:
        """Estimate the floating-point operations executed during local training.

        The estimate uses a dense forward-pass FLOP profile scaled by the
        effective density of the sparse parameters. The forward cost is inflated
        by ``1 + _BACKWARD_MULTIPLIER`` to approximate the backward pass and an
        additional optimizer cost is included for the active parameters. The
        resulting per-step cost is multiplied by the number of steps per epoch
        and by the number of epochs processed in the round.
        """

        if (
            epochs <= 0
            or batch_size <= 0
            or total_param_count <= 0
            or server_nonzero_count < 0
            or client_nonzero_count < 0
        ):
            return 0.0

        steps_per_epoch = 0
        if num_samples > 0 and batch_size > 0:
            steps_per_epoch = max(math.ceil(num_samples / batch_size), 1)
        elif num_batches > 0:
            steps_per_epoch = num_batches

        if steps_per_epoch <= 0:
            return 0.0

        avg_active_params = (
            float(server_nonzero_count) + float(client_nonzero_count)
        ) / 2.0
        if avg_active_params <= 0.0:
            return 0.0

        dense_forward_flops = self._get_dense_forward_flops(device)
        if dense_forward_flops <= 0.0:
            return 0.0

        density = min(max(avg_active_params / float(total_param_count), 0.0), 1.0)
        sparse_forward_flops = density * dense_forward_flops

        per_step_flops = (
            batch_size * sparse_forward_flops * (1.0 + self._BACKWARD_MULTIPLIER)
            + self._OPTIMIZER_COST * avg_active_params
        )

        return float(epochs * steps_per_epoch * per_step_flops)

    def _estimate_communication_flops(
        self,
        server_nonzero_count: int,
        client_nonzero_count: int,
        total_param_count: int,
        bits_per_parameter: int,
    ) -> tuple[float, float, float]:
        """Estimate client-side compression/decompression FLOPs.

        Even when Flower transport is dense, clients still execute sparse
        bookkeeping operations (scan active entries, build sparse index/value
        views, and reconstruct parameters). This method tracks that workload as
        communication-related FLOPs so it is not lost in round-level metrics.
        """

        if (
            total_param_count <= 0
            or server_nonzero_count < 0
            or client_nonzero_count < 0
            or bits_per_parameter <= 0
        ):
            return 0.0, 0.0, 0.0

        compression_scan_flops = self._SPARSE_SCAN_COST * float(total_param_count)
        compression_sparse_create_flops = (
            self._SPARSE_INDEX_WRITE_COST + self._QUANTIZE_COST
        ) * float(client_nonzero_count)
        compression_flops_clients = (
            compression_scan_flops + compression_sparse_create_flops
        )

        decompression_flops_clients = (
            self._DEQUANTIZE_COST + self._SPARSE_RECONSTRUCTION_COST
        ) * float(server_nonzero_count)

        serialization_flops = (
            self._SERIALIZATION_FLOPS_PER_BIT
            * float(bits_per_parameter)
            * float(total_param_count * 2)
        )

        return (
            float(compression_flops_clients),
            float(decompression_flops_clients),
            float(serialization_flops),
        )

    def _infer_parameter_bitwidth(self, parameters: NDArrays) -> int:
        """Infer a representative floating-point bitwidth for serialization FLOPs.

        Preference is given to floating-point tensors and weighted by tensor
        size (number of elements) to avoid bias from small integer buffers such
        as BatchNorm counters.
        """

        if not parameters:
            return 0

        floating_bitwidth_hist: dict[int, int] = {}
        all_bitwidth_hist: dict[int, int] = {}
        for parameter in parameters:
            if isinstance(parameter, np.ndarray):
                bitwidth = int(parameter.dtype.itemsize) * 8
                if bitwidth <= 0:
                    continue

                numel = int(parameter.size)
                all_bitwidth_hist[bitwidth] = (
                    all_bitwidth_hist.get(bitwidth, 0) + numel
                )
                if np.issubdtype(parameter.dtype, np.floating):
                    floating_bitwidth_hist[bitwidth] = (
                        floating_bitwidth_hist.get(bitwidth, 0) + numel
                    )

        source_hist = (
            floating_bitwidth_hist if floating_bitwidth_hist else all_bitwidth_hist
        )
        if not source_hist:
            return 0

        return max(source_hist, key=source_hist.get)

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
        self._dense_forward_flops_per_sample: float | None = None
        self._dense_forward_flops_device: str | None = None

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

        del parameters

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
            bits_per_parameter = self._infer_parameter_bitwidth(updated_parameters)

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

            epochs = int(config.run_config.get("epochs", 1))
            device = torch.device(config.run_config["device"])
            round_flops = self._estimate_round_flops(
                int(num_samples),
                num_batches,
                int(loader_batch_size),
                epochs,
                server_nonzero_count,
                client_nonzero_count,
                server_total_count,
                device,
            )

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
            (
                compression_flops_clients,
                decompression_flops_clients,
                serialization_flops,
            ) = self._estimate_communication_flops(
                server_nonzero_count,
                client_nonzero_count,
                server_total_count,
                bits_per_parameter,
            )
            metrics["compression_flops_clients"] = compression_flops_clients
            metrics["compression_flops_server"] = 0.0
            metrics["decompression_flops_clients"] = decompression_flops_clients
            metrics["decompression_flops_server"] = 0.0
            metrics["serialization_flops"] = serialization_flops
            metrics["round_flops_compression"] = (
                compression_flops_clients
                + decompression_flops_clients
                + serialization_flops
            )
            metrics["round_flops_decompression"] = decompression_flops_clients

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

            return (
                updated_parameters,
                num_samples,
                metrics,
            )
        finally:
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
