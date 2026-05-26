"""FL-related utility functions for the project."""

import logging
import struct
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path

from numbers import Number

import torch.nn.functional as F

import numpy as np

import torch
from torch.profiler import ProfilerActivity, profile
from flwr.common import (
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch import nn

from project.types.common import ClientGen, NetGen, OnEvaluateConfigFN, OnFitConfigFN


DEFAULT_SUM_METRICS = frozenset(
    {
        "server_to_client_nonzero",
        "client_to_server_nonzero",
        "nonzero_communication_total",
        "round_flops",
        "training_flops",
        "aggregation_flops",
        "fit_round_flops",
        "round_flops_without_evaluation",
        "evaluation_flops",
        "round_flops_compression",
        "compression_flops_clients",
        "compression_flops_server",
        "decompression_flops_clients",
        "decompression_flops_server",
        "serialization_flops",
        "upload_dense_bytes",
        "upload_payload_bytes",
        "upload_csr_tensors",
        "upload_mask_value_tensors",
        "upload_dense_tensors",
        "upload_total_nnz",
        "upload_total_numel",
        "upload_csr_expected_but_low_sparsity_tensors",
        "upload_csr_expected_but_low_sparsity_numel",
    }
)

DEFAULT_AVG_METRICS = frozenset(
    {
        "server_to_client_density",
        "client_to_server_density",
        "learning_rate",
        "upload_compression_ratio",
        "upload_actual_sparsity",
    }
)


def generic_set_parameters(
    net: nn.Module,
    parameters: NDArrays,
    to_copy: bool = True,
) -> None:
    """Set the parameters of a network.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be set.
    parameters : NDArrays
        The parameters to set.
    to_copy : bool (default=False)
        Whether to copy the parameters or use them directly.

    Returns
    -------
    None
    """
    sorted_dict = sorted(net.state_dict().items(), key=lambda x: x[0])  # Sort by keys

    params_dict = zip(
        (keys for keys, _ in sorted_dict),
        parameters,
        strict=False,
    )
    state_dict = OrderedDict(
        {k: torch.tensor(v if not to_copy else v.copy()) for k, v in params_dict},
    )
    net.load_state_dict(state_dict)


def generic_get_parameters(net: nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be returned.

    Returns
    -------
        NDArrays
        The parameters of the network.
    """
    state_dict_items = sorted(
        net.state_dict().items(), key=lambda x: x[0]
    )  # Sort by keys
    parameters = [val.cpu().numpy() for _, val in state_dict_items]

    return parameters


def count_nonzero_elements(weights: NDArrays) -> tuple[int, int]:
    """Return the number of non-zero and total elements in a list of arrays."""

    nonzero = 0
    total = 0
    for layer in weights:
        nonzero += int(np.count_nonzero(layer))
        total += int(layer.size)

    return nonzero, total


def load_parameters_from_file(path: Path) -> Parameters:
    """Load parameters from a binary file.

    Parameters
    ----------
    path : Path
        The path to the parameters file.

    Returns
    -------
    'Parameters
        The parameters.
    """
    byte_data = []
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            while True:
                # Read the length (4 bytes)
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # End of file
                length = struct.unpack("I", length_bytes)[0]

                # Read the data of the specified length
                data = f.read(length)
                byte_data.append(data)

        return Parameters(
            tensors=byte_data,
            tensor_type="numpy.ndarray",
        )

    raise ValueError(f"Unknown parameter format: {path}")


def estimate_forward_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device | str,
) -> float:
    """Estimate the dense forward-pass FLOPs per sample for a model.

    Parameters
    ----------
    model : nn.Module
        The model to profile.
    sample_input : torch.Tensor
        A representative batch of inputs. The returned FLOP count will be
        normalised by the batch dimension to obtain a per-sample value.
    device : torch.device | str
        Device on which to execute the forward pass.

    Returns
    -------
    float
        Estimated forward FLOPs per input sample. Returns ``0.0`` if profiling
        fails or produces no FLOP information.
    """

    if sample_input.ndim == 0:
        return 0.0

    device_obj = torch.device(device)

    was_training = model.training
    model.to(device_obj)
    model.eval()

    sample = sample_input.to(device_obj)

    activities = [ProfilerActivity.CPU]
    if device_obj.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    try:
        with torch.no_grad():
            with profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
            ) as prof:
                model(sample)
        total_flops = 0.0
        for event in prof.key_averages():
            if event.flops is not None:
                total_flops += float(event.flops)
    except Exception as exc:  # pragma: no cover - best effort guard
        log(
            logging.WARNING,
            "Unable to estimate forward FLOPs: %s",  # type: ignore[str-bytes-safe]
            exc,
        )
        total_flops = 0.0
    finally:
        if was_training:
            model.train()

    batch_size = max(int(sample.shape[0]), 1)
    return float(total_flops / batch_size) if total_flops > 0.0 else 0.0


def estimate_module_forward_flops(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> float:
    """Estimate per-batch forward FLOPs for dense and SparsyFed-like layers."""
    if not isinstance(output, torch.Tensor):
        return 0.0
    inp = inputs[0] if len(inputs) > 0 else None

    is_conv_like = isinstance(module, nn.Conv2d) or (
        hasattr(module, "in_channels")
        and hasattr(module, "out_channels")
        and hasattr(module, "kernel_size")
        and hasattr(module, "groups")
    )
    if is_conv_like and output.ndim >= 4 and isinstance(inp, torch.Tensor):
        batch = int(inp.shape[0]) if inp.ndim > 0 else 1
        out_channels = int(getattr(module, "out_channels"))
        out_h = int(output.shape[-2])
        out_w = int(output.shape[-1])
        in_channels = int(getattr(module, "in_channels"))
        groups = max(int(getattr(module, "groups", 1)), 1)
        kernel_size = getattr(module, "kernel_size", (1, 1))
        if isinstance(kernel_size, int):
            k_h = k_w = int(kernel_size)
        else:
            k_h, k_w = int(kernel_size[0]), int(kernel_size[1])
        kernel_ops = k_h * k_w
        macs = batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_ops
        return float(2 * macs)

    is_linear_like = isinstance(module, nn.Linear) or (
        hasattr(module, "in_features") and hasattr(module, "out_features")
    )
    if is_linear_like and isinstance(inp, torch.Tensor):
        batch = int(inp.shape[0]) if inp.ndim > 1 else 1
        return float(2 * batch * int(getattr(module, "in_features")) * int(getattr(module, "out_features")))

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        numel = int(output.numel())
        return float(4 * numel)
    return 0.0


def should_register_flop_hook(module: nn.Module) -> bool:
    """Return whether a module should be included in FLOP hook collection."""
    return bool(
        isinstance(
            module,
            (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        )
        or (
            hasattr(module, "in_channels")
            and hasattr(module, "out_channels")
            and hasattr(module, "kernel_size")
        )
        or (hasattr(module, "in_features") and hasattr(module, "out_features"))
    )


def get_initial_parameters(
    net_generator: NetGen,
    config: dict,
    load_from: Path | None,
    server_round: int | None,
) -> Parameters:
    """Get the initial parameters for the network.

    Parameters
    ----------
    net_generator : NetGen
        The function to generate the network.
    config : Dict
        The configuration.
    load_from : Optional[Path]
        The path to the parameters file.

    Returns
    -------
    'Parameters
        The parameters.
    """
    if load_from is None:
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )
        return ndarrays_to_parameters(
            generic_get_parameters(net_generator(config)),
        )
    try:
        if server_round is not None:
            # Load specific round parameters
            load_from = load_from / f"parameters_{server_round}.bin"
        else:
            # Load only the most recent parameters
            load_from = max(
                Path(load_from).glob("parameters_*.bin"),
                key=lambda f: (
                    int(f.stem.split("_")[1]),
                    int(f.stem.split("_")[2]),
                ),
            )

        log(
            logging.INFO,
            "Loading initial parameters from: %s",
            load_from,
        )

        return load_parameters_from_file(load_from)
    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
        OSError,
        EOFError,
        IsADirectoryError,
    ):
        log(
            logging.INFO,
            f"Loading parameters failed from: {load_from}",
        )
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )

        return ndarrays_to_parameters(
            generic_get_parameters(net_generator(config)),
        )


def get_save_parameters_to_file(
    working_dir: Path,
) -> Callable[[Parameters], None]:
    """Get a function to save parameters to a file.

    Parameters
    ----------
    working_dir : Path
        The working directory.

    Returns
    -------
    Callable[[Parameters], None]
        A function to save parameters to a file.
    """

    def save_parameters_to_file(
        parameters: Parameters,
    ) -> None:
        """Save the parameters to a file.

        Parameters
        ----------
        parameters : Parameters
            The parameters to save.

        Returns
        -------
        None
        """
        parameters_path = working_dir / "parameters"
        parameters_path.mkdir(parents=True, exist_ok=True)
        with open(
            parameters_path / "parameters.bin",
            "wb",
        ) as f:
            # Since Parameters is a list of bytes
            # save the length of each row and the data
            # for deserialization
            for data in parameters.tensors:
                # Prepend the length of the data as a 4-byte integer
                f.write(struct.pack("I", len(data)))
                f.write(data)

    return save_parameters_to_file


def get_weighted_avg_metrics_agg_fn(
    to_agg: set[str],
) -> Callable[[list[tuple[int, dict]]], dict]:
    """Return a function to compute a weighted average over pre-defined metrics.

    Parameters
    ----------
    to_agg : Set[str]
        The metrics to aggregate.

    Returns
    -------
    Callable[[List[Tuple[int, Dict]]], Dict]
        A function to compute a weighted average over pre-defined metrics.
    """

    def weighted_avg(
        metrics: list[tuple[int, dict]],
    ) -> dict:
        """Compute a weighted average over pre-defined metrics.

        Parameters
        ----------
        metrics : List[Tuple[int, Dict]]
            The metrics to aggregate.

        Returns
        -------
        Dict
            The weighted average over pre-defined metrics.
        """
        total_num_examples = sum(
            [num_examples for num_examples, _ in metrics],
        )
        weighted_metrics: dict = defaultdict(float)
        sum_metrics: dict = defaultdict(float)
        min_test_acc: float | None = None
        max_test_acc: float | None = None

        metrics_to_average = set(to_agg) | set(DEFAULT_AVG_METRICS)
        metrics_to_sum = set(DEFAULT_SUM_METRICS)

        for num_examples, metric in metrics:
            test_acc = metric.get("test_accuracy")
            if isinstance(test_acc, (int, float)):
                if min_test_acc is None or test_acc < min_test_acc:
                    min_test_acc = float(test_acc)
                if max_test_acc is None or test_acc > max_test_acc:
                    max_test_acc = float(test_acc)

            for key, value in metric.items():
                if not isinstance(value, Number):
                    continue
                if key in metrics_to_sum:
                    sum_metrics[key] += float(value)
                    continue
                if key in metrics_to_average:
                    weighted_metrics[key] += num_examples * float(value)

        aggregated_metrics: dict[str, float] = {}

        if total_num_examples > 0:
            aggregated_metrics.update(
                {
                    key: value / total_num_examples
                    for key, value in weighted_metrics.items()
                }
            )
        else:
            aggregated_metrics.update({key: 0.0 for key in weighted_metrics})

        aggregated_metrics.update(sum_metrics)

        if min_test_acc is not None and max_test_acc is not None:
            aggregated_metrics["acc_clients_lowest"] = min_test_acc
            aggregated_metrics["acc_clients_highest"] = max_test_acc

        return aggregated_metrics

    return weighted_avg


def test_client(
    test_all_clients: bool,
    test_one_client: bool,
    client_generator: ClientGen,
    initial_parameters: Parameters,
    total_clients: int,
    on_fit_config_fn: OnFitConfigFN | None,
    on_evaluate_config_fn: OnEvaluateConfigFN | None,
) -> None:
    """Debug the client code.

    Avoids the complexity of Ray.
    """
    parameters = parameters_to_ndarrays(initial_parameters)
    if test_all_clients or test_one_client:
        if test_one_client:
            client = client_generator(0)
            _, *res_fit = client.fit(
                parameters,
                on_fit_config_fn(0) if on_fit_config_fn else {},
            )
            res_eval = client.evaluate(
                parameters,
                on_evaluate_config_fn(0) if on_evaluate_config_fn else {},
            )
            log(
                logging.INFO,
                "Fit debug fit: %s  and eval: %s",
                res_fit,
                res_eval,
            )
        else:
            for i in range(total_clients):
                client = client_generator(i)
                _, *res_fit = client.fit(
                    parameters,
                    on_fit_config_fn(i) if on_fit_config_fn else {},
                )
                res_eval = client.evaluate(
                    parameters,
                    on_evaluate_config_fn(i) if on_evaluate_config_fn else {},
                )
                log(
                    logging.INFO,
                    "Fit debug fit: %s  and eval: %s",
                    res_fit,
                    res_eval,
                )


def set_non_value_to(model: nn.Module, value: float) -> None:
    """Set non-value parameters in the model to the specified value."""
    for param in model.parameters():
        param.data[param.data != 0] = value


def sum_recursive(net1: nn.Module, net2: nn.Module) -> nn.Module:
    """Recursively sum all parameters in the model."""
    for p1, p2 in zip(net1.parameters(), net2.parameters(), strict=True):
        p1 = p1.cpu() + p2.cpu()
    return net1


def count_values(model: nn.Module, value: float = 0) -> int:
    """Count the number of parameters in the model with the specified value."""
    count = 0
    for param in model.parameters():
        count += torch.sum(param.data == value).item()
    return count


def net_compare(
    net1: nn.Module, net2: nn.Module, value1: float = 2.0, value2: float = 1.0
) -> dict[str, float]:
    """Count the rate of different parameter between two network."""
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    net1.to(device)
    net2.to(device)

    set_non_value_to(net1, value1)
    set_non_value_to(net2, value2)

    for p1, p2 in zip(net1.parameters(), net2.parameters(), strict=True):
        p1.cpu()
        p2.cpu()
        p1.data = p1.data + p2.data

    # summed = sum_recursive(net1, net2)

    return {
        "activated": count_values(net1, value2),
        "deactivated": count_values(net1, value1),
    }
    # return {
    #     '0': count_values(net1, 0),
    #     '1': count_values(net1, value2),
    #     '2': count_values(net1, value1),
    #     '3': count_values(net1, value1+value2)
    # }


"""def net_compare(net1: nn.Module, net2: nn.Module, msg: str = "") -> dict[str, float]:

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    net1.to(device)
    net2.to(device)

    # Convert the networks to evaluation mode
    net1.eval()
    net2.eval()

    # Initialize counters
    count_0 = 0  # Weight was 0 in both
    count_1 = 0  # Weight non-zero in the second
    count_2 = 0  # Weight non-zero in the first
    count_3 = 0  # Weight non-zero in both

    # Iterate through the parameters of both networks
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        # Convert parameters to numpy arrays
        param1_np = param1.cpu().detach().numpy()
        param2_np = param2.cpu().detach().numpy()

        # Compare individual parameters
        for p1, p2 in zip(param1_np.flatten(), param2_np.flatten()):
            if p1 == 0 and p2 == 0:
                count_0 += 1
            elif p1 == 0:
                count_1 += 1
            elif p2 == 0:
                count_2 += 1
            else:
                count_3 += 1

    # Calculate the total count
    total_count = count_0 + count_1 + count_2 + count_3

    # Return the counts as a dictionary
    return {
        # '0': count_0 / total_count,
        'activate': count_1,
        'deactivate': count_2,
        # '3': count_3 / total_count
    }"""


def nonzeros_tensor(p: torch.tensor) -> tuple[int, int]:
    """Count the rate of non-zero parameter in a tensor."""
    tensor = p.data.cpu().numpy()
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    return nz_count, total_params


def print_nonzeros_tensor(p: torch.tensor, msg: str = "") -> float:
    """Print the count the rate of non-zero parameter in a tensor."""
    nz_count, total_params = nonzeros_tensor(p)
    # log(
    #     logging.INFO,
    #     f"{msg}       nonzeros ="
    #     f" {nz_count:7}/{total_params:7} ({100 * nz_count / total_params:6.2f}%) |"
    #     f" total_pruned = {total_params - nz_count:7} | shape = {p.shape}",
    # )
    return round((nz_count / total_params) * 100, 1)


def get_tensor_sparsity(p: torch.tensor) -> float:
    """Count the rate of non-zero parameter in a tensor."""
    tensor = p.data.cpu().numpy()
    nz_count = np.count_nonzero(tensor)
    total_params = np.prod(tensor.shape)
    return 1 - (nz_count / total_params)


def print_nonzeros(model: nn.Module, msg: str = "") -> float:
    """Print the rate of non-zero parameter in a model."""
    nonzero = total = 0
    for _, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params

    # log(
    #     logging.INFO,
    #     f"{msg}   alive: {nonzero}, pruned : {total - nonzero}, total: {total},"
    #     f" ({100 * (total - nonzero) / total:6.2f}% pruned)",
    # )
    return round(((total - nonzero) / total) * 100, 3)


def get_nonzeros(model: nn.Module) -> float:
    """Return the rate of non-zero parameter in a model."""
    nonzero = total = 0
    for _, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    return round(((total - nonzero) / total) * 100, 3)


def get_layer_sparsity(model: nn.Module) -> list[float]:
    """Count the rate of non-zero parameter in a model."""
    sparsity = []
    for _, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        if (nz_count / total_params) != 1 and (nz_count / total_params) != 0:
            sparsity.append(1 - (nz_count / total_params))
    return sparsity


def print_nonzeros_grad(model: nn.Module, msg: str = "") -> float:
    """Count the rate of non-zero parameter in a model."""
    nonzero = total = 0
    for _, p in model.named_parameters():
        tensor = p.grad.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
    # log(
    #     logging.INFO,
    #     f"{msg}   alive: {nonzero}, pruned : {total - nonzero}, total: {total},"
    #     f" Compression rate : {total / nonzero:10.2f}x "
    #     f" ({100 * (total - nonzero) / total:6.2f}% pruned)",
    # )
    return round((nonzero / total) * 100, 1)


def generate_random_state_dict(
    model: nn.Module, seed: int = 42, sparsity: float = 0.0
) -> OrderedDict:
    """Generate a random, eventually sparse, state dict for a model."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    random_state_dict = OrderedDict()

    # Create random tensors matching the shapes of the model parameters
    for key, data in model.state_dict().items():
        random_tensor = torch.randn(
            data.shape
        )  # Create a random tensor with the same shape
        random_tensor = F.dropout(random_tensor, sparsity)
        random_state_dict[key] = random_tensor

    return random_state_dict
