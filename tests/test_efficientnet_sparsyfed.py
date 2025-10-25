"""Tests for EfficientNet-B0 SparsyFed integration and sparse communication."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # type: ignore
from torch.utils.data import DataLoader, Dataset

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from project.client.client import Client
from project.fed.server.strategy.fedavgNZ import FedAvgNZ
from project.task.cifar_resnet18.models import (
    get_network_generator_efficientnet_sparsyfed,
)
from project.task.cifar_resnet18.train_test import train
from project.task.utils.sparsyfed_modules import SparsyFedConv2DEffnet


class TinyDataset(Dataset):
    """Small synthetic dataset for smoke testing."""

    def __init__(self, num_samples: int = 8, num_classes: int = 10) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(42)
        self.inputs = torch.randn(num_samples, 3, 32, 32, generator=rng)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=rng)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _make_dataloader_gen(dataset: Dataset) -> callable:
    def _generator(_cid: int | str, _is_eval: bool, dataloader_config: dict) -> DataLoader:
        batch_size = dataloader_config.get("batch_size", 2)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return _generator


def _dummy_fed_loader(_cid: int | str, _is_eval: bool, dataloader_config: dict) -> DataLoader:
    dataset = TinyDataset(num_samples=4)
    batch_size = dataloader_config.get("batch_size", 2)
    return DataLoader(dataset, batch_size=batch_size)


def _make_fit_config(round_idx: int) -> dict:
    return {
        "net_config": {},
        "dataloader_config": {"batch_size": 2},
        "run_config": {
            "device": torch.device("cpu"),
            "epochs": 1,
            "learning_rate": 0.01,
            "final_learning_rate": 0.01,
            "curr_round": round_idx,
            "warmup_rounds": 0,
            "warmup": 0,
            "tot_rounds": 2,
        },
        "extra": {"curr_round": round_idx},
    }


def test_activation_pruning_matches_weight_sparsity() -> None:
    """Activation sparsity should mirror weight sparsity within tolerance."""

    module = SparsyFedConv2DEffnet(
        alpha=1.25,
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=False,
        sparsity=0.6,
    )
    module.train()
    module.set_activation_pruning(True)

    with torch.no_grad():
        weight = module.weight
        mask = torch.rand_like(weight)
        k = int(weight.numel() * 0.6)
        if k > 0:
            _, indices = torch.topk(mask.view(-1), k)
            weight.view(-1)[indices] = 0.0
        weight += torch.randn_like(weight) * 0.1

    input_tensor = torch.randn(2, 8, 12, 12, requires_grad=True)
    output = module(input_tensor)
    loss = output.sum()
    loss.backward()

    proxy = torch.sign(module.weight) * torch.abs(module.weight) ** module.alpha
    nonzero = torch.count_nonzero(proxy).item()
    weight_density = nonzero / proxy.numel()

    threshold = module.in_threshold
    if threshold > 0.0:
        mask = input_tensor.detach().abs() >= threshold
    else:
        mask = input_tensor.detach().abs() > 0.0
    activation_density = mask.float().mean().item()

    assert mask.shape == input_tensor.shape
    assert abs(activation_density - weight_density) <= 5e-3


def test_grad_flow_for_depthwise_and_pointwise_layers() -> None:
    """Depthwise and pointwise convolutions should receive gradients."""

    depthwise = SparsyFedConv2DEffnet(
        alpha=1.25,
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=4,
        bias=False,
        sparsity=0.3,
    )
    pointwise = SparsyFedConv2DEffnet(
        alpha=1.25,
        in_channels=4,
        out_channels=8,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        sparsity=0.3,
    )
    depthwise.train()
    pointwise.train()
    depthwise.set_activation_pruning(True)
    pointwise.set_activation_pruning(True)

    tensor = torch.randn(3, 4, 14, 14, requires_grad=True)
    out = depthwise(tensor)
    out = pointwise(out)
    loss = out.pow(2).mean()
    loss.backward()

    assert depthwise.weight.grad is not None
    assert pointwise.weight.grad is not None
    assert torch.count_nonzero(depthwise.weight.grad) > 0
    assert torch.count_nonzero(pointwise.weight.grad) > 0


def test_depthwise_group_invariants_and_forward() -> None:
    """Depthwise parametrisation must keep grouping semantics intact."""

    base_conv = nn.Conv2d(6, 6, kernel_size=3, padding=1, groups=6, bias=False)
    wrapped = SparsyFedConv2DEffnet.from_conv(
        base_conv,
        alpha=1.25,
        sparsity=0.2,
    )
    wrapped.train()
    wrapped.set_activation_pruning(True)

    assert wrapped.groups == wrapped.in_channels == 6
    assert wrapped.weight.shape[1] == 1

    input_tensor = torch.randn(2, 6, 10, 10, requires_grad=True)
    output = wrapped(input_tensor)
    output.sum().backward()

    assert wrapped.groups == wrapped.in_channels == 6
    assert wrapped.weight.shape[1] == 1


def test_two_round_sparse_federated_smoke() -> None:
    """Two rounds of federated updates should reduce communication and hit sparsity."""

    alpha = 1.25
    target_sparsity = 0.9
    net_gen = get_network_generator_efficientnet_sparsyfed(
        alpha=alpha, sparsity=target_sparsity, num_classes=10
    )
    global_net = net_gen({})
    initial_parameters = [param.copy() for param in generic_get_parameters(global_net)]

    dataset = TinyDataset(num_samples=6)
    dataloader_gen = _make_dataloader_gen(dataset)
    clients = [
        Client(
            cid=i,
            working_dir=Path("."),
            net_generator=net_gen,
            dataloader_gen=dataloader_gen,
            train=train,
            test=train,
            fed_dataloader_gen=_dummy_fed_loader,
        )
        for i in range(2)
    ]

    strategy = FedAvgNZ(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_evaluate_clients=0,
        min_available_clients=2,
        accept_failures=True,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        working_dir=Path("."),
    )
    strategy.initialize_parameters(client_manager=None)  # type: ignore[arg-type]

    # Round 0 - dense communication
    round0_config = _make_fit_config(0)
    round0_results = []
    for client in clients:
        updated, num_samples, metrics = client.fit(initial_parameters, round0_config)
        assert metrics["client_to_server_density"] == 1.0
        params = ndarrays_to_parameters(updated)
        round0_results.append((None, SimpleNamespace(parameters=params, num_examples=num_samples, metrics=metrics)))

    params_round1, metrics_round1 = strategy.aggregate_fit(1, round0_results, [])
    assert metrics_round1["uplink_nonzero_total"] > 0

    # Round 1 - sparse communication
    round1_config = _make_fit_config(1)
    aggregated_arrays = parameters_to_ndarrays(params_round1)
    round1_results = []
    for client in clients:
        updated, num_samples, metrics = client.fit(aggregated_arrays, round1_config)
        assert metrics["client_to_server_density"] < 1.0
        params = ndarrays_to_parameters(updated)
        round1_results.append((None, SimpleNamespace(parameters=params, num_examples=num_samples, metrics=metrics)))

    _, metrics_round2 = strategy.aggregate_fit(2, round1_results, [])

    assert metrics_round2["uplink_nonzero_total"] < metrics_round1["uplink_nonzero_total"]
    assert abs(metrics_round2["global_sparsity"] - target_sparsity) <= 0.05
    assert 0.0 <= metrics_round2["mask_iou"] <= 1.0


def generic_get_parameters(net: nn.Module) -> list[np.ndarray]:
    """Utility mirroring the project implementation for test isolation."""

    return [param.detach().cpu().numpy() for _, param in sorted(net.state_dict().items())]
