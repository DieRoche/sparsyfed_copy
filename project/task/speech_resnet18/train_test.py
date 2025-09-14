"""CIFAR training and testing functions, local and federated."""

from collections.abc import Callable, Sized
from pathlib import Path
from typing import cast

import logging
from logging import ERROR
from flwr.common import log

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)

from torch.nn.utils import prune
from project.task.speech_resnet18.models import (
    get_parameters_to_prune,
)


from flwr.common import NDArrays

from project.client.client import ClientConfig
from project.fed.utils.utils import generic_set_parameters
from project.types.common import (
    FedDataloaderGen,
    FedEvalFN,
    NetGen,
    TestFunc,
)
from project.utils.utils import obtain_device


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float
    final_learning_rate: float  # ? to remove
    curr_round: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=new_sample_rate
    )
    transform.to(config.device)

    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )

            data = transform(data)

            if data.shape[0] != 1:
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output.squeeze(), target)
                final_epoch_per_sample_loss += loss.item()
                num_correct += (
                    (output.squeeze().max(1)[1] == target).clone().detach().sum().item()
                )
                loss.backward()
                optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(
            cast(Sized, trainloader.dataset)
        ),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def fixed_train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    # FLASH
    mask = []
    for param in net.parameters():
        mask.append((param != 0).float())

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=new_sample_rate
    )
    transform.to(config.device)

    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )

            data = transform(data)

            if data.shape[0] != 1:
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output.squeeze(), target)
                final_epoch_per_sample_loss += loss.item()
                num_correct += (
                    (output.squeeze().max(1)[1] == target).clone().detach().sum().item()
                )
                loss.backward()

                # FLASH
                # apply the mask to the gradients
                with torch.no_grad():
                    for param, m in zip(net.parameters(), mask, strict=True):
                        param.grad *= m.to(config.device)

                optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss / len(
            cast(Sized, trainloader.dataset)
        ),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def get_train_and_prune(
    alpha: float = 1.0,
    amount: float = 0.0,
    pruning_method: str = "l1",
    sparsity_inc: float = 0.0,
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":  # ? not working
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        log(logging.DEBUG, "Start training")

        # train the network, with the current parameter
        metrics = train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

        if amount > 0:
            """
            The net must be pruned:
            - at the first round if we are using powerprop
            - every round if we are not using powerprop (alpha=1.0)
            """
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=amount + sparsity_inc,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)
        torch.cuda.empty_cache()
        return metrics

    return train_and_prune


def get_flash_train_and_prune(
    alpha: float = 1.0,
    amount: float = 0.0,
    pruning_method: str = "l1",
    sparsity_inc: float = 0.0,
) -> Callable[[nn.Module, DataLoader, dict, Path], tuple[int, dict]]:
    """Return the training loop with one step pruning at the end.

    Think about moving 'amount' to the config file
    """
    if pruning_method == "base":  # ? not working
        pruning_method = prune.BasePruningMethod
    elif pruning_method == "l1":
        pruning_method = prune.L1Unstructured
    else:
        log(ERROR, f"Pruning method {pruning_method} not recognised, using base")

    def train_and_prune(
        net: nn.Module,
        trainloader: DataLoader,
        _config: dict,
        _working_dir: Path,
    ) -> tuple[int, dict]:
        """Training and pruning process."""
        log(logging.DEBUG, "Start training")

        # FLASH
        if _config["curr_round"] == 1 and _config["warmup"] > 0:
            # temp_net = deepcopy(net) # FOR WARMUP
            log(logging.DEBUG, "First round, warmup training")
            _config["epochs"] = _config["warmup"]

        # train the network, with the current parameter
        metrics = fixed_train(
            net=net,
            trainloader=trainloader,
            _config=_config,
            _working_dir=_working_dir,
        )

        if amount > 0:
            """
            The net must be pruned:
            - at the first round if we are using powerprop
            - every round if we are not using powerprop (alpha=1.0)
            """
            parameters_to_prune = get_parameters_to_prune(net)

            prune.global_unstructured(
                parameters=[
                    (module, tensor_name)
                    for module, tensor_name, _ in parameters_to_prune
                ],
                pruning_method=pruning_method,
                amount=amount + sparsity_inc,
            )
            for module, name, _ in parameters_to_prune:
                prune.remove(module, name)

        return metrics

    return train_and_prune


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member acces and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=new_sample_rate
    )
    transform.to(config.device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )

            images = transform(images)

            outputs = net(images)
            per_sample_loss += criterion(
                outputs.squeeze(),
                labels,
            ).item()
            _, predicted = torch.max(outputs.squeeze().data, 1)
            correct += (predicted == labels).sum().item()

            # del images, labels, outputs, predicted
    torch.cuda.empty_cache()
    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
        },
    )


def get_fed_eval_fn(
    net_generator: NetGen,
    fed_dataloater_generator: FedDataloaderGen,
    test_func: TestFunc,
    _config: dict,
    working_dir: Path,
) -> FedEvalFN | None:
    """Get the federated evaluation function.

    Parameters
    ----------
    net_generator : NetGenerator
        The function to generate the network.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.

    Returns
    -------
    Optional[FedEvalFN]
        The evaluation function for the server
        if the testloader is not empty, else None.
    """
    config: ClientConfig = ClientConfig(**_config)
    del _config

    testloader = fed_dataloater_generator(
        True,
        config.dataloader_config,
    )

    def fed_eval_fn(
        _server_round: int,
        parameters: NDArrays,
        fake_config: dict,
    ) -> tuple[float, dict] | None:
        """Evaluate the model on the given data.

        Parameters
        ----------
        server_round : int
            The current server round.
        parameters : NDArrays
            The parameters of the model to evaluate.
        _config : Dict
            The configuration for the evaluation.

        Returns
        -------
        Optional[Tuple[float, Dict]]
            The loss and the accuracy of the input model on the given data.
        """
        net = net_generator(config.net_config)
        generic_set_parameters(net, parameters)
        config.run_config["device"] = obtain_device()

        if len(cast(Sized, testloader.dataset)) == 0:
            return None

        loss, _num_samples, metrics = test_func(
            net,
            testloader,
            config.run_config,
            working_dir,
        )

        return loss, metrics

    torch.cuda.empty_cache()
    return fed_eval_fn


# Use defaults as they are completely determined
# by the other functions defined in mnist_classification
# get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
