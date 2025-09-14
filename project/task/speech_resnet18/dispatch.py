"""Dispatch the functionality of the task to project.main.

The dispatch functions are used to dynamically select
the correct functions from the task
based on the hydra config file.
You need to write dispatch functions for three categories:
    - train/test and fed test functions
    - net generator and dataloader generator functions
    - fit/eval config functions

The top-level project.dipatch module operates as a pipeline
and selects the first function which does not return None.
Do not throw any errors based on not finding
a given attribute in the configs under any circumstances.
If you cannot match the config file,
return None and the dispatch of the next task
in the chain specified by project.dispatch will be used.
"""

from pathlib import Path

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.speech_resnet18.dataset import get_dataloader_generators
from project.task.speech_resnet18.models import (
    get_resnet18,
    get_powerprop_resnet18,
    get_powerswat_resnet18,
    get_zerofl_resnet18,
)
from project.task.speech_resnet18.train_test import (
    get_fed_eval_fn,
    get_flash_train_and_prune,
    test,
    get_train_and_prune,
)
from project.types.common import DataStructure, TrainStructure


def dispatch_train(
    cfg: DictConfig,
) -> TrainStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[TrainStructure]
        The train function, test function and the get_fed_eval_fn function.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with None default
    train_structure: str | None = cfg.get("task", {}).get(
        "train_structure",
        None,
    )

    # Only consider not None and uppercase matches
    if train_structure is not None and train_structure.upper() == "SPEECH_RN18_PRUNE":
        alpha = cfg.get("task", {}).get(
            "alpha",
            1.0,
        )
        sparsity = cfg.get("task", {}).get(
            "sparsity",
            0.0,
        )
        # ZeroFL: the sparsity must be modified to include the mask
        mask = cfg.get("task", {}).get("mask", 0.0)
        sparsity = sparsity - mask
        return (
            # train,
            get_train_and_prune(alpha=alpha, amount=sparsity, pruning_method="l1"),
            test,
            get_fed_eval_fn,
        )
    elif train_structure is not None and train_structure.upper() == "SPEECH_RN18_FIX":
        alpha = cfg.get("task", {}).get(
            "alpha",
            1.0,
        )
        sparsity = cfg.get("task", {}).get(
            "sparsity",
            0.0,
        )
        return (
            # train,
            get_flash_train_and_prune(
                alpha=alpha, amount=sparsity, pruning_method="l1"
            ),
            test,
            get_fed_eval_fn,
        )

    # Cannot match, send to next dispatch in chain
    return None


def dispatch_data(cfg: DictConfig) -> DataStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[DataStructure]
        The net generator, client dataloader generator and fed dataloader generator.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with {} default at nested dicts
    # and None default at the final key
    client_model_and_data: str | None = cfg.get(
        "task",
        {},
    ).get("model_and_data", None)

    # Select the partition dir
    # if it does not exist data cannot be loaded
    # for MNIST and the dispatch should return None
    partition_dir: str | None = cfg.get("dataset", {}).get(
        "partition_dir",
        None,
    )

    # Only consider situations where both are not None
    # otherwise data loading would failr later
    if client_model_and_data is not None and partition_dir is not None:
        # Obtain the dataloader generators
        # for the provided partition dir
        (
            client_dataloader_gen,
            fed_dataloader_gen,
        ) = get_dataloader_generators(
            Path(partition_dir),
        )

        # Case insensitive matches
        # Case insensitive matches
        alpha: float = cfg.get("task", {}).get("alpha", 1.0)
        sparsity: float = cfg.get("task", {}).get("sparsity", 0.0)
        num_classes: int = cfg.get("dataset", {}).get(
            "num_classes",
            35,
        )
        if client_model_and_data.upper() == "SPEECH_RN18":
            return (
                get_resnet18(num_classes=num_classes),
                client_dataloader_gen,
                fed_dataloader_gen,
            )
        if client_model_and_data.upper() == "SPEECH_SPARSYFED_NA_RN18":
            return (
                get_powerprop_resnet18(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )
        if client_model_and_data.upper() == "SPEECH_SPARSYFED_RN18":
            return (
                get_powerswat_resnet18(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )
        if client_model_and_data.upper() == "SPEECH_ZEROFL_RN18":
            return (
                get_zerofl_resnet18(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

    # Cannot match, send to next dispatch in chain
    return None


dispatch_config = dispatch_default_config
