"""MNIST dataset utilities for federated learning."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import ClientDataloaderGen, FedDataloaderGen


# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


class _ArrayDataset(Dataset):
    """Lightweight dataset wrapping array-like data and targets."""

    def __init__(self, data, targets) -> None:
        if len(data) != len(targets):
            raise ValueError(
                "Data and targets must have the same length.",
            )
        self._data = data
        self._targets = targets

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._to_tensor(self._data[index]), self._to_tensor(
            self._targets[index],
        )

    @staticmethod
    def _to_tensor(item) -> torch.Tensor:
        if isinstance(item, torch.Tensor):
            return item
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        if isinstance(item, np.generic):
            return torch.tensor(item.item())
        return torch.as_tensor(item)


def get_dataloader_generators(
    partition_dir: Path,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """

    def get_client_dataloader(
        cid: str | int,
        test: bool,
        _config: dict,
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = torch.load(client_dir / "train.pt", weights_only=False)
        else:
            dataset = torch.load(client_dir / "test.pt", weights_only=False)

        dataset_loader = DataLoader(
            _ArrayDataset(dataset["data"], dataset["targets"]),
            batch_size=config.batch_size,
            shuffle=not test,
        )

        return dataset_loader

    def get_federated_dataloader(
        test: bool,
        _config: dict,
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config

        if not test:
            return DataLoader(
                torch.load(partition_dir / "train.pt", weights_only=False),
                batch_size=config.batch_size,
                shuffle=not test,
            )

        return DataLoader(
            torch.load(partition_dir / "test.pt", weights_only=False),
            batch_size=config.batch_size,
            shuffle=not test,
        )

    return get_client_dataloader, get_federated_dataloader
