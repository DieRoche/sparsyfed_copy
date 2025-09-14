"""Functions for CIFAR download and processing."""

import logging
from collections.abc import Sequence, Sized
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from project.task.utils.common import (
    XYList,
    create_lda_partitions,
)

HYDRA_FULL_ERROR = 1


def _download_data(
    dataset_dir: Path,
    num_classes: int = 10,
) -> tuple[CIFAR10, CIFAR10] | tuple[CIFAR100, CIFAR100]:  # ?
    """Download (if necessary) and returns the CIFAR10 dataset.

    Returns
    -------
    Tuple[CIFAR10, CIFAR10]
        The dataset for training and the dataset for testing CIFAR10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if num_classes == 100:  # noqa: PLR2004
        trainset = CIFAR100(
            str(dataset_dir),
            train=True,
            download=True,
            transform=transform,
        )
        testset = CIFAR100(
            str(dataset_dir),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        trainset = CIFAR10(
            str(dataset_dir),
            train=True,
            download=True,
            transform=transform,
        )
        testset = CIFAR10(
            str(dataset_dir),
            train=False,
            download=True,
            transform=transform,
        )
    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(
    trainset: CIFAR10 | CIFAR100,
    testset: CIFAR10 | CIFAR100,
    num_clients: int,
    seed: int,
    iid: bool,
    power_law: bool,
    lda: bool,
    lda_alpha: float,
    balance: bool,
) -> tuple[
    list[Subset] | list[ConcatDataset] | tuple[XYList, np.ndarray], CIFAR10 | CIFAR100
]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed
        by chunks to each client (used to test the convergence in a worst case scenario)
        , by default False
    power_law: bool
        Whether to follow a power-law distribution when assigning number of samples
        for each client, defaults to True
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    seed : int
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[CIFAR10], CIFAR10]
        A list of dataset for each client and a single dataset to be used for testing
        the model.
    """
    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(
        len(cast(Sized, trainset)) / num_clients,
    )
    lengths = [partition_size] * num_clients

    if iid:
        datasets = random_split(
            trainset,
            lengths,
            torch.Generator().manual_seed(seed),
        )
    elif power_law:
        trainset_sorted = _sort_by_class(trainset)
        datasets = _power_law_split(
            trainset_sorted,
            num_partitions=num_clients,
            num_labels_per_partition=2,
            min_data_per_partition=10,
            mean=0.0,
            sigma=2.0,
        )
    elif lda:
        x = torch.from_numpy(np.array([sample[0] for sample in trainset]))
        y = torch.from_numpy(np.array([sample[1] for sample in trainset]))
        # Pack into a tuple
        xy = (x, y)
        datasets = create_lda_partitions(
            dataset=xy,
            num_partitions=num_clients,
            concentration=lda_alpha,
        )

    else:
        shard_size = int(partition_size / 2)
        idxs = trainset.targets.argsort()
        sorted_data = Subset(
            trainset,
            cast(Sequence[int], idxs),
        )
        tmp = []
        for idx in range(num_clients * 2):
            tmp.append(
                Subset(
                    sorted_data,
                    cast(
                        Sequence[int],
                        np.arange(
                            shard_size * idx,
                            shard_size * (idx + 1),
                        ),
                    ),
                ),
            )
        idxs_list = torch.randperm(
            num_clients * 2,
            generator=torch.Generator().manual_seed(seed),
        )
        datasets = [
            ConcatDataset(
                (
                    tmp[idxs_list[2 * i]],
                    tmp[idxs_list[2 * i + 1]],
                ),
            )
            for i in range(num_clients)
        ]

    return datasets, testset


def _balance_classes(
    trainset: CIFAR10,
    seed: int,
) -> CIFAR10:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : CIFAR10
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    CIFAR10
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [
        Subset(
            trainset,
            cast(Sequence[int], idxs[: int(smallest)]),
        ),
    ]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(
                trainset,
                cast(
                    Sequence[int],
                    idxs[int(count) : int(count + smallest)],
                ),
            ),
        )
        tmp_targets.append(
            trainset.targets[idxs[int(count) : int(count + smallest)]],
        )
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled),
        generator=torch.Generator().manual_seed(seed),
    )
    shuffled = cast(
        CIFAR10,
        Subset(
            unshuffled,
            cast(Sequence[int], shuffled_idxs),
        ),
    )
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: CIFAR10,
) -> CIFAR10:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : CIFAR10
        The training dataset that needs to be sorted.

    Returns
    -------
    CIFAR10
        The sorted training dataset.

    ??? changet tensor to Tensto
    """
    class_counts = torch.bincount(torch.tensor(trainset.targets))
    idxs = torch.argsort(
        torch.Tensor(trainset.targets)
    )  # sort targets in ascending order

    tmp = []  # create a subset of the smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in torch.cumsum(class_counts, dim=0):
        tmp.append(
            Subset(
                trainset,
                list(idxs[start : int(count + start)]),
            )
        )  # add rest of classes
        tmp_targets.append(
            torch.Tensor(trainset.targets)[idxs[start : int(count + start)]],
        )
        start += count.item()

    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: CIFAR10,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> list[Subset]:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : CIFAR10
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will belong to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    CIFAR10
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: list[list[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(
        min_data_per_partition / num_labels_per_partition,
    )
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ],
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (
            num_classes,
            int(num_partitions / num_classes),
            num_labels_per_partition,
        ),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(
                probs[cls, u_id // num_classes, cls_idx],
            )

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct partition subsets
    return [Subset(sorted_trainset, p) for p in partitions_idx]


@hydra.main(
    config_path="../../conf",
    config_name="base",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customised (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Check if the partitioning already exists
    if Path.exists(Path(cfg.dataset.partition_dir)):
        log(
            logging.INFO, f"Partitioning already exists at: {cfg.dataset.partition_dir}"
        )
        return

    # Download the dataset
    trainset, testset = _download_data(
        Path(cfg.dataset.dataset_dir),
        cfg.dataset.num_classes,
    )

    # Partition the dataset
    # ideally, the fed_test_set can be composed in three ways:
    # 1. fed_test_set = centralised test set like MNIST
    # 2. fed_test_set = concatenation of all test sets of all clients
    # 3. fed_test_set = test sets of reserved unseen clients
    client_datasets, fed_test_set = _partition_data(
        trainset,
        testset,
        cfg.dataset.num_clients,
        cfg.dataset.seed,
        cfg.dataset.iid,
        cfg.dataset.power_law,
        cfg.dataset.lda,
        cfg.dataset.lda_alpha,
        cfg.dataset.balance,
    )

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralised test set a centrailsed training set would also be possible
    # but is not used here
    torch.save(fed_test_set, partition_dir / "test.pt")

    if cfg.dataset.lda:
        client_trainsets = client_datasets[0]

        # Create the test sets for each client, following the same distribution
        dirichlet_dict = client_datasets[1]
        x = torch.from_numpy(np.array([sample[0] for sample in fed_test_set]))
        y = torch.from_numpy(np.array([sample[1] for sample in fed_test_set]))
        xy = (x, y)
        client_testsets, _ = create_lda_partitions(
            dataset=xy,
            dirichlet_dist=dirichlet_dict,
            num_partitions=cfg.dataset.num_clients,
            concentration=cfg.dataset.lda_alpha,
        )

        for idx, client_dataset in enumerate(client_trainsets):
            log(logging.INFO, len(client_dataset[0]))
            client_dir = partition_dir / f"client_{idx}"
            client_dir.mkdir(parents=True, exist_ok=True)

            # Saving the client trainset
            subset_dict = {"data": client_dataset[0], "targets": client_dataset[1]}
            torch.save(subset_dict, client_dir / "train.pt")

            # Saving the client testset
            subset_dict = {
                "data": client_testsets[idx][0],
                "targets": client_testsets[idx][1],
            }
            torch.save(subset_dict, client_dir / "test.pt")

    else:
        # Save the client datasets
        for idx, client_dataset in enumerate(client_datasets):
            log(logging.INFO, f"client_{idx}: {len(client_dataset)}")
            client_dir = partition_dir / f"client_{idx}"
            client_dir.mkdir(parents=True, exist_ok=True)

            len_val = int(len(client_dataset) / (1 / cfg.dataset.val_ratio))
            lengths = [len(client_dataset) - len_val, len_val]
            ds_train, ds_val = random_split(
                client_dataset,
                lengths,
                torch.Generator().manual_seed(cfg.dataset.seed),
            )

            # Get indices of the subset
            subset_indices = ds_train.indices
            # Access the necessary subset tensors
            ds_train_data = torch.stack([client_dataset[i][0] for i in subset_indices])
            ds_train_targets = torch.tensor([
                client_dataset[i][1] for i in subset_indices
            ])
            # Save the subset tensors in a dictionary format
            subset_dict = {"data": ds_train_data, "targets": ds_train_targets}
            torch.save(subset_dict, client_dir / "train.pt")

            subset_indices = ds_val.indices
            # Access the necessary subset tensors
            ds_val_data = torch.stack([client_dataset[i][0] for i in subset_indices])
            ds_val_targets = torch.tensor([
                client_dataset[i][1] for i in subset_indices
            ])
            # Save the subset tensors in a dictionary format
            subset_dict = {"data": ds_val_data, "targets": ds_val_targets}
            torch.save(subset_dict, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
