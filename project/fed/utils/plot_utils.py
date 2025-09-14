"""Plot utility."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn

# Use TNR for all figures
# to match paper templates
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
] + plt.rcParams["font.serif"]

# Whitegrid is most appropriate
# for scientific papers
sns.set_style("whitegrid")

# An optional colorblind palette
# for figures
CB_color_cycle = [
    "#377EB8",
    "#FF7F00",
    "#4DAF4A",
    "#F781BF",
    "#A65628",
    "#984EA3",
    "#999999",
    "#E41A1C",
    "#DEDE00",
]


def plot_abs_parameter_distribution(net1: nn.Module, net2: nn.Module) -> None:
    """Plot the distribution of absolute parameter values for two networks."""
    abs_params_net1 = []
    abs_params_net2 = []

    # Collect absolute parameter values for net1
    for param in net1.parameters():
        abs_params_net1.extend(torch.abs(param).cpu().detach().numpy().flatten())

    # Collect absolute parameter values for net2
    for param in net2.parameters():
        abs_params_net2.extend(torch.abs(param).cpu().detach().numpy().flatten())

    _, axs = plt.subplots(nrows=2, figsize=(8, 10))

    sns.kdeplot(abs_params_net1, label="Net1", ax=axs[0])
    axs[0].set_title("Normalized Distribution of Absolute Parameter Values - Net1")
    axs[0].set_xlabel("Absolute Parameter Values")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    sns.kdeplot(abs_params_net2, label="Net2", ax=axs[1])
    axs[1].set_title("Normalized Distribution of Absolute Parameter Values - Net2")
    axs[1].set_xlabel("Absolute Parameter Values")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Get integer values and their counts for net1 and net2
    integer_values_net1, counts_net1 = np.unique(
        np.round(abs_params_net1).astype(int), return_counts=True
    )
    integer_values_net2, counts_net2 = np.unique(
        np.round(abs_params_net2).astype(int), return_counts=True
    )

    # Annotate the KDE plots with occurrence of each integer value for both networks
    for values, counts, ax in zip(
        [integer_values_net1, integer_values_net2],
        [counts_net1, counts_net2],
        axs,
        strict=False,
    ):
        for value, count in zip(values, counts, strict=False):
            ax.text(value, 0, str(count), color="black", ha="center")

    plt.tight_layout()
    plt.show()
