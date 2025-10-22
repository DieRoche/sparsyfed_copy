"""Define our models, and training and eval functions."""

from copy import deepcopy
from collections.abc import Callable, Iterable
import logging
from flwr.common.logger import log

import numpy as np

import torch

from torch import nn

from project.task.utils.sparsyfed_modules import (
    SparsyFedConv2D,
    SparsyFedConv2DEffnet,
    SparsyFedLinear,
)
from project.task.utils.sparsyfed_no_act_modules import (
    SparsyFed_no_act_Conv1D,
    SparsyFed_no_act_Conv2D,
    SparsyFed_no_act_linear,
)

from project.task.cifar_resnet18.efficientnet import EfficientNetB0_CIFAR

from project.task.utils.spectral_norm import SpectralNormHandler
from project.task.utils.swat_modules import SWATConv2D as ZeroflSwatConv2D
from project.task.utils.swat_modules import SWATLinear as ZeroflSwatLinear

def _make_norm_layer(num_features: int, use_group_norm: bool) -> nn.Module:
    """Return the normalization layer used by the custom ResNet blocks."""

    if use_group_norm:
        return nn.GroupNorm(2, num_features)
    return nn.BatchNorm2d(num_features)


class BasicBlock(nn.Module):
    """Basic residual block matching the custom ResNet-18 definition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = _make_norm_layer(out_channels, use_group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = _make_norm_layer(out_channels, use_group_norm)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _make_norm_layer(out_channels, use_group_norm),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the residual block."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NetCifarResnet18(nn.Module):
    """A ResNet18 adapted to CIFAR10."""

    def __init__(
        self, num_classes: int, device: str = "cuda", groupnorm: bool = False
    ) -> None:
        """Initialize network."""
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.use_group_norm = groupnorm
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        self.bn1 = _make_norm_layer(64, self.use_group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for current_stride in strides:
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    current_stride,
                    use_group_norm=self.use_group_norm,
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# get_resnet18: NetGen = lazy_config_wrapper(NetCifarResnet18)


def init_weights(module: nn.Module) -> None:
    """Initialise standard and custom layers in the input module."""
    if isinstance(
        module,
        SparsyFed_no_act_linear
        | SparsyFed_no_act_Conv2D
        | SparsyFed_no_act_Conv1D
        | SparsyFedLinear
        | SparsyFedConv2D
        | SparsyFedConv2DEffnet
        | ZeroflSwatLinear
        | ZeroflSwatConv2D
        | nn.Linear
        | nn.Conv2d
        | nn.Conv1d,
    ):
        weight = getattr(module, "weight", None)
        if weight is None:
            return

        fan_in = calculate_fan_in(weight.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978

        std = np.sqrt(1.0 / fan_in) / distribution_stddev
        a, b = -2.0 * std, 2.0 * std

        u = nn.init.trunc_normal_(weight.data, std=std, a=a, b=b)
        if (
            isinstance(
                module,
                SparsyFed_no_act_linear
                | SparsyFed_no_act_Conv2D
                | SparsyFedLinear
                | SparsyFedConv2D
                | SparsyFedConv2DEffnet,
            )
            and module.alpha > 1
        ):
            u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        weight.data = u
        if module.bias is not None:
            module.bias.data.zero_()


def calculate_fan_in(tensor: torch.Tensor) -> float:
    """Calculate fan in.

    Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    """
    min_fan_in = 2
    dimensions = tensor.dim()
    if dimensions < min_fan_in:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > min_fan_in:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return float(fan_in)


def replace_layer_with_swat(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
    pruning_type: str = "unstructured",
    first_layer: bool = True,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            if first_layer:
                first_layer = False
                continue
            new_conv = ZeroflSwatConv2D(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
                sparsity=sparsity,
                pruning_type=pruning_type,
                warm_up=0,
                period=1,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            if first_layer:
                first_layer = False
                continue
            new_conv = ZeroflSwatLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_conv)

    for model, immediate_child_module in module.named_children():
        replace_layer_with_swat(
            immediate_child_module, model, alpha, sparsity, first_layer=first_layer
        )


def get_network_generator_resnet_zerofl(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
    pruning_type: str = "unstructured",
) -> Callable[[dict], NetCifarResnet18]:
    """Swat network generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)

    replace_layer_with_swat(
        module=untrained_net,
        name="NetCifarResnet18",
        alpha=alpha,
        sparsity=sparsity,
        pruning_type=pruning_type,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        """Initialize the weights of the layers."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> NetCifarResnet18:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


def replace_layer_with_sparsyfed(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
    pruning_type: str = "unstructured",
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the SWAT versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = SparsyFedConv2D(
                alpha=alpha,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
                sparsity=sparsity,
                pruning_type=pruning_type,
                warm_up=0,
                period=1,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = SparsyFedLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_conv)

    for model, immediate_child_module in module.named_children():
        replace_layer_with_sparsyfed(immediate_child_module, model, alpha, sparsity)


def replace_layer_with_sparsyfed_effnet(
    module: nn.Module,
    name: str = "Model",
    alpha: float = 1.0,
    sparsity: float = 0.0,
    pruning_type: str = "unstructured",
) -> None:
    """Replace EfficientNet Conv2d layers with SparsyFed equivalents preserving groups."""

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, SparsyFedConv2DEffnet):
            # Already wrapped, avoid descending into the inner Conv2d again.
            continue
        if isinstance(target_attr, nn.Conv2d):
            new_conv = SparsyFedConv2DEffnet.from_conv(
                target_attr,
                alpha=alpha,
                sparsity=sparsity,
                pruning_type=pruning_type,
            )
            setattr(module, attr_str, new_conv)
        elif isinstance(target_attr, nn.Linear):
            new_linear = SparsyFedLinear(
                alpha=alpha,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
                sparsity=sparsity,
            )
            setattr(module, attr_str, new_linear)

    for child_name, immediate_child_module in module.named_children():
        if isinstance(immediate_child_module, SparsyFedConv2DEffnet):
            # Skip recursion into wrapped convolutions to prevent infinite wrapping.
            continue
        replace_layer_with_sparsyfed_effnet(
            immediate_child_module, child_name, alpha, sparsity, pruning_type
        )


def get_network_generator_resnet_sparsyfed(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
    pruning_type: str = "unstructured",
) -> Callable[[dict], NetCifarResnet18]:
    """Swat network generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)

    replace_layer_with_sparsyfed(
        module=untrained_net,
        name="NetCifarResnet18",
        alpha=alpha,
        sparsity=sparsity,
        pruning_type=pruning_type,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        """Initialize the weights of the layers."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> NetCifarResnet18:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


def replace_layer_with_sparsyfed_no_act(
    module: nn.Module,
    name: str = "Model",  # ? Never used. Give some problem
    alpha: float = 1.0,
    sparsity: float = 0.0,
) -> None:
    """Replace every nn.Conv2d and nn.Linear layers with the PowerProp versions."""
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Conv2d:
            new_conv = SparsyFed_no_act_Conv2D(
                alpha=alpha,
                sparsity=sparsity,
                in_channels=target_attr.in_channels,
                out_channels=target_attr.out_channels,
                kernel_size=target_attr.kernel_size[0],
                bias=target_attr.bias is not None,
                padding=target_attr.padding,
                stride=target_attr.stride,
            )
            setattr(module, attr_str, new_conv)
        if type(target_attr) == nn.Linear:
            new_conv = SparsyFed_no_act_linear(
                alpha=alpha,
                sparsity=sparsity,
                in_features=target_attr.in_features,
                out_features=target_attr.out_features,
                bias=target_attr.bias is not None,
            )
            setattr(module, attr_str, new_conv)

    # ? for name, immediate_child_module in module.named_children(): # Previus version
    for model, immediate_child_module in module.named_children():
        replace_layer_with_sparsyfed_no_act(
            immediate_child_module, model, alpha, sparsity
        )


def get_network_generator_resnet_sparsyfed_no_act(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 10,
) -> Callable[[dict], NetCifarResnet18]:
    """Powerprop Resnet generator."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)

    replace_layer_with_sparsyfed_no_act(
        module=untrained_net,
        name="NetCifarResnet18",
        alpha=alpha,
        sparsity=sparsity,
    )

    def init_model(
        module: nn.Module,
    ) -> None:
        """Initialize the weights of the layers."""
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict) -> NetCifarResnet18:
        """Return a deep copy of the untrained network."""
        return deepcopy(untrained_net)

    return generated_net


def get_efficientnet_b0(
    num_classes: int = 100,
) -> Callable[[dict], EfficientNetB0_CIFAR]:
    """EfficientNet-B0 network generator configured for CIFAR-sized inputs."""

    untrained_net = EfficientNetB0_CIFAR(
        num_classes=num_classes,
    )

    def generated_net(config: dict) -> EfficientNetB0_CIFAR:
        """Return a freshly initialized EfficientNet-B0 instance."""

        if config is None:
            config = {}
        elif not isinstance(config, dict):
            try:
                config = dict(config)
            except TypeError:
                config = {}
        sanitized_config: dict[str, int] = {}
        num_classes_override = config.get("num_classes")
        if num_classes_override is not None:
            sanitized_config["num_classes"] = num_classes_override

        requested_num_classes = sanitized_config.get("num_classes", num_classes)
        if requested_num_classes is None:
            requested_num_classes = num_classes
        if (
            requested_num_classes == num_classes
            and not sanitized_config
        ):
            # Fast-path when no overrides are provided
            return deepcopy(untrained_net)
        return EfficientNetB0_CIFAR(
            num_classes=requested_num_classes,
        )

    return generated_net


def get_network_generator_efficientnet_sparsyfed(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 100,
    pruning_type: str = "unstructured",
) -> Callable[[dict], EfficientNetB0_CIFAR]:
    """Create a SparsyFed EfficientNet-B0 generator."""

    untrained_net = EfficientNetB0_CIFAR(
        num_classes=num_classes,
    )

    replace_layer_with_sparsyfed_effnet(
        module=untrained_net,
        name="EfficientNetB0_CIFAR",
        alpha=alpha,
        sparsity=sparsity,
        pruning_type=pruning_type,
    )

    def init_model(module: nn.Module) -> None:
        """Initialize the weights of the layers."""

        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> EfficientNetB0_CIFAR:
        """Return a deep copy of the untrained network."""

        return deepcopy(untrained_net)

    return generated_net


def get_network_generator_efficientnet_sparsyfed_no_act(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 100,
) -> Callable[[dict], EfficientNetB0_CIFAR]:
    """Create a SparsyFed (no activation) EfficientNet-B0 generator."""

    untrained_net = EfficientNetB0_CIFAR(
        num_classes=num_classes,
    )

    replace_layer_with_sparsyfed_no_act(
        module=untrained_net,
        name="EfficientNetB0_CIFAR",
        alpha=alpha,
        sparsity=sparsity,
    )

    def init_model(module: nn.Module) -> None:
        """Initialize the weights of the layers."""

        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> EfficientNetB0_CIFAR:
        """Return a deep copy of the untrained network."""

        return deepcopy(untrained_net)

    return generated_net


def get_network_generator_efficientnet_zerofl(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 100,
    pruning_type: str = "unstructured",
) -> Callable[[dict], EfficientNetB0_CIFAR]:
    """Create a ZeroFL EfficientNet-B0 generator."""

    untrained_net = EfficientNetB0_CIFAR(
        num_classes=num_classes,
    )

    replace_layer_with_swat(
        module=untrained_net,
        name="EfficientNetB0_CIFAR",
        alpha=alpha,
        sparsity=sparsity,
        pruning_type=pruning_type,
        first_layer=True,
    )

    def init_model(module: nn.Module) -> None:
        """Initialize the weights of the layers."""

        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    def generated_net(_config: dict | None) -> EfficientNetB0_CIFAR:
        """Return a deep copy of the untrained network."""

        return deepcopy(untrained_net)

    return generated_net


def get_resnet18(num_classes: int = 10) -> Callable[[dict], NetCifarResnet18]:
    """Cifar Resnet18 network generatror."""
    untrained_net: NetCifarResnet18 = NetCifarResnet18(num_classes=num_classes)
    # untrained_net.load_state_dict(
    #     generate_random_state_dict(untrained_net, seed=42, sparsity=0.9)
    # )

    def generated_net(_config: dict) -> NetCifarResnet18:
        return deepcopy(untrained_net)

    def init_model(
        module: nn.Module,
    ) -> None:
        init_weights(module)
        for _, immediate_child_module in module.named_children():
            init_model(immediate_child_module)

    init_model(untrained_net)

    return generated_net


def get_parameters_to_prune(
    net: nn.Module,
    _first_layer: bool = False,
) -> Iterable[tuple[nn.Module, str, str]]:
    """Pruning.

    Return an iterable of tuples containing the SparsyFed_no_act_Conv2D and
    SparsyFed_no_act_Conv1D layers in the input model.
    """
    parameters_to_prune = []
    first_layer = _first_layer

    def add_immediate_child(
        module: nn.Module,
        name: str,
    ) -> None:
        nonlocal first_layer
        if (
            type(module) == SparsyFed_no_act_Conv2D
            or type(module) == SparsyFed_no_act_Conv1D
            or type(module) == SparsyFed_no_act_linear
            or type(module) == SparsyFedConv2D
            or type(module) == SparsyFedConv2DEffnet
            or type(module) == SparsyFedLinear
            or type(module) == ZeroflSwatConv2D
            or type(module) == ZeroflSwatLinear
            or type(module) == nn.Conv2d
            or type(module) == nn.Conv1d
            or type(module) == nn.Linear
        ):
            if first_layer:
                first_layer = False
            else:
                parameters_to_prune.append((module, "weight", name))

        for _name, immediate_child_module in module.named_children():
            add_immediate_child(immediate_child_module, _name)

    add_immediate_child(net, "Net")

    return parameters_to_prune


def set_spectral_global_exponent(net: nn.Module, apply: bool = False) -> float:
    """Compute the average spectral exponent across all layers.

    Then set this value as the alpha for all custom layers in the network.

    Parameters
    ----------
    net : nn.Module
        The neural network module to process

    Returns
    -------
    float
        The average spectral exponent that was computed and set for all layers
    """
    spectral_handler = SpectralNormHandler()
    exponents: list[float] = []

    # First pass: compute exponents for all layers
    def compute_layer_exponent(module: nn.Module) -> None:
        """Compute the spectral exponent for a single layer."""
        if isinstance(
            module,
            nn.Linear
            | nn.Conv2d
            | SparsyFedLinear
            | SparsyFedConv2D
            | SparsyFedConv2DEffnet
            | SparsyFed_no_act_linear
            | SparsyFed_no_act_Conv2D,
        ):
            # Get the weight tensor
            weight = module.weight.data

            # Compute the normalized weight using spectral norm
            weight_normalized = spectral_handler._compute_spectral_norm(weight)

            # Compute average of non-zero normalized weights
            weight_normalized_avg = torch.mean(
                weight_normalized[weight_normalized != 0]
            ).item()

            # Compute exponent for this layer
            exponent = 1 + weight_normalized_avg
            exponent = round(exponent, 4)
            # check if exponent is nan or all the weights are zero
            if exponent is None or len(weight_normalized[weight_normalized != 0]) == 0:
                exponent = 1.0
            exponents.append(exponent)

    # Apply first pass to collect all exponents
    net.apply(compute_layer_exponent)

    # Compute average exponent
    if not exponents:
        log(
            logging.INFO, "No applicable layers found for spectral exponent computation"
        )

        return 1.0  # Default value if no layers processed

    avg_exponent = sum(exponents) / len(exponents)
    # avg_exponent = max(exponents)

    # Second pass: set the average exponent for all custom layers
    def set_layer_exponent(module: nn.Module) -> None:
        """Set the computed average exponent for a layer."""
        if isinstance(
            module,
            SparsyFedLinear
            | SparsyFedConv2D
            | SparsyFedConv2DEffnet
            | SparsyFed_no_act_linear
            | SparsyFed_no_act_Conv2D,
        ) and hasattr(module, "alpha"):
            module.alpha = avg_exponent

    # Apply second pass to set the average exponent
    if apply:
        net.apply(set_layer_exponent)
        log(
            logging.INFO,
            f"Average spectral exponent applied to layers. Exponent: {avg_exponent}",
        )
    else:
        log(
            logging.INFO,
            "Average spectral exponent NOT applied to layers. Exponent:"
            f" {avg_exponent}",
        )

    return avg_exponent


def prevent_layer_collapse(
    dense_net: nn.Module, sparse_net: nn.Module, amount: float = 0.01
) -> nn.Module:
    """Prevent layer collapse.

    This is done copying the weights from the dense network to the sparse network.

    The function does:
        - check if some layer of the sparse model has collapsed,
            i.e. all the weights are zero
        - if a layer has collapsed, it copies the top-k of the weights (amount)
            from the dense model to the sparse model
        - return the sparse model with the copied weights
    """
    for dense_layer, sparse_layer in zip(
        dense_net.modules(), sparse_net.modules(), strict=True
    ):
        if (
            isinstance(
                sparse_layer,
                nn.Conv2d
                | nn.Linear
                | SparsyFedLinear
                | SparsyFedConv2D
                | SparsyFedConv2DEffnet,
            )
            and torch.sum(sparse_layer.weight.data != 0) == 0
        ):
            # log(logging.WARNING, f"Layer collapsed: {sparse_layer}")
            k = int(amount * dense_layer.weight.data.numel())
            topk_indices = torch.topk(
                dense_layer.weight.data.abs().flatten(), k
            ).indices
            sparse_layer.weight.data.flatten()[
                topk_indices
            ] = dense_layer.weight.data.flatten()[topk_indices]
            if sparse_layer.bias is not None:
                sparse_layer.bias.data = dense_layer.bias.data.clone()
    return sparse_net
