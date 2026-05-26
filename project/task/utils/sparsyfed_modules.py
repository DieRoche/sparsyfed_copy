"""SparsyFed custom modules for federated learning.

This module contains custom PyTorch modules and functions for implementing SparsyFed
models in a federated learning setting.
"""

from copy import deepcopy
from logging import log
import logging
from typing import Union
from matplotlib import pyplot as plt
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.types import _int, _size
from project.fed.utils.utils import (
    get_tensor_sparsity,
    nonzeros_tensor,
    print_nonzeros_tensor,
)

from project.task.utils.drop import (
    drop_nhwc_send_th,
    drop_structured,
    drop_structured_filter,
    drop_threshold,
    matrix_drop,
)
from project.task.utils.spectral_norm import SpectralNormHandler

torch.autograd.set_detect_anomaly(True)


def convolution_backward(
    ctx,
    grad_output,
):
    sparse_input, sparse_weight, bias = ctx.saved_tensors
    conf = ctx.conf
    input_grad = (
        weight_grad
    ) = (
        bias_grad
    ) = (
        sparsity_grad
    ) = (
        grad_in_th
    ) = grad_wt_th = stride_grad = padding_grad = dilation_grad = groups_grad = None

    # Compute gradient w.r.t. input
    if ctx.needs_input_grad[0]:
        input_grad = conv2d_input(
            sparse_input.shape,
            sparse_weight,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. weight
    if ctx.needs_input_grad[1]:
        weight_grad = conv2d_weight(
            sparse_input,
            sparse_weight.shape,
            grad_output,
            conf["stride"],
            conf["padding"],
            conf["dilation"],
            conf["groups"],
        )

    # Compute gradient w.r.t. bias (works for every Conv2d shape)
    if bias is not None and ctx.needs_input_grad[2]:
        bias_grad = grad_output.sum(dim=(0, 2, 3))

    return (
        input_grad,
        weight_grad,
        bias_grad,
        sparsity_grad,
        grad_in_th,
        grad_wt_th,
        stride_grad,
        padding_grad,
        dilation_grad,
        groups_grad,
    )


class sparsyfed_linear(Function):
    threshold = 1e-7

    @staticmethod
    def forward(ctx, input, weight, bias, sparsity):

        if input.dim() == 2 and bias is not None:
            # The fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias

        topk = max(1 - sparsity, sparsyfed_linear.threshold)

        sparse_input = matrix_drop(input, topk)

        ctx.save_for_backward(sparse_input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, sparse_weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(sparse_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(sparse_input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class SparsyFedLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(
        self,
        alpha: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.3,
    ):
        super(SparsyFedLinear, self).__init__()
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if self.b else None
        self.spectral_norm_handler = SpectralNormHandler()
        self.sparsity = sparsity
        self.last_input_density = 1.0
        self.last_sparse_input_nnz = 0
        self.last_sparse_input_numel = 0
        self.last_sparsification_overhead = 0.0

    def __repr__(self):
        return (
            f"SparsyFedLinear(alpha={self.alpha}, in_features={self.in_features},"
            f" out_features={self.out_features}, bias={self.b},"
            f" sparsity={self.sparsity})"
        )

    def get_weights(self):
        weights = self.weight.detach()
        if self.alpha == 1.0:
            return weights
        elif self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weights)
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def _call_sparsyfed_linear(self, input, weight) -> torch.Tensor:
        if self.training:
            sparsity = get_tensor_sparsity(weight)
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0
        return sparsyfed_linear.apply(input, weight, self.bias, sparsity), sparsity

    def forward(self, input):
        # Apply the re-parametrisation to `self.weight` using `self.alpha`
        if self.alpha == 1.0:
            sparsyfed_weight = self.weight
        elif self.alpha < 0:
            sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(
                self.weight
            )
        else:
            sparsyfed_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        rng_state = torch.random.get_rng_state() if self.training else None
        output, applied_sparsity = self._call_sparsyfed_linear(input, sparsyfed_weight)
        with torch.no_grad():
            if self.training and rng_state is not None:
                torch.random.set_rng_state(rng_state)
            sparse_input = matrix_drop(input, max(1 - float(applied_sparsity), 1e-7))
            nnz = int(torch.count_nonzero(sparse_input).item())
            numel = int(sparse_input.numel())
            self.last_sparse_input_nnz = nnz
            self.last_sparse_input_numel = numel
            self.last_input_density = float(nnz / numel) if numel else 1.0
            self.last_sparsification_overhead = float(2 * numel + nnz)

        return output


class sparsyfed_conv2d(Function):
    threshold = 1e-7

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        sparsity,
        in_threshold,
        stride,
        padding,
        dilation,
        groups,
    ):
        # Ensure input tensor is contiguous
        input = input.contiguous()

        output = F.conv2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        topk = max(1 - sparsity, sparsyfed_conv2d.threshold)

        sparse_input = matrix_drop(input, topk)
        if in_threshold < 0.0:
            sparse_input, in_threshold_tensor = drop_nhwc_send_th(input, topk)
            in_threshold = in_threshold_tensor.item()
        else:
            sparse_input = drop_threshold(input, in_threshold)

        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        ctx.save_for_backward(sparse_input, weight, bias)
        nnz = int(torch.count_nonzero(sparse_input).item())
        numel = int(sparse_input.numel())
        overhead = float(2 * numel + nnz)

        return (
            output,
            in_threshold,
            torch.tensor(nnz, device=input.device, dtype=torch.int64),
            torch.tensor(numel, device=input.device, dtype=torch.int64),
            torch.tensor(overhead, device=input.device, dtype=input.dtype),
        )

    # Use @once_differentiable by default unless we intend to double backward
    @staticmethod
    @once_differentiable
    # def backward(ctx, grad_output, grad_wt_th, grad_in_th):
    def backward(
        ctx,
        grad_output,
        grad_in_th,
        grad_sparse_nnz,
        grad_sparse_numel,
        grad_sparse_overhead,
    ):
        grad_output = grad_output.contiguous()
        return convolution_backward(ctx, grad_output)


class SparsyFedConv2D(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(
        self,
        alpha: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 1,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = False,
        sparsity: float = 0.3,
        pruning_type: str = "unstructured",
        warm_up: int = 0,
        period: int = 1,
    ):
        super(SparsyFedConv2D, self).__init__()
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.b = bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = -1.0
        self.in_threshold = -1.0
        self.epoch = 0
        self.batch_idx = 0
        self.spectral_norm_handler = SpectralNormHandler()
        self.last_input_density = 1.0
        self.last_sparse_input_nnz = 0
        self.last_sparse_input_numel = 0
        self.last_sparsification_overhead = 0.0

    def __repr__(self):
        return (
            f"SparsyFedConv2D(alpha={self.alpha}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, kernel_size={self.kernel_size},"
            f" bias={self.b}, stride={self.stride}, padding={self.padding},"
            f" dilation={self.dilation}, groups={self.groups},"
            f" sparsity={self.sparsity}, pruning_type={self.pruning_type},"
            f" warm_up={self.warmup}, period={self.period})"
        )

    def get_weight(self):
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        if self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_sparsyfed_conv2d(self, input, weight) -> torch.Tensor:

        if self.training:
            # for the activation the sparsity used is proportional to the weight sparsity
            sparsity = get_tensor_sparsity(weight)
        else:
            # Avoid to sparsify during the evaluation
            sparsity = 0.0

        output, in_threshold, sparse_input_nnz, sparse_input_numel, sparsification_overhead = sparsyfed_conv2d.apply(
            input,
            weight,
            self.bias,
            sparsity,
            self.in_threshold,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # update self.in_threshold
        if sparsity != 0.0:
            # otherwise, it is not updated
            self.in_threshold = in_threshold
        self.last_sparse_input_nnz = int(sparse_input_nnz.item())
        self.last_sparse_input_numel = int(sparse_input_numel.item())
        self.last_input_density = (
            float(self.last_sparse_input_nnz / self.last_sparse_input_numel)
            if self.last_sparse_input_numel > 0
            else 1.0
        )
        self.last_sparsification_overhead = float(sparsification_overhead.item())

        return output

    def forward(self, input):
        if self.alpha == 1.0:
            sparsyfed_weight = self.weight
        elif self.alpha < 0:
            sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(
                self.weight
            )
        else:
            sparsyfed_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        return self._call_sparsyfed_conv2d(input, sparsyfed_weight)


class SparsyFedConv2DEffnet(nn.Module):
    """SparsyFed Conv2D variant that keeps EfficientNet depthwise settings intact."""

    def __init__(
        self,
        alpha: float,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[_size, _int] = 3,
        stride: Union[_size, _int] = 1,
        padding: Union[_size, _int] = 0,
        dilation: Union[_size, _int] = 1,
        groups: _int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sparsity: float = 0.3,
        pruning_type: str = "unstructured",
        warm_up: int = 0,
        period: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        self.warmup = warm_up
        self.period = period
        self.wt_threshold = -1.0
        self.in_threshold = -1.0
        self.epoch = 0
        self.batch_idx = 0
        self.spectral_norm_handler = SpectralNormHandler()

        self.inner = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.weight = nn.Parameter(torch.empty_like(self.inner.weight))
        if bias:
            self.bias = nn.Parameter(torch.empty_like(self.inner.bias))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            self.weight.copy_(self.inner.weight)
            if bias and self.bias is not None and self.inner.bias is not None:
                self.bias.copy_(self.inner.bias)

        # Remove the parameters from the inner conv so they are only tracked once.
        self.inner.register_parameter("weight", None)
        if bias:
            self.inner.register_parameter("bias", None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.b = bias
        self.last_input_density = 1.0
        self.last_sparse_input_nnz = 0
        self.last_sparse_input_numel = 0
        self.last_sparsification_overhead = 0.0

    def __repr__(self) -> str:
        return (
            "SparsyFedConv2DEffnet("
            f"alpha={self.alpha}, in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"bias={self.b}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, padding_mode={self.padding_mode}, "
            f"sparsity={self.sparsity}, pruning_type={self.pruning_type}, "
            f"warm_up={self.warmup}, period={self.period})"
        )

    def get_weight(self) -> torch.Tensor:
        weight = self.weight.detach()
        if self.alpha == 1.0:
            return weight
        if self.alpha < 0:
            return self.spectral_norm_handler.compute_weight_update(weight)
        return torch.sign(weight) * torch.pow(torch.abs(weight), self.alpha)

    def _call_sparsyfed_conv2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            sparsity = get_tensor_sparsity(weight)
        else:
            sparsity = 0.0

        output, in_threshold, sparse_input_nnz, sparse_input_numel, sparsification_overhead = sparsyfed_conv2d.apply(
            input,
            weight,
            self.bias,
            sparsity,
            self.in_threshold,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if sparsity != 0.0:
            self.in_threshold = in_threshold
        self.last_sparse_input_nnz = int(sparse_input_nnz.item())
        self.last_sparse_input_numel = int(sparse_input_numel.item())
        self.last_input_density = (
            float(self.last_sparse_input_nnz / self.last_sparse_input_numel)
            if self.last_sparse_input_numel > 0
            else 1.0
        )
        self.last_sparsification_overhead = float(sparsification_overhead.item())

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1.0:
            sparsyfed_weight = self.weight
        elif self.alpha < 0:
            sparsyfed_weight = self.spectral_norm_handler.compute_weight_update(self.weight)
        else:
            sparsyfed_weight = torch.sign(self.weight) * torch.pow(
                torch.abs(self.weight), self.alpha
            )

        return self._call_sparsyfed_conv2d(input, sparsyfed_weight)

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        *,
        alpha: float,
        sparsity: float,
        pruning_type: str = "unstructured",
        warm_up: int = 0,
        period: int = 1,
    ) -> "SparsyFedConv2DEffnet":
        new_module = cls(
            alpha=alpha,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            sparsity=sparsity,
            pruning_type=pruning_type,
            warm_up=warm_up,
            period=period,
        )

        new_module = new_module.to(conv.weight.device, conv.weight.dtype)

        with torch.no_grad():
            new_module.weight.copy_(conv.weight)
            if conv.bias is not None and new_module.bias is not None:
                new_module.bias.copy_(conv.bias)

        new_module.train(conv.training)
        return new_module
