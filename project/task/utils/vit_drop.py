"""
Taken from the original code on [this repository](https://github.com/AamirRaihan/SWAT)
Author: [Md Aamir Raihan](https://github.com/AamirRaihan)
"""

import math

import torch


def drop_structured_filter(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0])))
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1] * tensor_shape[2] * tensor_shape[3],
    )

    new_tensor_shape = tensor.shape
    value = torch.sum(tensor.abs(), 1)
    topk = value.view(-1).abs().topk(k)
    interleaved: torch.Tensor = topk[0][-1]
    index = value.abs() >= (interleaved)
    index = (
        index.repeat_interleave(tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
        .type_as(tensor)
        .reshape(new_tensor_shape)
    )

    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def drop_structured(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0] * tensor_shape[1])))
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1],
        tensor_shape[2] * tensor_shape[3],
    )

    new_tensor_shape = tensor.shape
    value = torch.sum(tensor.abs(), 2)
    topk = value.view(-1).abs().topk(k)
    interleaved: torch.Tensor = topk[0][-1]
    index = value.abs() >= (interleaved)
    index = (
        index.repeat_interleave(tensor_shape[2] * tensor_shape[3])
        .type_as(tensor)
        .reshape(new_tensor_shape)
    )

    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def drop_nhwc_send_th(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(
        math.ceil(
            select_percentage
            * (tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
        )
    )
    topk = tensor.view(-1).abs().topk(k)
    threshold: torch.Tensor = topk[0][-1]
    index = tensor.abs() >= (threshold)
    index = index.type_as(tensor)
    tensor = tensor * index
    return tensor, threshold


def drop_nhwc(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    return drop_nhwc_send_th(
        tensor=tensor,
        select_percentage=select_percentage,
    )[0]


def drop_hwc(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(
        math.ceil(
            select_percentage * (tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
        )
    )
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1] * tensor_shape[2] * tensor_shape[3],
    )
    new_tensor_shape = tensor.shape
    topk = tensor.abs().topk(k)
    threshold: torch.Tensor = topk[0][:, -1]
    interleaved = threshold.repeat_interleave(
        tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
    )
    interleaved = interleaved.reshape(new_tensor_shape)
    index = tensor.abs() >= (interleaved)
    index = index.type_as(tensor)
    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def drop_hw(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[2] * tensor_shape[3])))
    tensor = tensor.reshape(
        tensor_shape[0],
        tensor_shape[1],
        tensor_shape[2] * tensor_shape[3],
    )
    new_tensor_shape = tensor.shape
    topk = tensor.abs().topk(k)
    th: torch.Tensor = topk[0][:, :, -1]
    interleaved = th.repeat_interleave(tensor_shape[2] * tensor_shape[3])
    interleaved = interleaved.reshape(new_tensor_shape)
    index = tensor.abs() >= (interleaved)
    index = index.type_as(tensor)
    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor


def drop_hwn(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    tensor = tensor.permute(1, 0, 2, 3).contiguous()
    tensor = drop_hwc(tensor, select_percentage)
    tensor = tensor.permute(1, 0, 2, 3).contiguous()
    return tensor


def matrix_drop_th(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    """Remove the elements of the input `tensor` using `TopK` algorithm, where `k` is
    computed using the `select_percentage` parameter and the dimensions of the input
    `tensor`. The input `tensor` is assumed to be a matrix, i.e. a tensor of rank 2.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        select_percentage (float): defines the `k` for `TopK`.

    Returns
    -------
        torch.Tensor: output pruned tensor.
        torch.Tensor: value of the threshold used.
    """
    tensor_shape = tensor.shape
    k = int(math.ceil(select_percentage * (tensor_shape[0] * tensor_shape[1])))
    topk = tensor.view(-1).abs().topk(k)
    threshold: torch.Tensor = topk[0][-1]
    index = tensor.abs() >= (threshold)
    index = index.type_as(tensor)
    tensor = tensor * index
    return tensor, threshold


def matrix_drop(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    """Same as `matrix_drop_th`, but returns just the pruned tensor.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        select_percentage (float): defines the `k` for `TopK`.

    Returns
    -------
        torch.Tensor: output pruned tensor.
    """
    return matrix_drop_th(
        tensor=tensor,
        select_percentage=select_percentage,
    )[0]


def drop_threshold(
    tensor: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Remove the elements of the input tensor that are lower than the input threshold.
    The element-wise pruning decision is performed on the absolute value of the element,
    i.e. the element `x` is pruned if `$|x|<t$` where `t` is the threshold.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        threshold (float): threshold used for pruning.

    Returns
    -------
        torch.Tensor: output pruned tensor
    """
    index = tensor.abs() >= threshold
    index = index.type_as(tensor)
    tensor = tensor * index
    return tensor


def drop_random(
    tensor: torch.Tensor,
    select_percentage: float,
) -> torch.Tensor:
    """Set to zero a random subsample of the input `tensor`. The random selection is
    performed using a thresholded uniform distribution between 0 and 1. The
    `select_percentage` parameter defines the threshold.

    Args:
        tensor (torch.Tensor): input tensor to be pruned.
        select_percentage (float): defines the threshold.

    Returns
    -------
        torch.Tensor: output pruned tensor.
    """
    tensor_shape = tensor.shape
    drop_percentage = 1 - select_percentage
    tensor = tensor.flatten()
    index = torch.ones(tensor.shape).type_as(tensor)
    index = index.uniform_() > drop_percentage
    index = index.type_as(tensor)
    tensor = tensor * index
    tensor = tensor.reshape(tensor_shape)
    return tensor
