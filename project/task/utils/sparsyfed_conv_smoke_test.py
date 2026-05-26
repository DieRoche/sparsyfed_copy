"""Minimal runtime smoke test for SparsyFed convolution custom autograd."""

import torch

from project.task.utils.sparsyfed_modules import SparsyFedConv2D, SparsyFedConv2DEffnet


def _assert_conv_module_smoke(module: torch.nn.Module, input_tensor: torch.Tensor) -> None:
    module.train()
    output = module(input_tensor)
    loss = output.mean()
    loss.backward()

    assert input_tensor.grad is not None, "Expected gradient on input tensor"
    assert module.weight.grad is not None, "Expected gradient on module weight"
    if getattr(module, "bias", None) is not None:
        assert module.bias.grad is not None, "Expected gradient on module bias"

    for attr in (
        "last_input_density",
        "last_sparse_input_nnz",
        "last_sparse_input_numel",
        "last_sparsification_overhead",
    ):
        assert hasattr(module, attr), f"Missing diagnostic attribute: {attr}"


def run_smoke_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    conv = SparsyFedConv2D(
        alpha=1.0,
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    _assert_conv_module_smoke(conv, x)

    x_effnet = torch.randn(2, 3, 8, 8, requires_grad=True)
    effnet_conv = SparsyFedConv2DEffnet(
        alpha=1.0,
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    _assert_conv_module_smoke(effnet_conv, x_effnet)


if __name__ == "__main__":
    run_smoke_test()
    print("SparsyFed conv smoke test passed.")
