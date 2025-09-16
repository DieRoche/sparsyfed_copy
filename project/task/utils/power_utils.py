"""Utility functions for stable power-based reparameterisations."""

from __future__ import annotations

import torch

# Minimum magnitude added before power operations to avoid NaNs during backward
POWER_EPS = 1e-12


def stable_sign_power(
    tensor: torch.Tensor,
    exponent: float | torch.Tensor,
    *,
    min_abs: float = POWER_EPS,
) -> torch.Tensor:
    """Apply a sign-preserving power while keeping the backward pass stable.

    Args:
        tensor: The input tensor that will be reparameterised.
        exponent: The exponent to apply to the magnitude of ``tensor``.
        min_abs: Minimum absolute value used before the power operation. A small
            positive value prevents derivatives from exploding when the tensor
            contains zeros and the exponent is smaller than one.

    Returns:
        A tensor where the original sign of ``tensor`` is preserved and the
        magnitude has been safely raised to ``exponent``.
    """

    abs_tensor = torch.abs(tensor)
    if min_abs is not None:
        # ``clamp`` keeps values above ``min_abs`` untouched while ensuring
        # strictly positive magnitudes for very small entries.
        abs_tensor = torch.clamp(abs_tensor, min=min_abs)
    else:
        abs_tensor = abs_tensor + POWER_EPS

    return torch.sign(tensor) * torch.pow(abs_tensor, exponent)


__all__ = ["POWER_EPS", "stable_sign_power"]
