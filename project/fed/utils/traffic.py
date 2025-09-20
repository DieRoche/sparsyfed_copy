"""Utility functions for tracking communication traffic."""

from __future__ import annotations

from flwr.common import Parameters, parameters_to_ndarrays


def parameters_size_bytes(parameters: Parameters | None) -> int:
    """Return the serialized size in bytes for the given parameters."""

    if parameters is None:
        return 0

    if parameters.tensors and isinstance(parameters.tensors[0], (bytes, bytearray)):
        return sum(len(tensor) for tensor in parameters.tensors)

    return sum(arr.nbytes for arr in parameters_to_ndarrays(parameters))
