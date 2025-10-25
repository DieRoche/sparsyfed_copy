"""Utilities for serialising and deserialising sparse parameter updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from flwr.common import Parameters


_DTYPE_TO_CODE: dict[np.dtype, int] = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
    np.dtype(np.float16): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.int32): 4,
}

_CODE_TO_DTYPE: dict[int, np.dtype] = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _dtype_to_code(dtype: np.dtype) -> int:
    if dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype for sparse communication: {dtype}")
    return _DTYPE_TO_CODE[dtype]


def _code_to_dtype(code: int) -> np.dtype:
    if code not in _CODE_TO_DTYPE:
        raise ValueError(f"Unsupported dtype code in sparse update: {code}")
    return _CODE_TO_DTYPE[code]


@dataclass
class SparseSerialization:
    """Container holding the components of a sparse update serialization."""

    values: np.ndarray
    indices: np.ndarray
    metadata: np.ndarray

    def to_parameters(self) -> Parameters:
        tensors = [
            self.values.astype(np.float32, copy=False).tobytes(),
            self.indices.astype(np.int64, copy=False).tobytes(),
            self.metadata.astype(np.int64, copy=False).tobytes(),
        ]
        return Parameters(tensors=tensors, tensor_type="sparse_update_v1")


def serialise_sparse_update(
    deltas: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    dtypes: Sequence[np.dtype],
    prunable_flags: Sequence[bool],
) -> SparseSerialization:
    """Pack sparse parameter updates into a compact representation."""

    if not (len(deltas) == len(masks) == len(dtypes) == len(prunable_flags)):
        raise ValueError("Delta, mask, dtype and flag lists must be aligned")

    if not deltas:
        empty = np.array([], dtype=np.float32)
        meta = np.zeros((0, 6), dtype=np.int64)
        return SparseSerialization(empty, empty.astype(np.int64), meta)

    values: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    metadata = np.zeros((len(deltas), 6), dtype=np.int64)

    value_offset = 0
    index_offset = 0

    for idx, (delta, mask, dtype, is_prunable) in enumerate(
        zip(deltas, masks, dtypes, prunable_flags, strict=True)
    ):
        flat_delta = delta.reshape(-1)
        flat_mask = mask.reshape(-1)
        if flat_delta.shape != flat_mask.shape:
            raise ValueError("Delta and mask must share the same flattened shape")

        nonzero_idx = np.nonzero(flat_mask)[0].astype(np.int64, copy=False)
        kept_values = flat_delta[flat_mask].astype(np.float32, copy=False)

        values.append(kept_values)
        indices.append(nonzero_idx)

        metadata[idx, 0] = flat_delta.size
        metadata[idx, 1] = nonzero_idx.size
        metadata[idx, 2] = value_offset
        metadata[idx, 3] = index_offset
        metadata[idx, 4] = _dtype_to_code(np.dtype(dtype))
        metadata[idx, 5] = 1 if is_prunable else 0

        value_offset += kept_values.size
        index_offset += nonzero_idx.size

    values_concat = np.concatenate(values) if values else np.array([], dtype=np.float32)
    indices_concat = (
        np.concatenate(indices).astype(np.int64, copy=False)
        if indices
        else np.array([], dtype=np.int64)
    )

    return SparseSerialization(values_concat, indices_concat, metadata)


def deserialise_sparse_update(parameters: Parameters) -> SparseSerialization:
    """Convert a Flower ``Parameters`` payload back to sparse update tensors."""

    if parameters.tensor_type not in {"sparse_update_v1", "numpy.ndarray"}:
        raise ValueError(
            f"Unexpected tensor_type '{parameters.tensor_type}' for sparse update"
        )

    if len(parameters.tensors) != 3:
        raise ValueError("Sparse update payload must contain exactly three tensors")

    values = np.frombuffer(parameters.tensors[0], dtype=np.float32)
    indices = np.frombuffer(parameters.tensors[1], dtype=np.int64)
    metadata_raw = np.frombuffer(parameters.tensors[2], dtype=np.int64)

    if metadata_raw.size % 6 != 0:
        raise ValueError("Sparse update metadata is malformed")

    metadata = metadata_raw.reshape(-1, 6)
    return SparseSerialization(values, indices, metadata)


def reconstruct_dense_update(
    serialization: SparseSerialization,
    shapes: Sequence[tuple[int, ...]],
    dtypes: Sequence[np.dtype],
) -> list[np.ndarray]:
    """Rebuild dense parameter updates using stored shapes and dtypes."""

    if len(shapes) != serialization.metadata.shape[0]:
        raise ValueError("Shape metadata mismatch during sparse reconstruction")
    if len(dtypes) != serialization.metadata.shape[0]:
        raise ValueError("Dtype metadata mismatch during sparse reconstruction")

    updates: list[np.ndarray] = []

    for meta, shape, dtype in zip(
        serialization.metadata, shapes, dtypes, strict=True
    ):
        size, nnz, value_offset, index_offset, dtype_code, _ = meta
        expected_dtype = _code_to_dtype(int(dtype_code))
        # Trust the stored dtype from the metadata when reconstructing
        out_dtype = expected_dtype if expected_dtype == np.dtype(dtype) else np.dtype(dtype)

        flat_update = np.zeros(int(size), dtype=out_dtype)
        if int(nnz) > 0:
            slice_indices = serialization.indices[int(index_offset) : int(index_offset + nnz)]
            slice_values = serialization.values[int(value_offset) : int(value_offset + nnz)]
            flat_update[slice_indices] = slice_values.astype(out_dtype, copy=False)

        updates.append(flat_update.reshape(shape))

    return updates
