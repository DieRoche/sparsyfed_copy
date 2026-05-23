from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
from flwr.common import Parameters

SCHEMA_VERSION = 1
HEADER_FMT = "<I"


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    c = config or {}
    return {
        "tensor_type": str(c.get("tensor_type", "sparse_transport_v1")),
        "csr_min_sparsity": float(c.get("csr_min_sparsity", 0.85)),
        "dense_fallback_if_not_smaller": bool(c.get("dense_fallback_if_not_smaller", True)),
        "index_dtype": str(c.get("index_dtype", "int32")),
        "bitorder": str(c.get("bitorder", "little")),
        "task_sparsity": float(c.get("task_sparsity", 0.0)),
    }


def is_sparse_transport(parameters: Parameters) -> bool:
    return parameters.tensor_type.startswith("sparse_transport")


def _pack(meta: dict[str, Any], buffers: list[bytes]) -> bytes:
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack(HEADER_FMT, len(meta_json)) + meta_json + b"".join(buffers)


def _unpack(payload: bytes) -> tuple[dict[str, Any], memoryview]:
    (hsize,) = struct.unpack(HEADER_FMT, payload[:4])
    meta = json.loads(payload[4 : 4 + hsize].decode("utf-8"))
    return meta, memoryview(payload)[4 + hsize :]


def _finalize_meta(meta: dict[str, Any], buffers: list[bytes]) -> bytes:
    data_nbytes = int(sum(len(b) for b in buffers))
    meta["data_nbytes"] = data_nbytes
    payload = _pack(meta, buffers)
    meta["encoded_nbytes"] = int(len(payload))
    return _pack(meta, buffers)


def _encode_dense(arr: np.ndarray) -> bytes:
    data = np.ascontiguousarray(arr).tobytes(order="C")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "encoding": "dense",
        "original_shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
        "nnz": int(np.count_nonzero(arr)),
        "dense_nbytes": int(arr.nbytes),
        "buffers": [{"name": "data", "nbytes": len(data)}],
    }
    return _finalize_meta(meta, [data])


def _encode_bitmap(arr: np.ndarray, bitorder: str) -> bytes:
    flat = np.ascontiguousarray(arr).ravel(order="C")
    mask = flat != 0
    packed_mask = np.packbits(mask, bitorder=bitorder).tobytes()
    values = np.ascontiguousarray(flat[mask]).tobytes(order="C")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "encoding": "bitmap_values",
        "original_shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
        "nnz": int(mask.sum()),
        "dense_nbytes": int(arr.nbytes),
        "bitorder": bitorder,
        "buffers": [
            {"name": "packed_mask", "nbytes": len(packed_mask)},
            {"name": "values", "nbytes": len(values)},
        ],
    }
    return _finalize_meta(meta, [packed_mask, values])


def _encode_csr(arr: np.ndarray, index_dtype: str) -> bytes:
    matrix = np.ascontiguousarray(arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1))
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols]
    n_rows, n_cols = matrix.shape
    idx_dtype = np.dtype(index_dtype)
    counts = np.bincount(rows, minlength=n_rows)
    crow = np.empty(n_rows + 1, dtype=idx_dtype)
    crow[0] = 0
    crow[1:] = np.cumsum(counts, dtype=idx_dtype)
    col = cols.astype(idx_dtype, copy=False)

    crow_b = np.ascontiguousarray(crow).tobytes(order="C")
    col_b = np.ascontiguousarray(col).tobytes(order="C")
    val_b = np.ascontiguousarray(values).tobytes(order="C")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "encoding": "csr",
        "original_shape": list(arr.shape),
        "csr_shape": [int(n_rows), int(n_cols)],
        "dtype": str(arr.dtype),
        "index_dtype": index_dtype,
        "numel": int(arr.size),
        "nnz": int(values.size),
        "dense_nbytes": int(arr.nbytes),
        "buffers": [
            {"name": "crow_indices", "nbytes": len(crow_b)},
            {"name": "col_indices", "nbytes": len(col_b)},
            {"name": "values", "nbytes": len(val_b)},
        ],
    }
    return _finalize_meta(meta, [crow_b, col_b, val_b])


def _decode_one(payload: bytes) -> np.ndarray:
    meta, rest = _unpack(payload)
    dtype = np.dtype(meta["dtype"])
    enc = meta["encoding"]
    if enc == "dense":
        n = meta["buffers"][0]["nbytes"]
        return np.frombuffer(rest[:n], dtype=dtype).copy().reshape(meta["original_shape"], order="C")
    if enc == "bitmap_values":
        m_n = meta["buffers"][0]["nbytes"]
        v_n = meta["buffers"][1]["nbytes"]
        packed_mask = np.frombuffer(rest[:m_n], dtype=np.uint8)
        values = np.frombuffer(rest[m_n : m_n + v_n], dtype=dtype)
        mask = np.unpackbits(packed_mask, count=meta["numel"], bitorder=meta.get("bitorder", "little")).astype(bool)
        flat = np.zeros(meta["numel"], dtype=dtype)
        flat[mask] = values
        return flat.reshape(meta["original_shape"], order="C")
    if enc == "csr":
        idx_dtype = np.dtype(meta.get("index_dtype", "int32"))
        c_n, col_n, v_n = [b["nbytes"] for b in meta["buffers"]]
        crow = np.frombuffer(rest[:c_n], dtype=idx_dtype)
        col = np.frombuffer(rest[c_n : c_n + col_n], dtype=idx_dtype)
        values = np.frombuffer(rest[c_n + col_n : c_n + col_n + v_n], dtype=dtype)
        n_rows, n_cols = meta["csr_shape"]
        dense = np.zeros((n_rows, n_cols), dtype=dtype)
        for row in range(n_rows):
            start = int(crow[row])
            end = int(crow[row + 1])
            dense[row, col[start:end]] = values[start:end]
        return dense.reshape(meta["original_shape"], order="C")
    raise ValueError(f"Unknown sparse encoding: {enc}")


def encode_parameters(ndarrays: list[np.ndarray], config: dict[str, Any]) -> Parameters:
    c = _cfg(config)
    tensors: list[bytes] = []
    for arr in ndarrays:
        arr = np.asarray(arr)
        if c["task_sparsity"] >= c["csr_min_sparsity"] and arr.ndim >= 2:
            payload = _encode_csr(arr, c["index_dtype"])
        else:
            payload = _encode_bitmap(arr, c["bitorder"])
        if c["dense_fallback_if_not_smaller"] and len(payload) >= arr.nbytes:
            payload = _encode_dense(arr)
        tensors.append(payload)
    return Parameters(tensors=tensors, tensor_type=c["tensor_type"])


def decode_parameters(parameters: Parameters) -> list[np.ndarray]:
    return [_decode_one(tensor) for tensor in parameters.tensors]


def estimate_encoded_size(ndarrays: list[np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    parameters = encode_parameters(ndarrays, config)
    encoding_counts = {"dense": 0, "bitmap_values": 0, "csr": 0}
    per_tensor = []
    for tensor in parameters.tensors:
        meta, _ = _unpack(tensor)
        encoding_counts[meta["encoding"]] += 1
        per_tensor.append(meta)
    return {
        "dense_bytes": int(sum(np.asarray(arr).nbytes for arr in ndarrays)),
        "transport_bytes": int(sum(len(tensor) for tensor in parameters.tensors)),
        "tensor_count": len(parameters.tensors),
        "encoding_counts": encoding_counts,
        "per_tensor": per_tensor,
    }
