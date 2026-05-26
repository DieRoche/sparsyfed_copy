"""Sparse upload transport encoding/decoding helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

MAGIC = "SPARSYFED_SPARSE_UPLOAD"
FORMAT = "csr_header_arrays_v1"
VERSION = 1


def _dtype_from_name(name: str) -> np.dtype:
    return np.dtype(name)


def _is_float_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


def _bytes_for_mask_values(numel: int, nnz: int, value_dtype: np.dtype) -> int:
    return int((numel + 7) // 8 + nnz * value_dtype.itemsize)




def _estimate_transport_flops(entries: list[dict[str, Any]], payload_arrays: list[np.ndarray]) -> dict[str, float]:
    """Estimate algorithmic transport operation counts from actual payload metadata."""

    compression = 0.0
    decompression = 0.0
    for entry in entries:
        scheme = str(entry.get("scheme", "dense"))
        numel = int(entry.get("numel", 0))
        nnz = int(entry.get("nnz", 0))

        if scheme == "mask_values":
            compression += float(numel + numel + numel + nnz + nnz)
            decompression += float(numel + numel + nnz + nnz)
        elif scheme == "csr":
            matrix_shape = entry.get("matrix_shape", [numel, 1])
            rows = int(matrix_shape[0]) if matrix_shape else numel
            compression += float(numel + rows + nnz + nnz + nnz)
            decompression += float((rows + 1) + rows + nnz + numel)

    payload_bytes = float(sum(int(arr.nbytes) for arr in payload_arrays))
    serialization = payload_bytes * 8.0
    return {
        "compression_flops_clients": compression,
        "compression_flops_server": 0.0,
        "decompression_flops_clients": 0.0,
        "decompression_flops_server": decompression,
        "serialization_flops": serialization,
    }


def encode_sparse_transport(
    arrays: list[np.ndarray],
    cfg: dict,
    curr_round: int,
    target_sparsity: float,
) -> tuple[list[np.ndarray], dict]:
    """Encode model arrays into sparse transport payload and metrics."""

    index_dtype = _dtype_from_name(str(cfg.get("index_dtype", "int32")))
    value_dtype = _dtype_from_name(str(cfg.get("value_dtype", "float32")))
    csr_min_sparsity = float(cfg.get("csr_min_sparsity", 0.9))
    mask_threshold = float(cfg.get("mask_values_below_sparsity", 0.85))
    min_tensor_size = int(cfg.get("min_tensor_size", 1024))
    compress_only_float_tensors = bool(cfg.get("compress_only_float_tensors", True))
    keep_integer_buffers_dense = bool(cfg.get("keep_integer_buffers_dense", True))

    payload_arrays: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []

    dense_bytes = 0
    payload_no_header_bytes = 0
    csr_tensors = 0
    mask_tensors = 0
    dense_tensors = 0
    total_nnz = 0
    total_numel = 0
    low_sparsity_tensors = 0
    low_sparsity_numel = 0

    round_policy = (
        str(cfg.get("first_round_policy", "measured_auto"))
        if curr_round <= 1
        else str(cfg.get("later_round_policy", "config_sparsity"))
    )

    for tensor_index, arr in enumerate(arrays):
        numel = int(arr.size)
        nnz = int(np.count_nonzero(arr))
        sparsity = 1.0 - (float(nnz) / float(numel)) if numel > 0 else 0.0

        dense_bytes += int(arr.nbytes)
        total_nnz += nnz
        total_numel += numel

        force_dense = (
            numel == 0
            or arr.ndim == 0
            or numel < min_tensor_size
            or (compress_only_float_tensors and not _is_float_array(arr))
            or (keep_integer_buffers_dense and np.issubdtype(arr.dtype, np.integer))
        )

        csr_matrix = None
        reshape_policy = None
        if arr.ndim == 2:
            csr_matrix = arr
            reshape_policy = "2d_identity"
        elif arr.ndim == 4:
            csr_matrix = arr.reshape(arr.shape[0], -1)
            reshape_policy = "4d_out_channels_flatten"

        dense_cast = arr
        dense_cost = int(arr.nbytes)
        mask_cost = _bytes_for_mask_values(numel, nnz, value_dtype)
        csr_cost = None
        if csr_matrix is not None:
            rows = int(csr_matrix.shape[0])
            csr_cost = int((rows + 1 + nnz) * index_dtype.itemsize + nnz * value_dtype.itemsize)

        scheme = "dense"
        if not force_dense:
            if curr_round <= 1:
                if sparsity >= csr_min_sparsity and csr_matrix is not None:
                    scheme = "csr"
                elif sparsity < mask_threshold:
                    scheme = "mask_values"
                else:
                    options = [("dense", dense_cost), ("mask_values", mask_cost)]
                    if csr_cost is not None:
                        options.append(("csr", csr_cost))
                    scheme = min(options, key=lambda x: x[1])[0]
            else:
                if target_sparsity >= csr_min_sparsity and csr_matrix is not None:
                    scheme = "csr"
                    if sparsity < csr_min_sparsity:
                        low_sparsity_tensors += 1
                        low_sparsity_numel += numel
                elif target_sparsity < mask_threshold:
                    scheme = "mask_values"
                else:
                    options = [("mask_values", mask_cost), ("dense", dense_cost)]
                    if csr_cost is not None:
                        options.append(("csr", csr_cost))
                    scheme = min(options, key=lambda x: x[1])[0]

        entry: dict[str, Any] = {
            "tensor_index": tensor_index,
            "scheme": scheme,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "numel": numel,
            "nnz": nnz,
            "sparsity": float(sparsity),
        }

        if scheme == "csr" and csr_matrix is not None:
            rows, cols = csr_matrix.shape
            crow = [0]
            col_idx: list[int] = []
            values_list: list[float] = []
            for row_idx in range(rows):
                row = csr_matrix[row_idx]
                nz_cols = np.nonzero(row)[0]
                col_idx.extend(nz_cols.tolist())
                values_list.extend(row[nz_cols].astype(value_dtype, copy=False).tolist())
                crow.append(len(col_idx))

            crow_arr = np.asarray(crow, dtype=index_dtype)
            col_arr = np.asarray(col_idx, dtype=index_dtype)
            val_arr = np.asarray(values_list, dtype=value_dtype)

            start = 1 + len(payload_arrays)
            payload_arrays.extend([crow_arr, col_arr, val_arr])
            payload_no_header_bytes += int(crow_arr.nbytes + col_arr.nbytes + val_arr.nbytes)
            entry["array_indices"] = {
                "crow_indices": start,
                "col_indices": start + 1,
                "values": start + 2,
            }
            entry["reshape_policy"] = reshape_policy
            entry["matrix_shape"] = [int(rows), int(cols)]
            csr_tensors += 1
        elif scheme == "mask_values":
            flat = arr.reshape(-1)
            mask = flat != 0
            packed = np.packbits(mask.astype(np.uint8), bitorder="little")
            values = flat[mask].astype(value_dtype, copy=False)
            start = 1 + len(payload_arrays)
            payload_arrays.extend([packed.astype(np.uint8, copy=False), values])
            payload_no_header_bytes += int(payload_arrays[-2].nbytes + values.nbytes)
            entry["array_indices"] = {"mask_packed": start, "values": start + 1}
            mask_tensors += 1
        else:
            if _is_float_array(arr) and str(cfg.get("value_dtype", "")):
                dense_cast = arr.astype(value_dtype, copy=False)
            dense_arr = dense_cast.copy() if dense_cast is arr else dense_cast
            start = 1 + len(payload_arrays)
            payload_arrays.append(dense_arr)
            payload_no_header_bytes += int(dense_arr.nbytes)
            entry["array_indices"] = {"dense": start}
            dense_tensors += 1

        entries.append(entry)

    header = {
        "magic": MAGIC,
        "version": VERSION,
        "format": FORMAT,
        "num_tensors": len(arrays),
        "entries": entries,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_arr = np.frombuffer(header_bytes, dtype=np.uint8).copy()
    payload_bytes = int(header_arr.nbytes + payload_no_header_bytes)

    transport_flops = _estimate_transport_flops(entries=entries, payload_arrays=[header_arr, *payload_arrays])

    metrics = {
        "upload_dense_bytes": float(dense_bytes),
        "upload_payload_bytes": float(payload_bytes),
        "upload_compression_ratio": float(payload_bytes / dense_bytes) if dense_bytes > 0 else 1.0,
        "upload_transport_enabled": 1,
        "upload_csr_tensors": int(csr_tensors),
        "upload_mask_value_tensors": int(mask_tensors),
        "upload_dense_tensors": int(dense_tensors),
        "upload_total_nnz": int(total_nnz),
        "upload_total_numel": int(total_numel),
        "upload_actual_sparsity": float(1.0 - (float(total_nnz) / float(total_numel))) if total_numel > 0 else 0.0,
        "upload_round_policy": round_policy,
        "upload_sparse_format_version": int(VERSION),
        "upload_csr_expected_but_low_sparsity_tensors": int(low_sparsity_tensors),
        "upload_csr_expected_but_low_sparsity_numel": int(low_sparsity_numel),
    }
    metrics.update(transport_flops)
    return [header_arr, *payload_arrays], metrics


def is_sparse_transport(arrays: list[np.ndarray]) -> bool:
    """Detect sparse transport payload."""
    if not arrays:
        return False
    first = arrays[0]
    if not isinstance(first, np.ndarray) or first.dtype != np.uint8:
        return False
    try:
        header = json.loads(first.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        header.get("magic") == MAGIC
        and int(header.get("version", -1)) == VERSION
        and header.get("format") == FORMAT
    )


def decode_sparse_transport(encoded_arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Decode sparse transport payload into original dense tensor list."""
    if not is_sparse_transport(encoded_arrays):
        raise ValueError("Input does not use sparse transport format")

    header = json.loads(encoded_arrays[0].tobytes().decode("utf-8"))
    decoded: list[np.ndarray] = [None] * int(header["num_tensors"])

    for entry in header["entries"]:
        tensor_index = int(entry["tensor_index"])
        shape = tuple(int(v) for v in entry["shape"])
        original_dtype = np.dtype(entry["dtype"])
        scheme = str(entry["scheme"])

        if scheme == "csr":
            idxs = entry["array_indices"]
            crow = encoded_arrays[int(idxs["crow_indices"])].astype(np.int64, copy=False)
            cols = encoded_arrays[int(idxs["col_indices"])].astype(np.int64, copy=False)
            values = encoded_arrays[int(idxs["values"])]
            matrix_shape = tuple(int(v) for v in entry["matrix_shape"])

            mat = np.zeros(matrix_shape, dtype=values.dtype)
            for row_idx in range(matrix_shape[0]):
                start = int(crow[row_idx])
                end = int(crow[row_idx + 1])
                if end > start:
                    mat[row_idx, cols[start:end]] = values[start:end]

            if entry.get("reshape_policy") == "4d_out_channels_flatten":
                arr = mat.reshape(shape)
            else:
                arr = mat
            decoded[tensor_index] = arr.astype(original_dtype, copy=False)

        elif scheme == "mask_values":
            idxs = entry["array_indices"]
            packed = encoded_arrays[int(idxs["mask_packed"])]
            values = encoded_arrays[int(idxs["values"])]
            numel = int(entry["numel"])
            mask = np.unpackbits(packed, bitorder="little", count=numel).astype(bool)
            flat = np.zeros(numel, dtype=values.dtype)
            flat[mask] = values
            decoded[tensor_index] = flat.reshape(shape).astype(original_dtype, copy=False)

        else:
            dense = encoded_arrays[int(entry["array_indices"]["dense"])]
            decoded[tensor_index] = dense.reshape(shape).astype(original_dtype, copy=False)

    return decoded


def decode_sparse_transport_if_needed(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Decode only when sparse transport header is present."""
    if is_sparse_transport(arrays):
        return decode_sparse_transport(arrays)
    return arrays
