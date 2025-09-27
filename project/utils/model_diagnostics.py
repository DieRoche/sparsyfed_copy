"""Utilities to surface duplicated tensors and module instances.

These helpers are lightweight diagnostics that can be run before training to
spot common model construction pitfalls such as parameter sharing caused by
in-place tensor reuse or subclasses of ``torch.nn.Module`` that forgot to call
``super().__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, List, Tuple

from torch import Tensor, nn


@dataclass(slots=True)
class ModelIntegrityReport:
    """Summary of potential structural problems detected in a model."""

    duplicate_tensors: List[Tuple[str, str]]
    duplicate_modules: List[Tuple[str, str]]
    super_init_ok: bool

    def has_issues(self) -> bool:
        """Return ``True`` when at least one anomaly has been detected."""

        return bool(self.duplicate_tensors or self.duplicate_modules or not self.super_init_ok)


def _find_duplicate_tensor_storage(items: Iterable[Tuple[str, Tensor]]) -> List[Tuple[str, str]]:
    """Return pairs of tensors that share the same underlying storage."""

    seen: Dict[int, str] = {}
    duplicates: List[Tuple[str, str]] = []
    for name, tensor in items:
        if tensor is None:
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            duplicates.append((name, seen[ptr]))
        else:
            seen[ptr] = name
    return duplicates


def _find_duplicate_modules(module: nn.Module) -> List[Tuple[str, str]]:
    """Return module names that reference the same ``nn.Module`` instance."""

    seen: Dict[int, str] = {}
    duplicates: List[Tuple[str, str]] = []
    for name, child in module.named_modules():
        if name == "":
            continue
        ptr = id(child)
        if ptr in seen:
            duplicates.append((name, seen[ptr]))
        else:
            seen[ptr] = name
    return duplicates


def analyze_model_integrity(module: nn.Module) -> ModelIntegrityReport:
    """Inspect a module for duplicated tensors and improper initialisation."""

    duplicate_tensors = _find_duplicate_tensor_storage(
        chain(module.named_parameters(), module.named_buffers())
    )
    duplicate_modules = _find_duplicate_modules(module)
    super_init_ok = hasattr(module, "_parameters") and isinstance(module._parameters, dict)

    return ModelIntegrityReport(
        duplicate_tensors=duplicate_tensors,
        duplicate_modules=duplicate_modules,
        super_init_ok=super_init_ok,
    )

