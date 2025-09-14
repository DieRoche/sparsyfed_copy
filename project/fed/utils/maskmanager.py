"""Mask manager for generating, saving, and loading masks for model parameters.

Used for FLASH heterogeneity experiments.
"""

import numpy as np
from pathlib import Path
import pickle


class MaskManager:
    """Mask manager for generating, saving, and loading masks for model parameters."""

    def __init__(
        self, model_shapes: list[tuple], sparsity_levels: list[float], working_dir: Path
    ) -> None:
        """Initialize MaskManager with model layer shapes and desired sparsity levels.

        Args:
            model_shapes: List of tuples representing the shape of each layer
            sparsity_levels: List of floats (0-1) representing desired sparsity levels
            working_dir: Directory to save/load masks
        """
        self.model_shapes = model_shapes
        self.sparsity_levels = sparsity_levels
        self.working_dir = working_dir
        self.masks: dict[float, list[np.ndarray]] = {}

        # Create working directory if it doesn't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def _generate_mask(self, shape: tuple, sparsity: float) -> np.ndarray:
        """Generate a binary mask with given shape and sparsity level."""
        total_elements = np.prod(shape)
        num_zeros = int(total_elements * sparsity)

        # Create a flat array of ones and zeros
        mask_flat = np.ones(total_elements, dtype=bool)
        zero_indices = np.random.choice(total_elements, num_zeros, replace=False)
        mask_flat[zero_indices] = False

        # Reshape to original shape
        return mask_flat.reshape(shape)

    def generate_masks(self) -> None:
        """Generate masks for all sparsity levels."""
        for sparsity in self.sparsity_levels:
            masks = [
                self._generate_mask(shape, sparsity) for shape in self.model_shapes
            ]
            self.masks[sparsity] = masks

    def save_masks(self, use_pickle: bool = True) -> None:
        """Save masks to disk using either pickle or numpy format."""
        for sparsity, masks in self.masks.items():
            if use_pickle:
                filepath = self.working_dir / f"masks_{sparsity:.2f}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(masks, f)
            else:
                # Save as compressed numpy array
                filepath = self.working_dir / f"masks_{sparsity:.2f}.npz"
                np.savez_compressed(filepath, *masks)

    def load_masks(
        self, sparsity: float, use_pickle: bool = True
    ) -> list[np.ndarray] | None:
        """Load masks for a specific sparsity level."""
        if use_pickle:
            filepath = self.working_dir / f"masks_{sparsity:.2f}.pkl"
            if filepath.exists():
                with open(filepath, "rb") as f:
                    return pickle.load(f)
        else:
            filepath = self.working_dir / f"masks_{sparsity:.2f}.npz"
            if filepath.exists():
                npz_file = np.load(filepath)
                return [npz_file[f"arr_{i}"] for i in range(len(npz_file.files))]
        return None

    def apply_masks(
        self, parameters: list[np.ndarray], sparsity: float
    ) -> list[np.ndarray]:
        """Apply masks of given sparsity level to parameters."""
        if sparsity not in self.masks:
            masks = self.load_masks(sparsity)
            if masks is None:
                raise ValueError(f"No masks found for sparsity level {sparsity}")
            self.masks[sparsity] = masks

        return [
            param * mask
            for param, mask in zip(parameters, self.masks[sparsity], strict=True)
        ]
