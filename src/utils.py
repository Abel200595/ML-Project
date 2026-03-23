"""General helper functions for the PCA image compression project."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def reshape_sample(sample: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Reshape a flattened image vector back to its 2D image form.

    Parameters
    ----------
    sample : np.ndarray of shape (d,)
        Flattened image vector.
    image_shape : tuple[int, int]
        Target shape such as ``(28, 28)``.

    Returns
    -------
    np.ndarray of shape image_shape
        Two-dimensional image array.
    """

    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError("reshape_sample expects a 1D flattened image vector.")
    return sample.reshape(image_shape)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error between two arrays of the same shape."""

    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed must have the same shape.")

    return float(np.mean((original - reconstructed) ** 2))


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Convert an image array into a safe display range of ``[0, 1]``.

    This function keeps already-normalized images unchanged and clips any small
    numeric overshoot that may appear after PCA reconstruction.
    """

    image = np.asarray(image, dtype=np.float64)
    return np.clip(image, 0.0, 1.0)
