"""General helper functions for the PCA image compression project."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np


def reshape_sample(sample: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a flattened image vector back to image form.

    Parameters
    ----------
    sample : np.ndarray of shape (d,)
        Flattened image vector.
    image_shape : tuple[int, ...]
        Target image shape such as ``(28, 28)`` or ``(32, 32, 3)``.

    Returns
    -------
    np.ndarray of shape image_shape
        Reshaped image array.
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


def compute_pca_compression_ratio(original_dimension: int, n_components: int) -> float:
    """Compute PCA compression ratio as original_dim / compressed_dim."""

    if original_dimension < 1 or n_components < 1:
        raise ValueError("original_dimension and n_components must be positive.")
    return float(original_dimension) / float(n_components)


def compress_with_jpeg_at_ratio(
    image: np.ndarray,
    target_ratio: float,
) -> tuple[np.ndarray, float, int]:
    """JPEG-compress an image to approximately match a target compression ratio.

    Returns
    -------
    reconstructed_image : np.ndarray
        JPEG-decompressed image in [0, 1].
    achieved_ratio : float
        Raw bytes divided by compressed JPEG bytes.
    quality : int
        JPEG quality setting chosen by binary search.
    """

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for JPEG baseline comparison.") from exc

    image = np.asarray(image, dtype=np.float64)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim != 2:
        raise ValueError("Only grayscale images are supported for JPEG baseline.")
    if target_ratio <= 1.0:
        target_ratio = 1.01

    image_u8 = (image * 255.0).round().astype(np.uint8)
    raw_bytes = int(image_u8.size)
    target_jpeg_bytes = max(1, int(np.round(raw_bytes / target_ratio)))

    pil_image = Image.fromarray(image_u8, mode="L")
    best_quality = 95
    best_bytes = None
    best_jpeg_payload = None

    low, high = 1, 95
    while low <= high:
        quality = (low + high) // 2
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        payload = buffer.getvalue()
        payload_size = len(payload)

        best_quality = quality
        best_bytes = payload_size
        best_jpeg_payload = payload

        if payload_size > target_jpeg_bytes:
            high = quality - 1
        else:
            low = quality + 1

    if best_jpeg_payload is None or best_bytes is None:
        raise RuntimeError("Failed to create JPEG baseline image.")

    reconstructed = np.asarray(Image.open(io.BytesIO(best_jpeg_payload)), dtype=np.float64) / 255.0
    achieved_ratio = float(raw_bytes) / float(best_bytes)
    return reconstructed, achieved_ratio, best_quality
