"""Utilities for downloading and preparing the Fashion-MNIST dataset."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

FASHION_MNIST_CLASSES: List[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist(
    root: str = "data",
    train: bool = True,
    max_samples: int | None = 1000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], List[str]]:
    """Download and load Fashion-MNIST as a flattened matrix.

    Parameters
    ----------
    root : str, default="data"
        Local directory used by ``torchvision`` to store the downloaded data.
    train : bool, default=True
        If ``True``, load the training split. Otherwise load the test split.
    max_samples : int | None, default=1000
        Number of samples to keep for the demo. ``None`` keeps all samples.
    normalize : bool, default=True
        If ``True``, scale pixel values from ``[0, 255]`` to ``[0, 1]``.

    Returns
    -------
    X : np.ndarray of shape (N, d)
        Flattened image matrix. Each sample is one row.
    y : np.ndarray of shape (N,)
        Integer class labels corresponding to each image.
    image_shape : tuple[int, int]
        Original image height and width, typically ``(28, 28)``.
    class_names : list[str]
        Human-readable class names for Fashion-MNIST labels.
    """

    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to download Fashion-MNIST. "
            "Please install dependencies from requirements.txt."
        ) from exc

    dataset = FashionMNIST(root=root, train=train, download=True)
    images = dataset.data.numpy().astype(np.float64)
    labels = dataset.targets.numpy().astype(np.int64)

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer or None.")
        images = images[:max_samples]
        labels = labels[:max_samples]

    image_shape = (int(images.shape[1]), int(images.shape[2]))
    X = images.reshape(images.shape[0], -1)

    if normalize:
        X /= 255.0

    return X, labels, image_shape, list(FASHION_MNIST_CLASSES)
