"""Utilities for downloading and preparing image datasets."""

from __future__ import annotations

from pathlib import Path
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

STL10_CLASSES: List[str] = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

SUPPORTED_DATASETS: Tuple[str, ...] = ("Fashion-MNIST", "STL10")


def _to_grayscale(images: np.ndarray) -> np.ndarray:
    """Convert RGB images to grayscale and keep grayscale inputs unchanged."""

    if images.ndim == 3:
        return images
    if images.ndim != 4:
        raise ValueError("Expected image array with shape (N, H, W) or (N, H, W, C).")
    if images.shape[-1] == 1:
        return images[..., 0]
    if images.shape[-1] != 3:
        raise ValueError("Only 1-channel or 3-channel images are supported.")
    return 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]


def get_supported_datasets() -> Tuple[str, ...]:
    """Return dataset names available in the UI."""

    return SUPPORTED_DATASETS


def get_class_names(dataset_name: str) -> List[str]:
    """Return class names for a supported dataset."""

    if dataset_name == "Fashion-MNIST":
        return list(FASHION_MNIST_CLASSES)
    if dataset_name == "STL10":
        return list(STL10_CLASSES)
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def get_label_counts(dataset_name: str, root: str = "data", train: bool = True) -> np.ndarray:
    """Return per-label sample counts for a dataset split."""

    if dataset_name == "Fashion-MNIST":
        try:
            from torchvision.datasets import FashionMNIST
        except ImportError as exc:
            raise ImportError(
                "Could not import torchvision (needed for Fashion-MNIST). "
                "Run Streamlit with the project venv so the same Python has torch/torchvision, "
                "e.g. from the repo root: `.venv/bin/streamlit run app.py`, "
                "or `source .venv/bin/activate` then `pip install -r requirements.txt`."
            ) from exc
        dataset = FashionMNIST(root=root, train=train, download=True)
        labels = dataset.targets.numpy().astype(np.int64)
    elif dataset_name == "STL10":
        split = "train" if train else "test"
        _, labels = _load_or_build_stl10_gray_cache(root=root, split=split)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    num_classes = len(get_class_names(dataset_name))
    return np.bincount(labels, minlength=num_classes)


def load_image_dataset(
    dataset_name: str,
    root: str = "data",
    train: bool = True,
    max_samples: int | None = 1000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], List[str]]:
    """Load a supported dataset as a flattened matrix and label vector."""

    if dataset_name == "Fashion-MNIST":
        return load_fashion_mnist(root=root, train=train, max_samples=max_samples, normalize=normalize)
    if dataset_name == "STL10":
        return load_stl10(root=root, train=train, max_samples=max_samples, normalize=normalize)
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from: {', '.join(SUPPORTED_DATASETS)}.")


def load_image_dataset_by_label(
    dataset_name: str,
    label: int,
    root: str = "data",
    train: bool = True,
    max_samples: int | None = 1000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], List[str]]:
    """Load only one label subset as flattened matrix and labels."""

    if dataset_name == "Fashion-MNIST":
        try:
            from torchvision.datasets import FashionMNIST
        except ImportError as exc:
            raise ImportError(
                "Could not import torchvision (needed for Fashion-MNIST). "
                "Run Streamlit with the project venv, e.g. `.venv/bin/streamlit run app.py`, "
                "or activate `.venv` and `pip install -r requirements.txt`."
            ) from exc

        dataset = FashionMNIST(root=root, train=train, download=True)
        images = dataset.data.numpy()
        labels = dataset.targets.numpy().astype(np.int64)
        image_shape: Tuple[int, ...] = (int(images.shape[1]), int(images.shape[2]))
    elif dataset_name == "STL10":
        split = "train" if train else "test"
        images, labels = _load_or_build_stl10_gray_cache(root=root, split=split)
        image_shape = (int(images.shape[1]), int(images.shape[2]))
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    label_indices = np.where(labels == int(label))[0]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer or None.")
        label_indices = label_indices[:max_samples]

    label_images = images[label_indices].astype(np.float64)
    label_values = labels[label_indices]
    X = label_images.reshape(label_images.shape[0], -1)
    if normalize:
        X /= 255.0
    return X, label_values, image_shape, get_class_names(dataset_name)


def load_fashion_mnist(
    root: str = "data",
    train: bool = True,
    max_samples: int | None = 1000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], List[str]]:
    """Download and load Fashion-MNIST as a flattened matrix."""

    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise ImportError(
            "Could not import torchvision (needed for Fashion-MNIST). "
            "Run Streamlit with the project venv, e.g. `.venv/bin/streamlit run app.py`, "
            "or activate `.venv` and `pip install -r requirements.txt`."
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


def _load_or_build_stl10_gray_cache(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached grayscale STL10 arrays or build cache from raw RGB once."""

    root_path = Path(root)
    cache_path = root_path / f"stl10_gray_{split}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["images"], cached["labels"]

    try:
        from torchvision.datasets import STL10
    except ImportError as exc:
        raise ImportError(
            "Could not import torchvision (needed for STL10). "
            "Run Streamlit with the project venv, e.g. `.venv/bin/streamlit run app.py`, "
            "or activate `.venv` and `pip install -r requirements.txt`."
        ) from exc

    dataset = STL10(root=root, split=split, download=True)
    images = np.asarray(dataset.data, dtype=np.float64)  # (N, C, H, W)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    images = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)
    images = _to_grayscale(images)

    root_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        images=images.astype(np.uint8),
        labels=labels.astype(np.int64),
    )
    return images.astype(np.uint8), labels.astype(np.int64)


def load_stl10(
    root: str = "data",
    train: bool = True,
    max_samples: int | None = 1000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], List[str]]:
    """Load STL10 from precomputed grayscale cache."""

    split = "train" if train else "test"
    images, labels = _load_or_build_stl10_gray_cache(root=root, split=split)
    images = images.astype(np.float64)

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be a positive integer or None.")
        images = images[:max_samples]
        labels = labels[:max_samples]

    image_shape = (int(images.shape[1]), int(images.shape[2]))
    X = images.reshape(images.shape[0], -1)
    if normalize:
        X /= 255.0
    return X, labels, image_shape, list(STL10_CLASSES)
