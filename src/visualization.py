"""Plotting utilities for image comparison and PCA variance curves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_original_vs_reconstructed(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    mse: float,
    class_name: str | None = None,
):
    """Create a side-by-side comparison figure for one sample image."""

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))

    axes[0].imshow(
        original_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        reconstructed_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title(f"Reconstructed Image\nMSE = {mse:.6f}")
    axes[1].axis("off")

    if class_name:
        fig.suptitle(f"Sample Class: {class_name}", fontsize=12)

    fig.tight_layout()
    return fig


def plot_explained_variance(
    explained_variance_ratio: np.ndarray,
    selected_k: int | None = None,
    max_points: int = 100,
):
    """Plot explained variance ratio for the leading principal components."""

    explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
    num_points = min(max_points, explained_variance_ratio.shape[0])
    x_values = np.arange(1, num_points + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        x_values,
        explained_variance_ratio[:num_points],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Explained variance ratio",
    )

    if selected_k is not None and 1 <= selected_k <= num_points:
        ax.axvline(selected_k, color="tab:red", linestyle="--", label=f"Selected k = {selected_k}")

    ax.set_title("Explained Variance Ratio")
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_explained_variance(
    cumulative_variance_ratio: np.ndarray,
    selected_k: int | None = None,
    max_points: int = 100,
):
    """Plot cumulative explained variance for the leading components."""

    cumulative_variance_ratio = np.asarray(cumulative_variance_ratio, dtype=np.float64)
    num_points = min(max_points, cumulative_variance_ratio.shape[0])
    x_values = np.arange(1, num_points + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        x_values,
        cumulative_variance_ratio[:num_points],
        marker="o",
        markersize=3,
        linewidth=1.5,
        color="tab:green",
        label="Cumulative explained variance",
    )

    if selected_k is not None and 1 <= selected_k <= num_points:
        ax.axvline(selected_k, color="tab:red", linestyle="--", label=f"Selected k = {selected_k}")

    for threshold in (0.80, 0.90, 0.95):
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1)

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Cumulative Explained Variance")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
