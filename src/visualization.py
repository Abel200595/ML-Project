"""Plotting utilities for image comparison and PCA variance curves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_original_vs_reconstructed(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    class_name: str | None = None,
    display_scale: float = 1.0,
    baseline_image: np.ndarray | None = None,
    baseline_label: str = "JPEG (same ratio)",
    pca_ratio: float | None = None,
    baseline_ratio: float | None = None,
):
    """Create a comparison figure for original, PCA, and optional baseline."""

    # Keep a compact default and allow UI to tune figure size dynamically.
    base_width, base_height = 3.6, 2.2
    figure_scale = float(np.clip(display_scale, 0.5, 2.0))
    n_cols = 3 if baseline_image is not None else 2
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(base_width * n_cols / 2.0 * figure_scale, base_height * figure_scale),
    )
    if n_cols == 2:
        axes = np.asarray(axes)

    is_grayscale = original_image.ndim == 2

    axes[0].imshow(
        original_image,
        cmap="gray" if is_grayscale else None,
        vmin=0.0 if is_grayscale else None,
        vmax=1.0 if is_grayscale else None,
        interpolation="nearest",
    )
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        reconstructed_image,
        cmap="gray" if is_grayscale else None,
        vmin=0.0 if is_grayscale else None,
        vmax=1.0 if is_grayscale else None,
        interpolation="nearest",
    )
    axes[1].set_title("PCA Reconstructed")
    axes[1].axis("off")
    if pca_ratio is not None:
        axes[1].text(
            0.5,
            -0.12,
            f"PCA Compression Ratio: {pca_ratio:.2f}x",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="normal",
        )

    if baseline_image is not None:
        axes[2].imshow(
            baseline_image,
            cmap="gray" if is_grayscale else None,
            vmin=0.0 if is_grayscale else None,
            vmax=1.0 if is_grayscale else None,
            interpolation="nearest",
        )
        axes[2].set_title(baseline_label)
        axes[2].axis("off")
        if baseline_ratio is not None:
            axes[2].text(
                0.5,
                -0.12,
                f"JPEG Compression Ratio: {baseline_ratio:.2f}x",
                transform=axes[2].transAxes,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="normal",
            )

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
