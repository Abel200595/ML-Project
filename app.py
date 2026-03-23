"""Streamlit app for exploring PCA-based image compression on Fashion-MNIST."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.data_loader import load_fashion_mnist
from src.pca import PCA
from src.utils import compute_mse, normalize_for_display, reshape_sample
from src.visualization import (
    plot_cumulative_explained_variance,
    plot_explained_variance,
    plot_original_vs_reconstructed,
)

st.set_page_config(
    page_title="Interactive PCA Image Compression Explorer",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_display_dataset(display_split: str, display_samples: int):
    """Load the dataset split used for browsing images in the UI."""

    X, y, image_shape, class_names = load_fashion_mnist(
        root="data",
        train=(display_split == "Train"),
        max_samples=display_samples,
        normalize=True,
    )
    return X, y, image_shape, class_names


@st.cache_resource(show_spinner=False)
def fit_pca_model(fit_samples: int):
    """Fit PCA on a subset of the training split for stable demonstrations."""

    X_train, _, _, _ = load_fashion_mnist(
        root="data",
        train=True,
        max_samples=fit_samples,
        normalize=True,
    )
    pca_model = PCA(n_components=None)
    pca_model.fit(X_train)
    return pca_model


def main() -> None:
    """Run the Streamlit interface."""

    st.title("Interactive PCA Image Compression Explorer")
    st.write(
        "This local demo shows how PCA can compress Fashion-MNIST images into a "
        "low-dimensional representation and reconstruct them back into image form."
    )

    with st.sidebar:
        st.header("Controls")
        display_split = st.selectbox(
            "Display dataset split",
            options=["Train", "Test"],
            index=0,
            help="Choose which split to browse in the image viewer.",
        )
        fit_samples = st.select_slider(
            "Number of training samples used to fit PCA",
            options=[1000, 2000, 5000, 10000],
            value=5000,
            help="PCA is always fitted on the training split. More samples usually produce more stable principal components.",
        )
        display_samples = st.select_slider(
            "Number of samples available to browse",
            options=[500, 1000, 2000, 5000],
            value=1000,
            help="This only affects the sample browser, not the PCA fitting process.",
        )
        st.caption(
            "Note: using too few fitting samples can make principal components less stable "
            "and reduce reconstruction quality."
        )

    try:
        with st.spinner("Loading Fashion-MNIST and fitting PCA on the training split..."):
            X_display, y_display, image_shape, class_names = load_display_dataset(
                display_split=display_split,
                display_samples=display_samples,
            )
            pca_model = fit_pca_model(fit_samples=fit_samples)
    except Exception as error:  # pragma: no cover - UI error branch
        st.error(f"Unable to load data or fit PCA: {error}")
        st.stop()

    max_k = pca_model.n_features_in_

    with st.sidebar:
        sample_index = st.slider(
            "Sample index",
            min_value=0,
            max_value=len(X_display) - 1,
            value=0,
            help="Pick a sample image from the loaded subset.",
        )
        selected_k = st.slider(
            "Number of principal components (k)",
            min_value=1,
            max_value=max_k,
            value=min(50, max_k),
            help="Smaller k means stronger compression but more information loss.",
        )

    selected_sample = X_display[sample_index]
    reconstructed_sample = pca_model.reconstruct(selected_sample, n_components=selected_k)[0]

    # The original image comes directly from the normalized pixel space [0, 1].
    original_image = normalize_for_display(reshape_sample(selected_sample, image_shape))
    reconstructed_image = normalize_for_display(reshape_sample(reconstructed_sample, image_shape))
    reconstruction_mse = compute_mse(selected_sample, reconstructed_sample)
    class_name = class_names[int(y_display[sample_index])]

    st.subheader("Image Comparison")
    st.caption(
        f"PCA is fitted on the training split using the first {fit_samples} samples. "
        f"You are currently browsing the {display_split.lower()} split."
    )

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Selected class", class_name)
    info_col2.metric("Selected k", selected_k)
    info_col3.metric("Reconstruction MSE", f"{reconstruction_mse:.6f}")
    info_col4.metric("Fit sample count", fit_samples)

    comparison_figure = plot_original_vs_reconstructed(
        original_image=original_image,
        reconstructed_image=reconstructed_image,
        mse=reconstruction_mse,
        class_name=class_name,
    )
    st.pyplot(comparison_figure, width="content")

    with st.expander("Debug Information"):
        original_min, original_max = float(np.min(selected_sample)), float(np.max(selected_sample))
        reconstructed_min = float(np.min(reconstructed_sample))
        reconstructed_max = float(np.max(reconstructed_sample))
        st.write(f"Original sample range: [{original_min:.4f}, {original_max:.4f}]")
        st.write(
            f"Reconstructed sample range: [{reconstructed_min:.4f}, {reconstructed_max:.4f}]"
        )
        st.write(f"Image shape: {image_shape}")
        st.write(f"PCA fit sample count: {fit_samples}")
        st.write(f"Display split: {display_split}")
        st.write(f"Display subset size: {len(X_display)}")

    st.subheader("Variance Analysis")
    st.write(
        "The first curve shows how much variance each principal component explains. "
        "The second curve shows how the retained variance accumulates as k increases."
    )

    variance_col1, variance_col2 = st.columns(2)
    with variance_col1:
        explained_variance_figure = plot_explained_variance(
            explained_variance_ratio=pca_model.all_explained_variance_ratio_,
            selected_k=selected_k,
            max_points=min(100, max_k),
        )
        st.pyplot(explained_variance_figure, width="stretch")

    with variance_col2:
        cumulative_variance_figure = plot_cumulative_explained_variance(
            cumulative_variance_ratio=pca_model.cumulative_explained_variance_ratio_,
            selected_k=selected_k,
            max_points=min(100, max_k),
        )
        st.pyplot(cumulative_variance_figure, width="stretch")

    st.subheader("How the Compression Works")
    st.markdown(
        """
        1. The dataset is flattened into a matrix `X` with shape `(N, d)`.
        2. PCA computes the mean image and centers the data.
        3. A covariance matrix is built from the centered data.
        4. Eigen-decomposition finds the principal directions of variation.
        5. Keeping only the first `k` principal components compresses the image.
        6. The image is reconstructed from the compressed representation.
        """
    )


if __name__ == "__main__":
    main()
