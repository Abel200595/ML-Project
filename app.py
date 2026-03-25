"""Streamlit app for exploring PCA-based image compression on image datasets."""

from __future__ import annotations

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from src.data_loader import (
    get_class_names,
    get_label_counts,
    get_supported_datasets,
    load_image_dataset_by_label,
)
from src.pca import PCA
from src.utils import (
    compress_with_jpeg_at_ratio,
    compute_pca_compression_ratio,
    normalize_for_display,
    reshape_sample,
)
from src.visualization import (
    plot_cumulative_explained_variance,
    plot_explained_variance,
    plot_original_vs_reconstructed,
)

st.set_page_config(
    page_title="Interactive PCA Image Compression Explorer",
    layout="wide",
)

try:
    import torchvision  # noqa: F401
except ImportError:
    import sys

    st.error(
        "**torchvision is not installed in the Python that is running Streamlit.**\n\n"
        f"Interpreter: `{sys.executable}`\n\n"
        "Use the project virtualenv from the repo root:\n"
        "- `./run_app.sh`\n"
        "- or `.venv/bin/streamlit run app.py`\n\n"
        "If `.venv` does not exist yet: `python3 -m venv .venv` then "
        "`.venv/bin/pip install -r requirements.txt`."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_display_dataset(
    dataset_name: str,
    display_split: str,
    label: int,
):
    """Load one label subset from the selected split for browsing."""

    X, y, image_shape, class_names = load_image_dataset_by_label(
        dataset_name=dataset_name,
        label=label,
        root="data",
        train=(display_split == "Train"),
        max_samples=None,
        normalize=True,
    )
    return X, y, image_shape, class_names


@st.cache_data(show_spinner=False)
def load_training_dataset_for_pca(
    dataset_name: str,
    label: int,
    fit_samples: int,
    fit_on_train_split: bool,
):
    """Load one class subset from the selected split for PCA fitting."""

    X_train, y_train, _, _ = load_image_dataset_by_label(
        dataset_name=dataset_name,
        label=label,
        root="data",
        train=fit_on_train_split,
        max_samples=fit_samples,
        normalize=True,
    )
    return X_train, y_train


@st.cache_resource(show_spinner=False)
def fit_single_class_pca(
    dataset_name: str,
    fit_samples: int,
    label: int,
    fit_on_train_split: bool,
):
    """Fit one PCA model for a selected class label and cache the result."""

    X_train, y_train = load_training_dataset_for_pca(
        dataset_name=dataset_name,
        label=label,
        fit_samples=fit_samples,
        fit_on_train_split=fit_on_train_split,
    )
    class_samples = X_train
    if class_samples.shape[0] < 2:
        raise ValueError(
            f"Label {int(label)} has only {class_samples.shape[0]} sample(s). "
            "At least two samples are required to fit PCA."
        )
    pca_model = PCA(n_components=None)
    pca_model.fit(class_samples)
    return pca_model


def main() -> None:
    """Run the Streamlit interface."""

    st.title("Interactive PCA Image Compression Explorer")
    st.write(
        "This local demo shows how PCA can compress images from multiple datasets into a "
        "low-dimensional representation and reconstruct them back into image form."
    )

    with st.sidebar:
        st.header("Controls")
        dataset_name = st.selectbox(
            "Dataset",
            options=list(get_supported_datasets()),
            index=0,
            help="Choose a dataset. STL10 provides higher-resolution grayscale images than Fashion-MNIST.",
        )
        display_split = st.selectbox(
            "Display dataset split",
            options=["Train", "Test"],
            index=0,
            help="Choose which split to browse in the image viewer.",
        )

    class_names = get_class_names(dataset_name)
    selected_label = st.sidebar.selectbox(
        "Choose label group",
        options=list(range(len(class_names))),
        format_func=lambda label_id: f"{label_id} - {class_names[label_id]}",
        index=0,
        help="Browse samples from one class and reconstruct them with that class's PCA model.",
    )

    fit_on_train_split = display_split == "Train"
    train_counts = get_label_counts(
        dataset_name=dataset_name,
        root="data",
        train=fit_on_train_split,
    )
    display_counts = get_label_counts(
        dataset_name=dataset_name,
        root="data",
        train=(display_split == "Train"),
    )
    max_fit_samples_for_label = int(train_counts[selected_label])
    max_display_samples_for_label = int(display_counts[selected_label])

    if max_fit_samples_for_label < 2:
        st.error("Selected label has fewer than 2 training samples, cannot fit PCA.")
        st.stop()
    if max_display_samples_for_label < 1:
        st.error("Selected label has no samples in the current display split.")
        st.stop()

    with st.sidebar:
        fit_samples = st.slider(
            "Number of samples used to fit PCA",
            min_value=2,
            max_value=max_fit_samples_for_label,
            value=min(1000, max_fit_samples_for_label),
            help="This is the count within the selected label on the current split.",
        )
        st.caption("Sample count is label-specific, not global.")

    try:
        with st.spinner(
            f"Loading {dataset_name} label {selected_label} ({class_names[selected_label]}) data..."
        ):
            X_display, y_display, image_shape, class_names = load_display_dataset(
                dataset_name=dataset_name,
                display_split=display_split,
                label=selected_label,
            )
    except Exception as error:  # pragma: no cover - UI error branch
        st.error(f"Unable to load data or fit PCA: {error}")
        st.stop()

    try:
        with st.spinner(
            f"Fitting PCA for label {selected_label} ({class_names[selected_label]})..."
        ):
            class_pca_model = fit_single_class_pca(
                dataset_name=dataset_name,
                fit_samples=fit_samples,
                label=selected_label,
                fit_on_train_split=fit_on_train_split,
            )
    except Exception as error:  # pragma: no cover - UI error branch
        st.error(f"Unable to fit PCA for the selected class: {error}")
        st.stop()

    if len(X_display) == 0:
        st.warning(
            "No samples found for the selected label in the current display subset. "
            "Increase display samples or pick another label."
        )
        st.stop()

    max_k = class_pca_model.all_components_.shape[1]

    with st.sidebar:
        sample_index = st.slider(
            "Sample index within selected label",
            min_value=0,
            max_value=len(X_display) - 1,
            value=0,
            help="Pick a sample image from the selected class subset.",
        )
        selected_k = st.slider(
            "Number of principal components (k)",
            min_value=1,
            max_value=max_k,
            value=min(50, max_k),
            help="Smaller k means stronger compression but more information loss.",
        )
        display_scale = st.slider(
            "Image display size",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust image figure size. Lower values make low-resolution images look crisper.",
        )

    selected_sample = X_display[sample_index]
    reconstructed_sample = class_pca_model.reconstruct(selected_sample, n_components=selected_k)[0]

    # The original image comes directly from the normalized pixel space [0, 1].
    original_image = normalize_for_display(reshape_sample(selected_sample, image_shape))
    reconstructed_image = normalize_for_display(reshape_sample(reconstructed_sample, image_shape))
    class_name = class_names[int(y_display[sample_index])]
    pca_ratio = compute_pca_compression_ratio(original_dimension=selected_sample.shape[0], n_components=selected_k)

    jpeg_image = None
    jpeg_ratio = None
    jpeg_quality = None
    try:
        jpeg_image_raw, jpeg_ratio, jpeg_quality = compress_with_jpeg_at_ratio(
            original_image,
            target_ratio=pca_ratio,
        )
        jpeg_image = normalize_for_display(jpeg_image_raw)
    except Exception:
        jpeg_image = None

    st.subheader("Image Comparison")
    st.caption(
        f"Dataset: {dataset_name}. "
        f"PCA is fitted per class label on the {display_split.lower()} split using the first {fit_samples} samples. "
        f"You are browsing label {selected_label} ({class_name}) in the {display_split.lower()} split."
    )

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Selected class", class_name)
    info_col2.metric("Selected k", selected_k)
    info_col3.metric("Fit sample count", fit_samples)

    comparison_figure = plot_original_vs_reconstructed(
        original_image=original_image,
        reconstructed_image=reconstructed_image,
        class_name=class_name,
        display_scale=display_scale,
        baseline_image=jpeg_image,
        baseline_label="JPEG Compressed",
        pca_ratio=pca_ratio,
        baseline_ratio=jpeg_ratio,
    )
    st.pyplot(comparison_figure, width="content", clear_figure=True)
    plt.close(comparison_figure)

    if jpeg_ratio is not None and jpeg_quality is not None:
        ratio_gap = abs(jpeg_ratio - pca_ratio) / max(pca_ratio, 1e-9)
        if jpeg_quality == 1 and ratio_gap > 0.10:
            st.caption(
                f"JPEG reached its lowest quality limit (quality=1). "
                f"Achieved {jpeg_ratio:.2f}x, lower than target {pca_ratio:.2f}x."
            )
        else:
            st.caption(
                f"JPEG baseline compressed with quality {jpeg_quality}, "
                f"achieved ratio {jpeg_ratio:.2f}x (target {pca_ratio:.2f}x)."
            )
    else:
        st.caption(
            "JPEG baseline unavailable in current environment (Pillow may be missing)."
        )

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
        st.write(f"Selected label subset size: {len(X_display)}")

    st.subheader("Variance Analysis")
    st.write(
        "The first curve shows how much variance each principal component explains. "
        "The second curve shows how the retained variance accumulates as k increases."
    )

    variance_col1, variance_col2 = st.columns(2)
    with variance_col1:
        explained_variance_figure = plot_explained_variance(
            explained_variance_ratio=class_pca_model.all_explained_variance_ratio_,
            selected_k=selected_k,
            max_points=min(100, max_k),
        )
        st.pyplot(explained_variance_figure, width="stretch", clear_figure=True)
        plt.close(explained_variance_figure)

    with variance_col2:
        cumulative_variance_figure = plot_cumulative_explained_variance(
            cumulative_variance_ratio=class_pca_model.cumulative_explained_variance_ratio_,
            selected_k=selected_k,
            max_points=min(100, max_k),
        )
        st.pyplot(cumulative_variance_figure, width="stretch", clear_figure=True)
        plt.close(cumulative_variance_figure)

    st.subheader("How the Compression Works")
    st.markdown(
        """
        1. The training set is split into 10 groups by label (`0` to `9`).
        2. Each image is flattened into a vector, producing a matrix `(N_c, d)` per class.
        3. A separate PCA model is fitted for each class group.
        4. You choose a class in the sidebar, then pick `k` principal components.
        5. The selected sample is compressed and reconstructed by its class-specific PCA.
        """
    )


if __name__ == "__main__":
    main()
