"""Basic tests for the PCA implementation."""

from __future__ import annotations

import numpy as np

from src.pca import PCA


def make_demo_matrix() -> np.ndarray:
    """Create a small correlated dataset for PCA testing."""

    rng = np.random.default_rng(42)
    base = rng.normal(size=(40, 3))
    feature_4 = 0.6 * base[:, 0] - 0.2 * base[:, 2]
    return np.column_stack([base, feature_4])


def test_fit_stores_expected_shapes() -> None:
    """PCA should store mean and component matrices with consistent shapes."""

    X = make_demo_matrix()
    pca = PCA(n_components=2)
    pca.fit(X)

    assert pca.mean_.shape == (X.shape[1],)
    assert pca.components_.shape == (X.shape[1], 2)
    assert pca.explained_variance_.shape == (2,)


def test_transform_output_shape() -> None:
    """Projected data should have shape (N, k)."""

    X = make_demo_matrix()
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(X)

    assert transformed.shape == (X.shape[0], 2)


def test_inverse_transform_output_shape() -> None:
    """Reconstructed data should return to the original feature dimension."""

    X = make_demo_matrix()
    pca = PCA(n_components=2)
    pca.fit(X)

    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)

    assert X_hat.shape == X.shape


def test_explained_variance_ratio_exists_and_is_reasonable() -> None:
    """Explained variance ratio should be present and bounded."""

    X = make_demo_matrix()
    pca = PCA(n_components=3)
    pca.fit(X)

    assert hasattr(pca, "explained_variance_ratio_")
    assert len(pca.explained_variance_ratio_) == 3
    assert len(pca.all_explained_variance_ratio_) == X.shape[1]
    assert np.all(pca.explained_variance_ratio_ >= 0.0)
    assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-8
