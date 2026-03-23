"""PCA implementation from scratch for image compression demos."""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis implemented with eigen-decomposition.

    This class follows the standard PCA pipeline used in many textbooks:

    1. Compute the feature-wise mean of the data matrix ``X``.
    2. Center the data by subtracting the mean.
    3. Compute the covariance matrix ``C = (1 / N) * X_centered.T @ X_centered``.
    4. Perform eigen-decomposition on the covariance matrix.
    5. Sort eigenvalues/eigenvectors from large to small.
    6. Keep the top ``k`` eigenvectors as principal components.
    7. Project samples to the low-dimensional space.
    8. Reconstruct samples from the projected representation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (d,)
        Mean value of each feature/pixel across all training samples.
    components_ : np.ndarray of shape (d, k)
        The top ``k`` principal directions kept for compression.
    explained_variance_ : np.ndarray of shape (k,)
        Variance captured by each selected principal component.
    explained_variance_ratio_ : np.ndarray of shape (k,)
        Fraction of total variance explained by each selected component.
    all_components_ : np.ndarray of shape (d, d)
        All principal directions sorted by importance. This is useful for
        plotting curves and interactively changing ``k`` in the Streamlit app.
    all_explained_variance_ratio_ : np.ndarray of shape (d,)
        Explained variance ratio for all available components.
    cumulative_explained_variance_ratio_ : np.ndarray of shape (d,)
        Cumulative explained variance ratio across all components.
    """

    def __init__(self, n_components: int | None = None) -> None:
        """Initialize the PCA model.

        Parameters
        ----------
        n_components : int | None, default=None
            Number of principal components to keep by default.
            If ``None``, all components are stored after fitting.
        """

        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA on a data matrix ``X`` with shape ``(N, d)``.

        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            Data matrix where ``N`` is the number of samples and ``d`` is the
            number of features/pixels after flattening each image.

        Returns
        -------
        PCA
            The fitted PCA instance.
        """

        X = self._validate_input_matrix(X)
        num_samples, num_features = X.shape

        if num_samples < 2:
            raise ValueError("PCA requires at least two samples to estimate covariance.")

        # Step 1: compute the feature-wise mean, shape (d,)
        self.mean_ = np.mean(X, axis=0)

        # Step 2: center the data, shape (N, d)
        X_centered = X - self.mean_

        # Step 3: compute the covariance matrix, shape (d, d)
        covariance_matrix = (X_centered.T @ X_centered) / num_samples

        # Step 4: eigendecompose the symmetric covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 5: sort from large to small
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Numerical precision can produce tiny negative values near zero.
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

        total_variance = float(np.sum(eigenvalues))
        if total_variance == 0.0:
            all_explained_variance_ratio = np.zeros_like(eigenvalues)
        else:
            all_explained_variance_ratio = eigenvalues / total_variance

        self.n_features_in_ = num_features
        self.n_samples_ = num_samples
        self.all_components_ = eigenvectors
        self.all_explained_variance_ = eigenvalues
        self.all_explained_variance_ratio_ = all_explained_variance_ratio
        self.cumulative_explained_variance_ratio_ = np.cumsum(all_explained_variance_ratio)

        # Step 6: keep the top k eigenvectors
        self.n_components_ = self._resolve_n_components(num_features)
        self.components_ = self.all_components_[:, : self.n_components_]
        self.explained_variance_ = self.all_explained_variance_[: self.n_components_]
        self.explained_variance_ratio_ = self.all_explained_variance_ratio_[: self.n_components_]

        return self

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Project data from the original space into the PCA subspace.

        Parameters
        ----------
        X : np.ndarray of shape (N, d)
            Input data matrix to project.
        n_components : int | None, default=None
            Number of leading components to use for projection. If ``None``,
            the default number chosen at fit time is used.

        Returns
        -------
        np.ndarray of shape (N, k)
            Low-dimensional representation ``Z``.
        """

        self._check_is_fitted()
        X = self._validate_input_matrix(X)
        components = self._get_components(n_components)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but received {X.shape[1]}."
            )

        # Step 7: Z = X_centered @ components
        X_centered = X - self.mean_
        return X_centered @ components

    def inverse_transform(self, Z: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Reconstruct data from PCA space back to the original feature space.

        Parameters
        ----------
        Z : np.ndarray of shape (N, k)
            Projected data in the PCA subspace.
        n_components : int | None, default=None
            Number of components represented by ``Z``. If ``None``, the method
            infers the component count from ``Z.shape[1]``.

        Returns
        -------
        np.ndarray of shape (N, d)
            Reconstructed data matrix in the original feature space.
        """

        self._check_is_fitted()
        Z = self._validate_input_matrix(Z)

        inferred_components = Z.shape[1] if n_components is None else n_components
        components = self._get_components(inferred_components)

        if Z.shape[1] != components.shape[1]:
            raise ValueError(
                "The number of columns in Z must match the number of components used "
                "for reconstruction."
            )

        # Step 8: X_hat = Z @ components.T + mean
        return Z @ components.T + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return the projected data in one step."""

        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Convenience method: project and reconstruct in one call."""

        Z = self.transform(X, n_components=n_components)
        return self.inverse_transform(Z, n_components=n_components)

    def _resolve_n_components(self, num_features: int) -> int:
        """Validate and resolve the effective number of components."""

        if self.n_components is None:
            return num_features

        if not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer or None.")

        if self.n_components < 1 or self.n_components > num_features:
            raise ValueError(
                f"n_components must be in [1, {num_features}], got {self.n_components}."
            )

        return self.n_components

    def _get_components(self, n_components: int | None) -> np.ndarray:
        """Return the first ``k`` principal directions, shape ``(d, k)``."""

        self._check_is_fitted()
        effective_components = self.n_components_ if n_components is None else n_components

        if not isinstance(effective_components, int):
            raise TypeError("n_components must be an integer or None.")

        if effective_components < 1 or effective_components > self.all_components_.shape[1]:
            raise ValueError(
                "Requested n_components is outside the fitted component range."
            )

        return self.all_components_[:, :effective_components]

    @staticmethod
    def _validate_input_matrix(X: np.ndarray) -> np.ndarray:
        """Convert input to a 2D floating-point NumPy array."""

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D matrix with shape (N, d).")
        return X

    def _check_is_fitted(self) -> None:
        """Raise a helpful error if the model has not been fitted yet."""

        required_attributes = ("mean_", "all_components_", "n_features_in_")
        if not all(hasattr(self, attr_name) for attr_name in required_attributes):
            raise AttributeError("PCA model is not fitted yet. Call fit(X) first.")
