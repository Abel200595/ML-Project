# Interactive PCA Image Compression Explorer



This project is a course final project that demonstrates how **Principal Component Analysis (PCA)** can be used for **image compression and reconstruction**. The system is built around a local **Streamlit** app so users can interactively choose the number of principal components, browse Fashion-MNIST samples, compare original and reconstructed images, and observe how explained variance changes as the compression dimension changes.

The main emphasis of this project is not deep learning. Instead, it focuses on:

- implementing **PCA from scratch**
- using PCA for **compression and reconstruction**
- building a **clear and stable interactive demo**
- making the math easy to visualize and explain in a course presentation

## Project Goals

The project is designed to answer a simple but meaningful question:

**If we compress an image into a lower-dimensional representation using PCA, how much visual information can we keep?**

To support that goal, the application allows users to:

- choose a compression dimension `k`
- browse sample images from Fashion-MNIST
- compare the original image with the PCA reconstructed image
- view the explained variance ratio curve
- view the cumulative explained variance curve
- check reconstruction error using MSE

## Dataset

- **Dataset name:** Fashion-MNIST
- **Source:** Fashion-MNIST provided by Zalando Research
- **Recommended usage:** downloaded automatically through `torchvision.datasets.FashionMNIST`
- **Local storage directory:** `data/`

This project does not require users to manually download files from Kaggle or any private source. When the app runs for the first time, `torchvision` will automatically download the dataset into the local `data/` directory.

## Why PCA

PCA is a classic dimensionality reduction method. In this project, each `28 x 28` grayscale image is flattened into a vector of length `784`. PCA finds the most important directions of variation in the dataset, and then uses only the top `k` directions to represent each image.

In simple terms:

1. Compute the average image.
2. Subtract the average image from every sample.
3. Build the covariance matrix.
4. Find the eigenvectors with the largest eigenvalues.
5. Keep only the top `k` principal components.
6. Project the image to a lower-dimensional space.
7. Reconstruct the image from that compressed representation.

This makes PCA a strong fit for a course project because the method is mathematically clean, easy to explain, and visually intuitive.

## Project Structure

```text
project/
├── app.py
├── run_app.sh
├── README.md
├── requirements.txt
├── plan1.md
├── data/
├── src/
│   ├── __init__.py
│   ├── pca.py
│   ├── data_loader.py
│   ├── visualization.py
│   └── utils.py
├── tests/
│   └── test_pca.py
└── assets/
    └── screenshots/
```

## Module Description

- `app.py`: Streamlit entry point and page layout
- `src/pca.py`: the core PCA algorithm implemented from scratch
- `src/data_loader.py`: Fashion-MNIST download and preprocessing logic
- `src/visualization.py`: plotting functions for image comparison and variance curves
- `src/utils.py`: helper functions such as reshaping and MSE computation
- `tests/test_pca.py`: basic pytest tests for PCA behavior
- `plan1.md`: short design document for the project

## Installation

Use the project virtual environment so `torch` / `torchvision` (needed for dataset download) match the same Python that runs Streamlit:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the App

**Recommended** (always uses `.venv`):

```bash
./run_app.sh
```

Or explicitly:

```bash
.venv/bin/streamlit run app.py
```

Avoid running plain `streamlit run app.py` if your shell’s `streamlit` points at another Python (for example Conda) that does not have `torchvision` installed.

After the command starts, Streamlit will open a local browser page. On first run, the app may take a bit longer because Fashion-MNIST will be downloaded automatically into `data/`.

## Run Tests

```bash
pytest
```

## App Features

The Streamlit app includes the following interactive components:

- a project header and project introduction
- a control panel for choosing dataset split, sample index, and compression dimension `k`
- a side-by-side comparison of the original image and the reconstructed image
- a displayed reconstruction error (MSE)
- an explained variance ratio plot
- a cumulative explained variance plot

The app is intentionally simple so it remains stable and easy to demonstrate during a live course presentation.

## PCA Method Summary

Given a data matrix `X` with shape `(N, d)`:

1. Compute the mean vector `mean_` with shape `(d,)`.
2. Center the data: `X_centered = X - mean_`.
3. Compute covariance: `C = (1 / N) * X_centered^T * X_centered`.
4. Perform eigen-decomposition on `C`.
5. Sort eigenvalues from large to small and reorder eigenvectors accordingly.
6. Keep the first `k` eigenvectors as `components_` with shape `(d, k)`.
7. Project the original data: `Z = X_centered @ components_`.
8. Reconstruct the data: `X_hat = Z @ components_.T + mean_`.

This exact logic is explicitly implemented in `src/pca.py`.

