# Project Plan 1

## Project Title

Interactive PCA Image Compression Explorer

## Project Goal

The goal of this course final project is to build an interactive local system that demonstrates how PCA can be used for image compression and reconstruction. The project should be easy to run, easy to explain, and suitable for classroom presentation.

The system will focus on:

- PCA from scratch
- image compression
- image reconstruction
- visualization of variance and reconstruction quality
- Streamlit-based interaction

## Core Idea

Each Fashion-MNIST image is a `28 x 28` grayscale image. We flatten each image into a vector of length `784`, so the dataset becomes a matrix `X` with shape `(N, d)`.

We then apply PCA using the following steps:

1. Compute the mean vector of the dataset.
2. Center the data.
3. Compute the covariance matrix.
4. Perform eigen-decomposition.
5. Sort eigenvalues and eigenvectors from large to small.
6. Keep the top `k` principal components.
7. Project each image into the low-dimensional space.
8. Reconstruct the image from the compressed representation.

The user can change `k` and immediately observe the reconstruction result.

## Dataset Plan

- Dataset: Fashion-MNIST
- Download method: `torchvision.datasets.FashionMNIST(download=True)`
- Local storage folder: `data/`
- Input format: flattened vectors with shape `(N, 784)`
- Output labels: integer category labels with human-readable class names

To keep the app responsive, the app will support using only the first part of the dataset, such as `500`, `1000`, `2000`, or `5000` samples.

## Module Division

### 1. `src/pca.py`

This is the core algorithm file.

Responsibilities:

- mean centering
- covariance matrix computation
- eigen-decomposition
- explained variance computation
- dimensionality reduction
- image reconstruction

### 2. `src/data_loader.py`

Responsibilities:

- download Fashion-MNIST
- load train or test split
- flatten images into `(N, d)`
- optionally normalize pixel values to `[0, 1]`
- return class labels and image shape

### 3. `src/visualization.py`

Responsibilities:

- original image vs reconstructed image comparison
- explained variance ratio curve
- cumulative explained variance curve

### 4. `app.py`

Responsibilities:

- Streamlit page layout
- control panel for sample index and `k`
- display of reconstruction results
- display of error metrics and plots
- user-friendly interaction

### 5. `src/utils.py`

Responsibilities:

- reshape flattened vectors back into images
- compute MSE
- normalize values for display

### 6. `tests/test_pca.py`

Responsibilities:

- basic PCA behavior tests
- shape checks for fit, transform, and inverse transform
- explained variance ratio checks

## Expected User Flow

1. Install dependencies with `pip install -r requirements.txt`.
2. Run the app with `streamlit run app.py`.
3. The app downloads Fashion-MNIST into `data/` if it is not already present.
4. The system fits PCA on the selected number of samples.
5. The user chooses a sample index and a compression dimension `k`.
6. The app displays:
   - original image
   - reconstructed image
   - MSE
   - explained variance ratio curve
   - cumulative explained variance curve

## Five-Person Team Division

1. **Abel**
   Responsible for the PCA core compression algorithm, including centering, covariance matrix computation, eigen-decomposition, projection, and reconstruction.
2. **Member B**
   Responsible for data loading and preprocessing, including Fashion-MNIST download, data conversion, and data format management.
3. **Member C**
   Responsible for visualization, including image comparison, explained variance plots, and cumulative explained variance plots.
4. **Member D**
   Responsible for frontend interaction and Streamlit page development, including layout, widgets, and interaction logic.
5. **Member E**
   Responsible for integration, testing, and documentation, including module integration, testing, one-command execution, and README writing.

This division keeps algorithm, data, visualization, interaction, and integration clearly separated, which makes collaboration easier.

## Why This Plan Works

- The method is mathematically clean and easy to explain.
- The interface is visual and presentation-friendly.
- The project is stable because it runs locally and does not depend on external APIs.
- The workload is naturally split across five team members.
- The final product is suitable for both demonstration and written reporting.
