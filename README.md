# Functional Data Analysis: PCA, FPCA, and fFPCA

This repository provides tools, implementations, and experiments for **Functional Data Analysis (FDA)**, with a focus on Principal Component Analysis (PCA), Functional Principal Component Analysis (FPCA), and Fourier-based Functional PCA (fFPCA). The codebase includes Python modules and Jupyter notebooks for exploring, benchmarking, and visualizing these methods on synthetic and real-world datasets.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [PCA Example](#pca-example)
  - [FPCA Example](#fpca-example)
  - [fFPCA Example](#ffpca-example)
- [API Reference](#api-reference)
- [Notebooks](#notebooks)
- [Data](#data)
- [Experiments](#experiments)
- [References](#references)

---

## Overview

**Functional Data Analysis (FDA)** is a branch of statistics that analyzes data providing information about curves, surfaces, or anything else varying over a continuum. FDA is widely used in fields such as signal processing, biostatistics, econometrics, and more.

This project implements and compares:

- **PCA**: Standard Principal Component Analysis for multivariate data.
- **FPCA**: Functional PCA for data observed over a continuum (e.g., time series, curves).
- **fFPCA**: Functional PCA in the Fourier domain, leveraging frequency information for improved analysis of periodic or oscillatory data.

The repository includes:

- Modular Python implementations of PCA, FPCA, and fFPCA.
- Experiment scripts and notebooks for synthetic and real data.
- Utilities for estimating covariance and mean functions in both time and frequency domains.
- Visualization tools for interpreting results.

---

## Project Structure

```
.
├── fpca.py                # FPCA and fFPCA class implementations and utilities
├── main.py                # Simple API functions for FPCA and fFPCA
├── notebooks/             # Jupyter notebooks for experiments and demonstrations
│   ├── pca.ipynb
│   ├── fpca.ipynb
│   ├── kfpca.ipynb
│   └── experiments.ipynb
├── data/                  # (Optional) Data files for experiments
├── images/                # Generated figures and plots
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Functional-Data-Analysis.git
   cd Functional-Data-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include `numpy`, `matplotlib`, `pandas`, `tqdm`, and `scienceplots`. For some notebooks, you may also need `scikit-fda`.

---

## Usage

### PCA Example

See `notebooks/pca.ipynb` for a demonstration of standard PCA on synthetic data.

### FPCA Example

```python
from main import fpca
import numpy as np

# Simulate data: x of shape (N, n, p)
t = np.linspace(0, 1, 200)
x = ... # your data, shape (N, n, p)
k = 3   # number of principal components

x_hat = fpca(t, x, k)
```

### fFPCA Example

```python
from main import ffpca

x_hat = ffpca(t, x, k)
```

For more advanced usage, including fitting and transforming with the `FPCA` and `FFPCA` classes, see the API Reference below.

---

## API Reference

### `fpca(t: np.ndarray, x: np.ndarray, k: int) -> np.ndarray`

Performs Functional PCA on data `x` observed at time points `t` using `k` principal components.

- **Parameters:**
  - `t`: 1D array of time points (length N)
  - `x`: Data array of shape (N, n, p)
  - `k`: Number of principal components
- **Returns:** Reconstructed data array of shape (N, n, p)

### `ffpca(t: np.ndarray, x: np.ndarray, k: int) -> np.ndarray`

Performs Fourier-based Functional PCA.

- **Parameters:** Same as `fpca`
- **Returns:** Reconstructed data array

### `FPCA` class (`fpca.py`)

Object-oriented interface for FPCA.

- `fit(t, x)`: Fit the model to data.
- `transform(x, k)`: Project and reconstruct data using `k` components.

### `FFPCA` class (`fpca.py`)

Object-oriented interface for Fourier-based FPCA.

- `fit(t, x)`: Fit the model in the Fourier domain.
- `transform(x, k)`: Project and reconstruct data using `k` components.

### `estimate_cov_mean(x: pd.DataFrame, N: int, bs: int=256, n: int=100)`

Estimate covariance and mean functions from sliding windows of data.

### `estimate_fourier_cov_mean(x: pd.DataFrame, N: int, bs: int=256, n: int=100)`

Estimate covariance and mean functions in the Fourier domain.

---

## Notebooks

- **`notebooks/pca.ipynb`**: Standard PCA on synthetic data.
- **`notebooks/fpca.ipynb`**: Functional PCA on simulated Brownian motion.
- **`notebooks/kfpca.ipynb`**: Kernel FPCA and advanced covariance structures.
- **`notebooks/experiments.ipynb`**: Benchmarking and comparison of FPCA and fFPCA.

---

## Data

Synthetic data is generated within the notebooks and scripts. For real-world datasets, place your files in the `data/` directory and adapt the loading code as needed.

---

## Experiments

To reproduce experiments and figures, run the notebooks in the `notebooks/` directory. Results such as reconstruction errors and visualizations are saved in the `images/` directory.

---

## References

- Ramsay, J., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer. [doi:10.1007/b98888](https://doi.org/10.1007/b98888)
- Other references and links are provided in the respective notebooks.

---

## License

This project is licensed under the MIT License.
