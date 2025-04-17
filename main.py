import numpy as np


def fpca(t: np.ndarray, x: np.ndarray, k: int):
    N, n, p = x.shape

    mu = np.mean(x, axis=1, keepdims=True)
    x_norm = x - mu
    cov = (x_norm.transpose(0,2,1) @ x_norm) / (n-1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Order the eigenvalues and eigenvectors by descending order
    idx = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
    eigenvectors = np.take_along_axis(eigenvectors, idx[:,np.newaxis,:], axis=-1)

    # select correctly the sign of the eigenvectors
    for i in range(1, eigenvectors.shape[0]):
        eigenvectors[i] *= np.sign(np.sum(eigenvectors[i] * eigenvectors[i-1], axis=0, keepdims=True))

    Z = x_norm @ eigenvectors[:, :, :k]
    x_norm_hat = Z @ eigenvectors[:, :, :k].transpose(0,2,1)
    return np.mean(np.square(x_norm - x_norm_hat))


def ffpca(t: np.ndarray, x: np.ndarray, k: int):
    N, n, p = x.shape

    mu = np.mean(x, axis=1, keepdims=True)
    x = x - mu

    # Calculate $\hat{X}$ and its covariance statistic
    x_hat = np.fft.fft(x, axis=0)
    cov_hat = (x_hat.transpose(0,2,1) @ np.conjugate(x_hat)) / (n-1)

    # Calculate the eigenvalues and eigenvectors of the covariance in the Fourier domain and order by descending order
    eval_hat, evec_hat = np.linalg.eigh(cov_hat)

    idx = np.argsort(eval_hat, axis=1)[:, ::-1]
    eval_hat = np.take_along_axis(eval_hat, idx, axis=-1)
    evec_hat = np.take_along_axis(evec_hat, idx[:,np.newaxis,:], axis=-1)

    for i in range(1, evec_hat.shape[0]):
        evec_hat[i] *= np.sign(np.sum(evec_hat[i] * evec_hat[i-1], axis=0, keepdims=True))

    Z = x_hat @ evec_hat[:, :, :k].conjugate()
    y_hat = Z @ evec_hat[:, :, :k].transpose(0,2,1)

    y = np.fft.ifft(y_hat, axis=0)
    return np.mean(np.square(np.absolute(x - y)))