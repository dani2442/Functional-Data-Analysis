import numpy as np

class FPCA:
    def __init__(self, mu: np.ndarray = None, eigenvectors: np.ndarray = None):
        self.mu = mu
        self.eigenvectors = eigenvectors

    def set_eigenvectors(self, cov: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Order by descending eigenvalue
        idx = np.argsort(eigenvalues, axis=1)[:, ::-1]
        eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
        eigenvectors = np.take_along_axis(eigenvectors, idx[:, np.newaxis, :], axis=-1)

        # Align eigenvector signs
        for i in range(1, eigenvectors.shape[0]):
            eigenvectors[i] *= np.sign(np.sum(eigenvectors[i] * eigenvectors[i - 1], axis=0, keepdims=True))
        self.eigenvectors = eigenvectors

    def set_mean(self, mean: np.ndarray):
        self.mu = mean

    def fit(self, t: np.ndarray, x: np.ndarray):
        N, n, p = x.shape

        self.mu = np.mean(x, axis=1, keepdims=True)
        x_norm = x - self.mu
        cov = (x_norm.transpose(0, 2, 1) @ x_norm) / (n - 1)
        self.set_eigenvectors(cov)

    def transform(self, x: np.ndarray, k: int) -> np.ndarray:
        if self.mu is None or self.eigenvectors is None:
            raise RuntimeError("The model must be fitted before calling transform().")

        x_norm = x - self.mu
        Z = x_norm @ self.eigenvectors[:, :, :k]
        x_norm_hat = Z @ self.eigenvectors[:, :, :k].transpose(0, 2, 1)
        return x_norm_hat + self.mu
    


class FFPCA:
    def __init__(self, mu: np.ndarray = None, eigenvectors: np.ndarray = None):
        self.mu = mu
        self.eigenvectors = eigenvectors

    def set_eigenvectors(self, cov: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Order by descending eigenvalue
        idx = np.argsort(eigenvalues, axis=1)[:, ::-1]
        eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
        eigenvectors = np.take_along_axis(eigenvectors, idx[:, np.newaxis, :], axis=-1)

        # Align eigenvector signs
        for i in range(1, eigenvectors.shape[0]):
            eigenvectors[i] *= np.sign(np.sum(eigenvectors[i] * eigenvectors[i - 1], axis=0, keepdims=True))
        self.eigenvectors=eigenvectors

    def set_mean(self, mu: np.ndarray):
        self.mu = mu

    def fit(self, t: np.ndarray, x: np.ndarray):
        N, n, p = x.shape
        #self.mu = np.mean(x, axis=1, keepdims=True)
        #x_norm = x - self.mu
        x_hat = np.fft.fft(x, axis=0)
        
        self.mu = np.mean(x_hat, axis=1, keepdims=True)
        x_hat = x_hat - self.mu

        cov_hat = (x_hat.transpose(0,2,1) @ np.conjugate(x_hat)) / (n-1)
        self.set_eigenvectors(cov_hat)

    def transform(self, x: np.ndarray, k: int) -> np.ndarray:
        if self.mu is None or self.eigenvectors is None:
            raise RuntimeError("The model must be fitted before calling transform().")

        x_hat = np.fft.fft(x, axis=0)
        x_hat = x_hat - self.mu
        Z = x_hat @ self.eigenvectors[:, :, :k].conjugate()
        y_hat = Z @ self.eigenvectors[:, :, :k].transpose(0,2,1)

        y_hat = y_hat + self.mu

        y = np.fft.ifft(y_hat, axis=0)
        return y
    