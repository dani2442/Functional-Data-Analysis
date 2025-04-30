import numpy as np
import pandas as pd
from tqdm import tqdm

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
    

def estimate_fourier_cov_mean(x: pd.DataFrame, N: int, bs: int=256, n: int=100):
    p = x.shape[1]
    max_start = x.shape[0]-N

    cov_hat = np.zeros((N, p, p))
    mu_hat = np.zeros((N, 1, p))

    for i in tqdm(range(1,n+1)):
        starts = np.random.randint(0, max_start, size=bs)
        windows = [x.iloc[start:start+N].to_numpy() for start in starts]
        x_i = np.stack(windows).transpose(1,0,2)
        
        x_hat = np.fft.fft(x_i, axis=0)
        mu_hat_i = np.mean(x_hat, axis=1, keepdims=True)
        mu_hat = (mu_hat + mu_hat_i/(i*bs)) * (i/(i+1))

        x_hat = x_hat - mu_hat
        cov_hat_i = x_hat.transpose(0,2,1) @ x_hat
        cov_hat = (cov_hat + cov_hat_i/(i*bs)) *(i/(i+1))

    return cov_hat, mu_hat

    
def estimate_cov_mean(x: pd.DataFrame, N: int, bs: int=256, n: int=100):
    p = x.shape[1]
    max_start = x.shape[0]-N

    cov = np.zeros((N, p, p))
    mu = np.zeros((N, 1, p))

    for i in tqdm(range(1,n+1)):
        starts = np.random.randint(0, max_start, size=bs)
        windows = [x.iloc[start:start+N].to_numpy() for start in starts]
        x_i = np.stack(windows).transpose(1,0,2)

        mu_i = np.mean(x_i, axis=1, keepdims=True)
        mu = (mu + mu_i/(i*bs)) * (i/(i+1))

        x_i = x_i - mu
        cov_i = x_i.transpose(0,2,1) @ x_i
        cov = (cov + cov_i/(i*bs)) *(i/(i+1))

    return cov, mu
