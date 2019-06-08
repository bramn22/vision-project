import numpy as np
import cv2
from utils import normalize_data

class PCA:

    def __init__(self, data_raw):
        self.data_raw = data_raw
        self.data, self.mean, self.std = normalize_data(data_raw)
        print("Calculated mean: ", np.shape(self.mean))
        self.U, self.Sigma, self.V_T = np.linalg.svd(self.data, full_matrices=False)
        print("X:", np.shape(self.data))
        print("U:", self.U.shape)
        print("Sigma:", self.Sigma.shape)
        print("V^T:", self.V_T.shape)

    def subtract_mean(self, data_raw):
        mean = np.zeros(np.shape(data_raw[0]))
        for el in data_raw:
            mean += el
        mean /= (256 * len(data_raw))
        data = []
        for el in data_raw:
            sub = el / 256 - mean
            data.append(sub)
        return data, mean

    def get_reconstruction_error(self, pcs):
        coeffs = self.U[:, :pcs] * self.Sigma[:pcs]
        E = self.V_T[:pcs, :]
        #recon = (np.dot(coeffs, E)*self.std + self.mean)
        recon = np.dot(coeffs, E)
        mse = ((recon - self.data) ** 2).mean(axis=None)
        print(mse)
