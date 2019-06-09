import numpy as np
import cv2
from utils import normalize_data

class PCA:

    def __init__(self, data):
        self.data = data
        self.U, self.Sigma, self.V_T = np.linalg.svd(self.data, full_matrices=False)
        print("X:", np.shape(self.data))
        print("U:", self.U.shape)
        print("Sigma:", self.Sigma.shape)
        print("V^T:", self.V_T.shape)

    def get_reconstruction(self, pcs):
        coeffs = self.U[:, :pcs] * self.Sigma[:pcs]
        E = self.V_T[:pcs, :]
        # recon = (np.dot(coeffs, E)*self.std + self.mean)
        return np.dot(coeffs, E)

    def extract_pcs(self, pcs):
        coeffs = self.U[:, :pcs] * self.Sigma[:pcs]
        return coeffs
