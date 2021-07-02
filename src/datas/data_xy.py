from src.data import BaseData, Indices
import numpy as np


class Data(BaseData):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __getitem__(self, indices: Indices):
        return Data(self.x[indices], self.y[indices])

    def __len__(self):
        return self.x.shape[0]
