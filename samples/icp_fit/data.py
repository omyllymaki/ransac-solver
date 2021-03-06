import numpy as np

from src.data import BaseData, Indices


class Data(BaseData):

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source
        self.target = target

    def __getitem__(self, indices: Indices):
        return Data(self.source[indices], self.target)

    def __len__(self):
        return self.source.shape[0]
