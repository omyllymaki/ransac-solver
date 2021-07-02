import numpy as np

from src.datas.data_xy import Data
from src.model import BaseModel


class Model(BaseModel):
    """
    Polynomial model with regularization.

    E.g. model
    y = k1*x^2 + k2*y2 + k3*xy + k4*x + k5*y + offset

    X is matrix with columns [x^2, y^2, xy, x, y, 1] (n_samples x 6)
    y is matrix with one column [y] (n_samples x 1)
    fit returns parameters [k1, k2, k3, k4, k5, offset] as solution
    """

    def __init__(self, l2_reg=0.0):
        self.l2_reg = l2_reg
        self.parameters = None

    def fit(self, data: Data):
        penalty = self.l2_reg * np.eye(data.x.shape[1])
        self.parameters = np.linalg.pinv(data.x.T @ data.x + penalty) @ data.x.T @ data.y

    def estimate(self, data: Data) -> np.ndarray:
        return data.x @ self.parameters

    def calculate_errors(self, data: Data):
        y_estimate = self.estimate(data)
        return np.abs(y_estimate - data.y)
