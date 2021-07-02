from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from src.datas.data_xy import Data
from src.model import BaseModel


class Model(BaseModel):
    """
    Wrapper for scipy curve fit.
    """

    def __init__(self, fit_function, **curve_fit_kwargs):
        self.fit_function = fit_function
        self.curve_fit = partial(curve_fit, f=fit_function, **curve_fit_kwargs)
        self.parameters = None

    def fit(self, data: Data):
        try:
            self.parameters, _ = self.curve_fit(xdata=data.x, ydata=data.y)
        except RuntimeError:  # curve_fit throws RuntimeError is solution is not found
            self.parameters = None

    def estimate(self, data: Data):
        if self.parameters is None:
            return None
        else:
            return self.fit_function(data.x, *self.parameters)

    def calculate_errors(self, data: Data):
        if self.parameters is None:
            return np.inf * np.ones(data.y.shape)
        y_estimate = self.estimate(data)
        return np.abs(y_estimate - data.y)
