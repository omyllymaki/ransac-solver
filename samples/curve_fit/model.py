from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from src.common_data_types.data_xy import Data
from src.model import BaseModel


class Model(BaseModel):
    """
    Wrapper for scipy curve fit.
    """

    def __init__(self, fit_function , **fit_method_kwargs):
        self.fit_function = fit_function
        self.curve_fit = partial(curve_fit, f=fit_function, **fit_method_kwargs)
        self.parameters = None

    def fit(self, data: Data):
        try:
            self.parameters, _ = self.curve_fit(xdata=data.x, ydata=data.y)
        except RuntimeError:
            self.parameters = np.array([0, 0, 0])

    def estimate(self, data: Data):
        return self.fit_function(data.x, *self.parameters)

    def calculate_errors(self, data: Data):
        y_estimate = self.estimate(data)
        return np.abs(y_estimate - data.y)
