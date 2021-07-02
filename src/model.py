from abc import abstractmethod, ABC

import numpy as np

from src.data import BaseData


class BaseModel(ABC):
    """
    Abstract Model class that needs to be implemented by user.
    """

    @abstractmethod
    def fit(self, data: BaseData) -> None:
        """
        Fit model parameters with data.

        Parameters
        ----------
        data : Data used to fit the model.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_errors(self, data: BaseData) -> np.ndarray:
        """
        Calculate error for every point in data, based on the current fitted model parameters. This errors are used to
        decide if data point is inlier or outlier, using some threshold value.

        Parameters
        ----------
        data : Full dataset that needs to be divided to inliers and outliers.

        Returns
        -------
        Error for every point in data, with current fitted model parameters.
        """
        raise NotImplementedError
