from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

Indices = Union[List[int], np.ndarray]


class BaseData(ABC):
    """
    Abstract Data class that needs to be implemented by user.
    """

    @abstractmethod
    def __getitem__(self, indices: Indices) -> 'BaseData':
        """
        Get sample points based on indices.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Get number of sample points.
        """
        raise NotImplementedError

    def get_random_sample(self, n_items: int) -> 'BaseData':
        indices = np.random.choice(len(self), n_items, replace=False)
        return self[indices]
