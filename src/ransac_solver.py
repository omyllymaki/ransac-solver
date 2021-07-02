import logging
from abc import abstractmethod, ABC

from src.data import BaseData
import numpy as np

logger = logging.getLogger(__name__)


class BaseRansacSolver(ABC):
    """
    General RANSAC solver. RANSAC is iterative method that can be used to fit model parameters when data contains
    high number of outliers.

    This is abstract base solver class. All the abstract methods need to implemented by inheritor.
    """
    model = None
    data = None
    inlier_indices_candidate = None
    inlier_indices = None
    best_score = None
    trial_number = None
    max_trials = None

    def fit(self, data: BaseData, fit_with_final_inliers=True) -> np.ndarray:
        """
        Fit model using RANSAC.

        Parameters
        ----------
        data : Use-case specific data that is defined by user.
        fit_with_final_inliers: Fit final model with final inliers solved by the algorithm.

        Returns
        -------
        inlier indices
        """
        self.data = data
        self.init()

        for trial in range(self.max_trials):
            sample = self.sample_data()
            self.model.fit(sample)
            self.inlier_indices_candidate = self.get_inliers()
            is_valid_solution = self.is_solution_valid()
            if not is_valid_solution:
                logger.debug(f"Solution is not valid. Do not update inlier indices.")
                continue
            score = self.calculate_score()
            logger.debug(f"Trial {self.trial_number}: score  {score}")

            # Update solution if best so far
            if score > self.best_score:
                logger.info(f"Found better solution. Score {score}")
                self.inlier_indices = self.inlier_indices_candidate.copy()
                self.best_score = score

                is_terminated = self.check_termination()
                if is_terminated:
                    logger.info(f"Termination criteria filled. End iteration")
                    break

        logger.info(f"Iteration finished. Score: {self.best_score}")

        if fit_with_final_inliers:
            logger.debug(f"Fit model with full inlier dataset.")
            inliers = data[self.inlier_indices]
            self.model.fit(inliers)

        return self.inlier_indices

    def init(self):
        self.inlier_indices_candidate = None
        self.inlier_indices = None
        self.trial_number = 0
        self.best_score = -np.inf

    def check_termination(self) -> bool:
        """
        Check termination criteria.

        Returns
        -------
        boolean indicating if iteration should be terminated.
        """
        if len(self.data) == len(self.inlier_indices):
            return True
        else:
            return False

    @abstractmethod
    def is_solution_valid(self) -> bool:
        """
        Check is solution valid.

        Returns
        -------
        boolean indicating if solution valid or not.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_data(self) -> BaseData:
        """
        Draw sample from full dataset for model fitting.

        Returns
        -------
        Data sample.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inliers(self) -> np.ndarray:
        """
        Get inliers, based on the current fitted model.

        Returns
        -------
        inlier indices.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_score(self) -> float:
        """
        Calculate score for current solution (inlier candidates). Larger is better.

        Returns
        -------
        Score.
        """
        raise NotImplementedError
