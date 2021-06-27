import logging
import sys

import numpy as np

from src.data import BaseData
from src.model import BaseModel

logger = logging.getLogger(__name__)


class RansacSolver:
    """
    General RANSAC solver. RANSAC is iterative method that can be used to fit model parameters when data contains
    high number of outliers.
    """

    def __init__(self,
                 model: BaseModel,
                 error_threshold: float,
                 n_sample_points: int,
                 max_trials: int = 100,
                 score_threshold: float = None
                 ):
        """
        @param model: Model that specifies fit and calculate_errors methods.
        @param error_threshold: Threshold value for errors; if error > threshold, it is considered as an outlier.
        @param n_sample_points: Number of points for random sample used to fit model candidate.
        @param max_trials: Max number of trials (iterations)
        @param score_threshold: Threshold value for score; stop iteration if score > threshold.
        """
        self.model = model
        self.error_threshold = error_threshold
        self.max_trials = max_trials
        self.n_sample_points = n_sample_points
        self.score_threshold = score_threshold

    def fit(self, data: BaseData) -> np.ndarray:
        """
        Fit model using RANSAC.

        Parameters
        ----------
        data : Use-case specific data that is defined by user.

        Returns
        -------
        inlier indices

        """

        if self.score_threshold is None:
            logger.debug(f"Score threshold is not specified. "
                         f"Set threshold to be equal to number of data points.")
            self.score_threshold = len(data)

        best_score = 0
        final_inlier_indices = None

        for trial_number in range(self.max_trials):

            sample = self.sample_data(data)
            self.model.fit(sample)

            inlier_indices = self.get_inliers(data)
            score = self.calculate_score(inlier_indices)
            logger.debug(f"Trial {trial_number}: score  {score}")

            # Update solution if best so far
            if score > best_score:
                logger.info(f"Found better solution. Score {score}")
                final_inlier_indices = inlier_indices
                best_score = score

                if best_score >= self.score_threshold:
                    logger.info(f"Score {best_score} > threshold {self.score_threshold}. End iteration.")
                    break

        logger.info(f"Iteration finished. Score: {best_score}")

        # Do final fitting with all the inliers
        logger.debug(f"Fit model with full inlier dataset.")
        inliers = data[final_inlier_indices]
        self.model.fit(inliers)

        return final_inlier_indices

    def sample_data(self, data: BaseData):
        return data.get_random_sample(self.n_sample_points)

    def get_inliers(self, data: BaseData):
        errors = self.model.calculate_errors(data)
        return np.where(errors < self.error_threshold)[0]

    def calculate_score(self, inlier_indices: np.ndarray) -> float:
        return len(inlier_indices)
