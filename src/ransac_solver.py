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
                 n_inlier_threshold: int = None
                 ):
        """
        @param model: Model that specifies fit and calculate_errors methods.
        @param error_threshold: Threshold value for errors; if error > threshold, it considered as outlier.
        @param n_sample_points: Number of points for random sample used to fit model candidate.
        @param max_trials: Max number of trials (iterations)
        @param n_inlier_threshold: Threshold value for number inliers; stop iteration if number of inliers > threshold.
        """
        self.model = model
        self.error_threshold = error_threshold
        self.max_trials = max_trials
        self.n_sample_points = n_sample_points
        self.n_inlier_threshold = n_inlier_threshold

    def solve(self, data: BaseData) -> np.ndarray:

        if self.n_inlier_threshold is None:
            logger.debug(f"Number of inliers threshold is not specified. "
                         f"Set threshold to be equal to number of data points.")
            self.n_inlier_threshold = len(data)

        max_n_inliers = 0
        final_inlier_indices = None

        for trial_number in range(self.max_trials):

            self._fit_with_random_sample(data)

            inlier_indices = self._get_inliers(data)
            n_inliers = len(inlier_indices)
            logger.debug(f"Trial {trial_number}: number of inliers  {n_inliers}")

            # Update solution if number of inliers has grown
            if n_inliers > max_n_inliers:
                logger.info(f"Found better solution. Number of inliers {n_inliers}")
                final_inlier_indices = inlier_indices
                max_n_inliers = n_inliers

                if max_n_inliers >= self.n_inlier_threshold:
                    logger.info(f"Number of inliers exceed threshold level {self.n_inlier_threshold}.")
                    break

        logger.info(f"Iteration finished. Number of inliers: {max_n_inliers}")

        # Do final fitting with all the inliers
        logger.debug(f"Fit model with full inlier dataset.")
        inliers = data.get_sample(final_inlier_indices)
        self.model.fit(inliers)

        return final_inlier_indices

    def _fit_with_random_sample(self, data):
        sample = data.get_random_sample(self.n_sample_points)
        self.model.fit(sample)

    def _get_inliers(self, data):
        errors = self.model.calculate_errors(data)
        return np.where(errors < self.error_threshold)[0]
