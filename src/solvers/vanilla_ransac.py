import logging

from src.model import BaseModel
from src.ransac_solver import BaseRansacSolver
import numpy as np

logger = logging.getLogger(__name__)


class RansacSolver(BaseRansacSolver):
    """
    Vanilla RANSAC.
    """

    def __init__(self,
                 model: BaseModel,
                 n_sample_points: int,
                 error_threshold: float = None,
                 max_trials: int = 100,
                 score_threshold: float = None,
                 ):
        """

        Parameters
        ----------
        model : Model that specifies fit and calculate_errors methods.
        error_threshold : Threshold value for errors; if error > threshold, it is considered as an outlier.
        n_sample_points : Number of points for random sample used to fit model candidate.
        max_trials : Max number of trials (iterations).
        score_threshold : Threshold value for score; stop iteration if score > threshold.
        """
        self.model = model
        self.max_trials = max_trials
        self.n_sample_points = n_sample_points
        self.error_threshold = error_threshold
        self.score_threshold = score_threshold

    def sample_data(self):
        return self.data.get_random_sample(self.n_sample_points)

    def get_inliers(self) -> np.ndarray:
        errors = self.model.calculate_errors(self.data)
        return np.where(errors < self.error_threshold)[0]

    def calculate_score(self) -> float:
        return len(self.inlier_indices_candidate)

    def is_solution_valid(self) -> bool:
        return True

    def check_termination(self) -> bool:
        if self.best_score >= self.score_threshold:
            logger.info(f"Score {self.best_score} >= threshold {self.score_threshold}. End iteration.")
            return True
        else:
            return False

    def init(self):
        if self.score_threshold is None:
            logger.debug(f"Score threshold is not specified. "
                         f"Set threshold to be equal to number of data points.")
            self.score_threshold = len(self.data)
        BaseRansacSolver.init(self)
