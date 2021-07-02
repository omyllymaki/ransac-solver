import numpy as np


def estimate_n_trials_needed(inlier_ratio: float, n_sample: int, probability: float = 0.95) -> int:
    """
    Estimate number of trials needed to select only true inliers with RANSAC.

    Parameters
    ----------
    inlier_ratio : number of inliers / number of all points
    n_sample : number of samples needed to fit model
    probability : desired probability of success.

    Returns
    -------
    Estimated number of trials needed.
    """
    return int(np.log(1 - probability) / np.log(1 - inlier_ratio ** n_sample))
