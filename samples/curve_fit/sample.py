import logging
import time

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from samples.curve_fit.model import Model
from src.common_data_types.data_xy import Data
from src.ransac_solver import RansacSolver

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(42)

NOISE = 0.1
OUTLIER_NOISE = 1
OUTLIER_OFFSET = 1
N_OUTLIERS = 20


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def generate_test_data():
    x = np.linspace(0, 5, 100)
    y_true = func(x, 2.5, 1.3, 0.5)
    y_noise = NOISE * np.random.randn(len(y_true))
    y = y_true + y_noise
    i_outliers = np.random.choice(len(x), N_OUTLIERS, replace=False)
    y[i_outliers] = y[i_outliers] + OUTLIER_NOISE * np.random.randn(N_OUTLIERS) + OUTLIER_OFFSET
    return x, y_true, y


def main():
    x, y_true, y = generate_test_data()
    parameters, _ = curve_fit(func, x, y, maxfev = 2000)
    y_estimate = func(x, *parameters)

    model = Model(fit_function=func)
    data = Data(x=x, y=y)

    solver = RansacSolver(model=model, error_threshold=0.2, n_sample_points=15, max_trials=500)
    time_start = time.time()
    inlier_indices = solver.fit(data)
    time_end = time.time()
    print(f"Solution took {time_end - time_start} s")

    y_ransac_estimate = model.estimate(data)
    outlier_indices = list(set(range(len(x))) - set(inlier_indices))

    plt.plot()
    plt.plot(x, y, "b.-", label="Noisy signal")
    plt.plot(x[outlier_indices], y[outlier_indices], "kx", label="Outlier")
    plt.plot(x, y_true, 'r-', label="True")
    plt.plot(x, y_estimate, 'm-', label="Regular fit")
    plt.plot(x, y_ransac_estimate, 'g-', label="RANSAC fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
