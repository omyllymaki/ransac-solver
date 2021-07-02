import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from samples.polynomial_fit.model import Model
from src.common_data_types.data_xy import Data
from src.solvers.vanilla_ransac import RansacSolver

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(42)

# Data generation parameters
N_INLIERS = 500
N_OUTLIERS = 20
COEFFICIENTS = np.array([0.02, -0.01, 0.00, 0.1, 0.3, 1.0])  # X2, Y2, XY, X, Y, OFFSET
RANGE = 200
NOISE = 0.2
NOISE_OUTLIER = 20
OUTLIER_OFFSET = 0
N_POINTS = N_INLIERS + N_OUTLIERS

# Ransac solver fit parameters
MAX_ERROR = 2
N_FIT_POINTS = 10


def generate_test_data():
    X2, Y2, XY, X, Y, OFFSET = COEFFICIENTS
    xs = np.array([np.random.uniform(2 * RANGE) - RANGE for i in range(N_POINTS)])
    ys = np.array([np.random.uniform(2 * RANGE) - RANGE for i in range(N_POINTS)])
    zs = []
    for i in range(N_INLIERS):
        z = X2 * xs[i] * xs[i] + Y2 * ys[i] * ys[i] + XY * xs[i] * ys[i] + X * xs[i] + Y * ys[i] + OFFSET
        zs.append(z + np.random.normal(scale=NOISE))
    for i in range(N_OUTLIERS):
        z = X2 * xs[i] * xs[i] + Y2 * ys[i] * ys[i] + XY * xs[i] * ys[i] + X * xs[i] + Y * ys[i] + OFFSET
        zs.append(z + np.random.normal(scale=NOISE_OUTLIER) + OUTLIER_OFFSET)
    zs = np.array(zs)

    xyz = np.vstack((xs, ys, zs)).T
    return xyz


def get_features_and_targets(xyz):
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    zs = xyz[:, 2]
    X = np.vstack((xs * xs, ys * ys, xs * ys, xs, ys, np.ones((len(zs))))).T
    targets = np.array(zs).reshape(-1, 1)
    return X, targets


def plot_results(xyz, inlier_indices):
    ax = plt.subplot(111, projection='3d')

    inliers = xyz[inlier_indices, :]
    xs_inliers = inliers[:, 0]
    ys_inliers = inliers[:, 1]
    zs_inliers = inliers[:, 2]
    ax.scatter(xs_inliers, ys_inliers, zs_inliers, color='r')

    outlier_indices = list(set(range(xyz.shape[0])) - set(inlier_indices))
    outliers = xyz[outlier_indices, :]
    xs_outliers = outliers[:, 0]
    ys_outliers = outliers[:, 1]
    zs_outliers = outliers[:, 2]
    ax.scatter(xs_outliers, ys_outliers, zs_outliers, color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def main():
    xyz = generate_test_data()
    X, targets = get_features_and_targets(xyz)
    data = Data(x=X, y=targets)

    model = Model(l2_reg=0.0)
    solver = RansacSolver(model=model,
                          error_threshold=MAX_ERROR,
                          n_sample_points=N_FIT_POINTS)
    time_start = time.time()
    inlier_indices = solver.fit(data)
    time_end = time.time()
    print(f"Solution took {time_end - time_start} s")

    print("Solution:")
    print(f"Fitted parameters vs true parameters:\n {np.hstack((model.parameters, COEFFICIENTS.reshape(-1, 1)))}")
    print(f"Number of inliers: {len(inlier_indices)}")

    plot_results(xyz, inlier_indices)
    plt.show()


if __name__ == "__main__":
    main()
