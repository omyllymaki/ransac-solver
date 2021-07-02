import logging

from samples.plane_fit.model import Model
from src.common_data_types.data_x import Data
from src.solvers.vanilla_ransac import RansacSolver
import time

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(42)

N_INLIERS = 100
N_OUTLIERS = 50
PLANE_PARAMETERS = [0.25, 0.5, 1]  # x, y, offset
RANGE = 20
NOISE = 0.1
NOISE_OUTLIER = 20
N_POINTS = N_INLIERS + N_OUTLIERS


def create_test_data():
    x_slope, y_slope, offset = PLANE_PARAMETERS
    xs = [np.random.uniform(2 * RANGE) - RANGE for i in range(N_POINTS)]
    ys = [np.random.uniform(2 * RANGE) - RANGE for i in range(N_POINTS)]
    zs = []
    for i in range(N_INLIERS):
        zs.append(xs[i] * x_slope + \
                  ys[i] * y_slope + \
                  offset + np.random.normal(scale=NOISE))
    for i in range(N_OUTLIERS):
        zs.append(xs[i] * x_slope + \
                  ys[i] * y_slope + \
                  offset + np.random.normal(scale=NOISE_OUTLIER))

    xyz = np.vstack((np.array(xs), np.array(ys), np.array(zs))).T
    return xyz


def main():
    xyz = create_test_data()
    model = Model()
    data = Data(xyz)
    solver = RansacSolver(model=model, error_threshold=1.0, n_sample_points=3, max_trials=200)
    time_start = time.time()
    inlier_indices = solver.fit(data)
    time_end = time.time()
    print(f"Solution took {time_end - time_start} s")

    param = model.parameters
    print("True plane equation:")
    print(f"z = {PLANE_PARAMETERS[0]:0.3f}x + {PLANE_PARAMETERS[1]:0.3f}y + {PLANE_PARAMETERS[2]:0.3f}")
    print("Solution:")
    print(f"z = {param[0]:0.3f}x + {param[1]:0.3f}y + {param[2]:0.3f}")
    print(f"Number of inliers: {len(inlier_indices)}")

    plt.figure()
    ax = plt.subplot(111, projection='3d')

    # plot inliers
    inliers = xyz[inlier_indices, :]
    xs_inliers = inliers[:, 0]
    ys_inliers = inliers[:, 1]
    zs_inliers = inliers[:, 2]
    ax.scatter(xs_inliers, ys_inliers, zs_inliers, color='r')

    # plot outliers
    outlier_indices = list(set(range(xyz.shape[0])) - set(inlier_indices))
    outliers = xyz[outlier_indices, :]
    xs_outliers = outliers[:, 0]
    ys_outliers = outliers[:, 1]
    zs_outliers = outliers[:, 2]
    ax.scatter(xs_outliers, ys_outliers, zs_outliers, color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


if __name__ == "__main__":
    main()
