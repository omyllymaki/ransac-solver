import logging

from samples.icp_fit.data import Data
from samples.icp_fit.icp.icp import icp, transform
from samples.icp_fit.model import Model
from src.ransac_solver import RansacSolver
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("samples.icp").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(1)

# Test data generation parameters
N_OUTLIERS = 20
N_INLIERS = 100
XY_RANGE = 20
ROTATION_AXIS = np.array([0, 0, 1])
ROTATION_ANGLE = 30
TRANSLATION_VECTOR = np.array([0, -10, 0])
NOISE_OUTLIER = 20
OFFSET_OUTLIER = np.array([70, 10, 0])


def generate_test_data():
    xyz = np.random.uniform(0, XY_RANGE, (N_INLIERS, 3))
    rotation_matrix = Rotation.from_euler('z', ROTATION_ANGLE, degrees=True).as_matrix()
    source = xyz.copy()
    target = xyz.copy()
    source = source @ rotation_matrix - TRANSLATION_VECTOR
    outliers = NOISE_OUTLIER * np.random.randn(N_OUTLIERS, 3) + OFFSET_OUTLIER
    source = np.vstack((source, outliers))
    source[:, 2] = 0
    target[:, 2] = 0
    return source, target


def main():
    source, target = generate_test_data()
    transformation_matrix, _ = icp(source.copy(),
                                   target.copy(),
                                   init_transform=None,
                                   max_iterations=100,
                                   cost_diff_threshold=0.0001)
    source_transformed = transform(transformation_matrix, source)

    data = Data(source, target)
    model = Model()
    solver = RansacSolver(model=model, error_threshold=10, n_sample_points=5, max_trials=200)
    inlier_indices = solver.fit(data)
    outlier_indices = list(set(range(len(data))) - set(inlier_indices))
    source_transformed_ransac = model.estimate(data)

    plt.figure()
    ax = plt.subplot(131)
    ax.plot(target[:, 0], target[:, 1], 'bo', alpha=0.5)
    ax.plot(source[:, 0], source[:, 1], 'ro', alpha=0.5)
    plt.title("Original")
    ax = plt.subplot(132)
    ax.plot(target[:, 0], target[:, 1], 'bo', alpha=0.5)
    ax.plot(source_transformed[:, 0], source_transformed[:, 1], 'ro', alpha=0.5)
    plt.title("ICP")
    ax = plt.subplot(133)
    ax.plot(target[:, 0], target[:, 1], 'bo', alpha=0.5)
    ax.plot(source_transformed_ransac[:, 0], source_transformed_ransac[:, 1], "ro", alpha=0.5)
    ax.plot(source_transformed_ransac[outlier_indices, 0], source_transformed_ransac[outlier_indices, 1], "kx")
    plt.title("ICP + RANSAC")

    plt.show()


if __name__ == "__main__":
    main()
