import numpy as np

from src.common_data_types.data_x import Data
from src.model import BaseModel


class Model(BaseModel):
    """
    Fit plane z = ax + by + c by minimising orthogonal distances.
    """

    def __init__(self):
        self.parameters = None

    def fit(self, data: Data):
        n_points = len(data)
        if n_points < 3:
            raise Exception("Not enough point to solve plane")
        elif n_points == 3:
            self.parameters = self._solve_plane_analytically(data.x)
        else:
            self.parameters = self._fit_plane(data.x)

    def calculate_errors(self, data: Data):
        """
        Errors are orthogonal distances to plane.
        """
        xyz1 = np.hstack((data.x, np.ones((len(data), 1))))
        abcd = np.insert(self.parameters, 2, -1).T
        return abs(xyz1 @ abcd) / np.linalg.norm(abcd[:-1])

    @staticmethod
    def _solve_plane_analytically(xyz):
        # These two vectors are in the plane
        v1 = xyz[2, :] - xyz[0, :]
        v2 = xyz[1, :] - xyz[0, :]

        # The cross product is a vector normal to the plane
        n = np.cross(v1, v2)

        # d = a * x + b * y + c * z
        a, b, c = n
        d = - np.dot(n, xyz[2, :])

        # z = c1 * x + c2 * y + c3
        c1 = -a / c
        c2 = -b / c
        c3 = -d / c
        return np.array([c1, c2, c3]).T

    @staticmethod
    def _fit_plane(xyz):
        """
        Singular value decomposition method.
        Minimizes orthogonal distances.
        Returns fitted coefficients of equation z = c1*x + c2*y + c3 as [c1, c2, c3]
        """

        # Find the average of points (centroid) along the columns
        xyz_mean = np.average(xyz, axis=0)

        # Create CX vector (centroid to point) matrix
        cx = xyz - xyz_mean

        # Singular value decomposition
        u, s, v = np.linalg.svd(cx)

        # The last row of V matrix is the eigenvector with smallest eigenvalue
        # This is normal of fitted plane
        n = v[-1]

        # Extract a, b, c, d coefficients
        # ax + by + cz + d = 0
        x0, y0, z0 = xyz_mean
        a, b, c = n
        d = -(a * x0 + b * y0 + c * z0)

        # z = c1 * x + c2 * y + c3
        c1 = -a / c
        c2 = -b / c
        c3 = -d / c
        return np.array([c1, c2, c3]).T