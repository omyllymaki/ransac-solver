from samples.icp_fit.data import Data
from samples.icp_fit.icp.icp import icp, transform, nearest_neighbor
from src.model import BaseModel


class Model(BaseModel):

    def __init__(self):
        self.transform = None

    def fit(self, data: Data):
        self.transform = icp(data.source.copy(),
                             data.target.copy(),
                             init_transform=None,
                             max_iterations=100,
                             cost_diff_threshold=0.0001)[0]

    def estimate(self, data: Data):
        return transform(self.transform, data.source.copy())

    def calculate_errors(self, data: Data):
        source_transformed = self.estimate(data)
        distances, _ = nearest_neighbor(source_transformed, data.target)
        return distances
