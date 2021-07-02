import numpy as np
from sklearn import clone
from sklearn.neighbors import KNeighborsClassifier

from src.datas.data_xy import Data
from src.model import BaseModel


class Model(BaseModel):

    def __init__(self, model):
        self.base_model = model

    def fit(self, data: Data):
        self.model = clone(self.base_model)
        self.model.fit(data.x, data.y)

    def calculate_errors(self, data: Data):
        prob = self.model.predict_proba(data.x)
        prob_for_labels = prob[np.arange(len(prob)), data.y]
        errors = 1 - prob_for_labels
        return errors