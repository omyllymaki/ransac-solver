import logging

import numpy as np
from sklearn import datasets, clone
from sklearn.neighbors import KNeighborsClassifier

from samples.classification.model import Model
from src.datas.data_xy import Data
from src.solvers.vanilla_ransac import RansacSolver

logging.basicConfig(level=logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.random.seed(42)

N_TEST = 500
N_MISLABELED = 800


def create_test_data():
    X, y = datasets.load_digits(return_X_y=True)

    indices_mislabeled = np.random.permutation(len(X))
    X_train = X[indices_mislabeled[:-N_TEST]]
    y_train = y[indices_mislabeled[:-N_TEST]]
    X_test = X[indices_mislabeled[-N_TEST:]]
    y_test = y[indices_mislabeled[-N_TEST:]]

    y_train_mislabeled = y_train.copy()
    indices_mislabeled = np.random.choice(len(y_train), N_MISLABELED, replace=False)
    original_labels = y_train[indices_mislabeled]
    mislabeled = np.random.choice(max(y_train), N_MISLABELED, replace=True)
    mislabeled[mislabeled == original_labels] = mislabeled[mislabeled == original_labels] - 1

    y_train_mislabeled[indices_mislabeled] = np.random.choice(max(y_train), N_MISLABELED, replace=True)

    return X_train, X_test, y_train, y_train_mislabeled, y_test, indices_mislabeled


def calculate_test_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy


def main():
    X_train, X_test, y_train, y_train_mislabeled, y_test, indices_mislabeled = create_test_data()
    base_model = KNeighborsClassifier(n_neighbors=5)

    model = clone(base_model)
    model.fit(X_train, y_train)
    print("Test accuracy with original data:", calculate_test_accuracy(model, X_test, y_test))

    model = clone(base_model)
    model.fit(X_train, y_train_mislabeled)
    print("Test accuracy with some mislabeled data:", calculate_test_accuracy(model, X_test, y_test))

    data = Data(x=X_train, y=y_train_mislabeled)
    solver = RansacSolver(model=Model(model=clone(base_model)),
                          error_threshold=0.8,
                          n_sample_points=len(data) // 5,
                          max_trials=500)
    inlier_indices = solver.fit(data)
    print("Test accuracy with some mislabeled data using RANSAC:", calculate_test_accuracy(solver.model.model,
                                                                                           X_test,
                                                                                           y_test))

    n_points_used = len(inlier_indices)
    n_mislabeled = len(set(indices_mislabeled) & set(inlier_indices))
    n_correctly_labeled = n_points_used - n_mislabeled
    print(f"RANSAC fit used {n_points_used} points out of {X_train.shape[0]} points for fitting")
    print(f"{n_correctly_labeled} of the points were labeled correctly and {n_mislabeled} were mislabeled")


if __name__ == "__main__":
    main()
