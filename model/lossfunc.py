import numpy as np



def RMSE(y, y_hat):
    dist: np = y - y_hat
    total: float = np.sum(np.multiply(dist, dist))
    return np.sqrt(total / y.shape[0])


def MSE(y, y_hat):
    dist: np = y - y_hat
    total: float = np.sum(np.multiply(dist, dist))
    return total / y.shape[0]


def CrossEntropy(y, y_hat):
    dist: np = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return -np.sum(dist)
    