import numpy as np
import pandas as pd



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


class ConfusionMatrix():
    def __init__(self):
        self.tp: int = 0
        self.tn: int = 0
        self.fp: int = 0
        self.fn: int = 0
    
    def addOne(self, type: str):
        if (type == "tp") or (type == "true positive"):
            self.tp += 1
        elif (type == "tn") or (type == "true negative"):
            self.tn += 1
        elif (type == "fp") or (type == "false positive"):
            self.fp += 1
        elif (type == "fn") or (type == "false negative"):
            self.fn += 1
    
    def accuracy(self):
        S: int = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / S

    def precision(self):
        return self.tp / (self.tp + self.fp)
    
    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    def matrix(self):
        df = pd.DataFrame({
            "predicted true": [self.tp, self.fp],
            "predicted false": [self.fn, self.tn]
        })
        return df

