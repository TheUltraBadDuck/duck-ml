import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.machine import Machine, Supervise
from model.active_func import *
from model.lossfunc import *



class RegressionGraph(Supervise):

    def __init__(self, r_file: str = "") -> None:
        Supervise.__init__(self, r_file)

    def train(self, active_func, hddn: list = [], learning_rate: int = 0.1, max_step: int = 100, print_error: bool = True):

        # Toggle size
        size_X: tuple = np.shape(self.X_train)
        size_y: tuple = np.shape(self.y_train)

        # Check too large value in the dataset:
        def havingLargeNumber() -> bool:
            for i in range(size_X[0]):
                for j in range(size_X[1]):
                    if abs(self.X_train[i][j]) > 25.0:
                        return True
            return False
        
        if havingLargeNumber():
            print("[WARNING]: Containing very large numbers. Please use the function modify to edit them")

        # Make all matrix to support the ANN
        W: list = [None for _ in range(len(hddn) + 1)]
        B: list = [None for _ in range(len(hddn) + 1)]
        Y: list = [None for _ in range(len(hddn) + 2)]     # Two

        s_iter = [ *[ size_X[1] ], *[ hddn[i] for i in range(len(hddn)) ], *[ size_y[1] ] ]
        for i in range(len(W)):
            W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
            B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))

        # For error
        y_hat: np = np.zeros(size_y[0])
        error_points: np = np.zeros(max_step)

        best_W: list = W
        best_B: list = B
        min_error: float = 32767.0

        # Start running
        step = 0
        while step < max_step:

            try:
                mix = np.random.permutation(size_X[0])
                for i in mix:

                    # Y[0] = X      Z[0] = Y[0] * W[0] + B[0]       Y[1] = activeFunc[0](Z[0])
                    #               Z[1] = Y[1] * W[1] + B[1]       Y[2] = activeFunc[1](Z[1])
                    #               Z[2] = Y[2] * W[2] + B[2]       Y[3] = activeFunc[2](Z[2])

                    Y[0] = np.reshape(self.X_train[i], (1, size_X[1]))
                    for j in range(len(hddn) + 1):
                        z        = np.dot(Y[j], W[j]) + B[j]
                        Y[j + 1] = np.array(active_func[j](z))

                    y_hat[i] = Y[-1]

                    # dLoss/dW[2] = Y[2] * Error
                    # dLoss/dW[1] = Y[1] * Error * W[2]
                    # dLoss/dW[0] = Y[0] * Error * W[2] * W[1]

    #                print(i, ') ', y_hat[i], '\t', self.y_train[i])
                    error = 2 / size_X[0] * (y_hat[i] - self.y_train[i])
                    for j in range(len(W) - 1, -1, -1):
                        multi = np.reshape(error, (1, 1))   if (j == len(W) - 1)   else np.dot(multi, W[j + 1].T)
                        W[j] = W[j] - learning_rate * np.dot(Y[j].T, multi)
                        B[j] = B[j] - learning_rate * multi

                error = RMSE(self.y_train, y_hat)
                error_points[step] = error
                step += 1
                if min_error > error:
                    best_W = W
                    best_B = B
                    min_error = error
                if print_error:
                    print("Step ", step, " done with RMSE = ", error)

            except:
                for i in range(len(W)):
                    W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
                    B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))
                if print_error:
                    print("[ERROR]: RESET at step ", step, " you should reduce the learning rate")
        
        if not print_error:
            plt.plot(np.arange(max_step), error_points)
            plt.show()

        self.W: list = best_W
        self.B: list = best_B
        self.active_func = active_func
        self.hddn: list = hddn


    def test(self):
        size_X = np.shape(self.X_test)
        self.y_pred: np = np.copy(self.y_test)
        
        for i in range(size_X[0]):

            Y = [None for _ in range(len(self.hddn) + 2)]
            Y[0] = np.reshape(self.X_test[i], (1, size_X[1]))

            for j in range(len(self.hddn) + 1):
                z        = np.dot(Y[j], self.W[j]) + self.B[j]
                Y[j + 1] = np.array(self.active_func[j](z))

            self.y_pred[i, :] = Y[-1][0]

        return self.y_pred




# ! WARNING:
# ! This class cannot be used for multi-label classification
# !
class RegressionSoftmax(Supervise):

    def __init__(self, r_file: str = "") -> None:
        Supervise.__init__(self, r_file)

    def train(self, hddn: list = [], learning_rate: int = 0.1, max_step: int = 100, print_error: bool = True):
        
        # Toggle size
        size_X: tuple = np.shape(self.X_train)
        size_y: tuple = np.shape(self.y_train)

        if size_y[1] > 1:
            raise("[ERROR]: This RegressionSoftmax class does not support multi-label classification yet!")

        # Check number of classes for classification
        def getNumberOfClasses():
            type_list: list = []
            for i in range(size_y[0]):
                if not self.y_train[i, 0] in type_list:
                    type_list.append(self.y_train[i, 0])
            return len(type_list)
        
        if getNumberOfClasses() > 2:
            print("COMING SOON")
            return

        # Make all matrix to support the ANN
        W: list = [None for _ in range(len(hddn) + 1)]
        B: list = [None for _ in range(len(hddn) + 1)]
        Y: list = [None for _ in range(len(hddn) + 2)]

        s_iter = [ *[ size_X[1] ], *[ hddn[i] for i in range(len(hddn)) ], *[ size_y[1] ] ]
        for i in range(len(W)):
            W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
            B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))

        # For error
        y_hat: np = np.zeros(size_y[0])
        error_points: np = np.zeros(max_step)

        best_W: list = W
        best_B: list = B
        min_error: float = 32767.0

        # Start running
        step = 0
        while step < max_step:

            #try:
                mix = np.random.permutation(size_X[0])
                for i in mix:

                    # Y[0] = X      Z[0] = Y[0] * W[0] + B[0]       Y[1] = Softmax(Z[0])
                    #               Z[1] = Y[1] * W[1] + B[1]       Y[2] = Softmax(Z[1])
                    #               Z[2] = Y[2] * W[2] + B[2]       Y[3] = Softmax(Z[2])

                    Y[0] = np.reshape(self.X_train[i], (1, size_X[1]))
                    for j in range(len(hddn) + 1):
                        z        = np.dot(Y[j], W[j]) + B[j]
                        Y[j + 1] = np.array(Sigmoid(z))

                    y_hat[i] = Y[-1]

                    # dLoss/dW[2] = (Y - Y[hat]) * Y[2]
                    # dLoss/dW[1] = (Y - Y[hat]) * W[2] * (Y[2] * (1 - Y[2])) * Y[1]
                    # dLoss/dW[0] = (Y - Y[hat]) * W[2] * (Y[2] * (1 - Y[2])) * W[1] * (Y[1] * (1 - Y[1])) * Y[0]

                    for j in range(len(W) - 1, -1, -1):
                        multi = np.reshape(self.y_train[i] - y_hat[i], (1, 1))   if (j == len(W) - 1)   else np.dot(multi, W[j + 1].T)
                        W[j] = W[j] + learning_rate * np.dot(Y[j].T, multi)
                        B[j] = B[j] + learning_rate * multi

                error = CrossEntropy(self.y_train, y_hat)
                error_points[step] = error
                step += 1
                if min_error > error:
                    best_W = W
                    best_B = B
                    min_error = error
                if print_error:
                    print("Step ", step, " done with Cross Log = ", error)

            # except:
            #     for i in range(len(W)):
            #         W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
            #         B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))
            #     if print_error:
            #         print("[ERROR]: RESET at step ", step, " you should reduce the learning rate")
            #         step += 1
        
        if not print_error:
            plt.plot(np.arange(max_step), error_points)
            plt.show()

        self.W: list = best_W
        self.B: list = best_B
        self.hddn: list = hddn



    def test(self):
        size_X = np.shape(self.X_test)
        self.y_pred: np = np.zeros((self.y_test.shape[0], 1), dtype=float)
        
        for i in range(size_X[0]):

            Y = [None for _ in range(len(self.hddn) + 2)]
            Y[0] = np.reshape(self.X_test[i], (1, size_X[1]))

            for j in range(len(self.hddn) + 1):
                if i < 3:
                    print('AT i = ', i)
                    print('>   ', Y[j])
                    print('>   ', self.W[j])
                    print('>   ', self.B[j])
                z        = np.dot(Y[j], self.W[j]) + self.B[j]
                Y[j + 1] = np.array(Sigmoid(z))
                if i == 0:
                    print('>   ', z)
                    print()
        
        return self.y_pred



class KNearestNeighbor(Supervise):

    def __init__(self, r_file: str = "") -> None:
        Supervise.__init__(self, r_file) 
    
    def predict(self, k_value: int = 3):

        if k_value % 2 == 0:
            raise("[ERROR]: K value is not odd")
        
        size_X: tuple = np.shape(self.X_train)
        size_y: tuple = np.shape(self.y_train)

        self.y_pred: np = np.zeros((self.y_test.shape[0], 1))
        
        for row in range(len(self.X_test)):
            dist_arr: np = np.zeros((size_X[0], 2), dtype=[("id", int), ("dist", float)])
            for i in range(size_X[0]):
                dist_arr[i, 0] = i
                dist_arr[i, 1] = np.linalg.norm(self.X_test[row, :] - self.X_train[i, :])
            dist_arr = np.sort(dist_arr, order=["dist"])[::-1]

            type_count_dict: dict = {}
            for k in range(k_value):
                #print(dist_arr[k, 0][0])
                #print(self.y_train[:5, :])
                label = self.y_train[dist_arr[k, 0][0]]
                if not label in type_count_dict:
                    type_count_dict[self.y_train[i]] = 1
                else:
                    type_count_dict[self.y_train[i]] += 1
            
            print(type_count_dict)
            
            self.y_pred[row] = max(type_count_dict, key=type_count_dict.get)





