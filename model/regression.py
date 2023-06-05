import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from model.machine import Machine



class Regression(Machine):

    def split(self, train_scale: float, test_scale: float, seed: int = 0):

        if (train_scale <= 0) or (test_scale <= 0):
            raise("[SPLIT ERROR]: Contains negative or zero ratio")

        ratio: float = train_scale / (train_scale + test_scale)
        cut_pos: int = int(ratio * self.shape[0])

        np.random.seed(seed)
        id_list: np = np.random.permutation((self.shape[0]))

        self.df_train = self.df.loc[id_list[:cut_pos], :]
        self.df_test = self.df.loc[id_list[cut_pos:], :]



    def giveTheory(self, features, labels):

        self.schema_X: list = features
        self.schema_y: list = labels

        print(self.schema_X, '\t', self.schema_y)

        self.X_train: np = self.df_train.loc[:, features].to_numpy()
        self.y_train: np = self.df_train.loc[:, labels].to_numpy()
        self.X_test: np = self.df_test.loc[:, features].to_numpy()
        self.y_test: np = self.df_test.loc[:, labels].to_numpy()

        if len(np.shape(self.y_train)) == 1:
            self.y_train = np.reshape(self.y_train, (-1, 1))
            self.y_test = np.reshape(self.y_test, (-1, 1))



    def charge(self, file_name: str = "", key: str = ""):
        pass



    def train(self, active_func, hddn: list = [], learning_rate: int = 0.1, max_step: int = 100):

        # Toggle size
        size_X = np.shape(self.X_train)
        size_y = np.shape(self.y_train)

        # Make all matrix to support the ANN
        W = [None for _ in range(len(hddn) + 1)]
        B = [None for _ in range(len(hddn) + 1)]
        Y = [None for _ in range(len(hddn) + 2)]     # Two

        s_iter = [ *[ size_X[1] ], *[ hddn[i] for i in range(len(hddn)) ], *[ size_y[1] ] ]
        for i in range(len(W)):
            W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
            B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))

        # For error
        y_predicted = np.zeros(size_y[0])
        error_points = np.zeros(max_step)

        # Start running
        step = 0
        while step < max_step:

            mix = np.random.permutation(size_X[0])
            for i in mix:

                # Y[0] = X      Z[0] = Y[0] * W[0] + B[0]       Y[1] = activeFunc[0](Z[0])
                #               Z[1] = Y[1] * W[1] + B[1]       Y[2] = activeFunc[1](Z[1])
                #               Z[2] = Y[2] * W[2] + B[2]       Y[3] = activeFunc[2](Z[2])

                Y[0] = np.reshape(self.X_train[i], (1, size_X[1]))
                for j in range(len(hddn) + 1):
                    z        = np.dot(Y[j], W[j]) + B[j]
                    Y[j + 1] = np.array(active_func[j](z))

                y_predicted[i] = Y[-1]

                # dLoss/dW[2] = Y[2] * Error
                # dLoss/dW[1] = Y[1] * Error * W[2]
                # dLoss/dW[0] = Y[0] * Error * W[2] * W[1]

#                print(i, ') ', y_predicted[i], '\t', self.y_train[i])
                error = 2 / size_X[0] * (y_predicted[i] - self.y_train[i])
                for j in range(len(W) - 1, -1, -1):
                    multi = np.reshape(error, (1, 1))   if (j == len(W) - 1)   else np.dot(multi, W[j + 1].T)
                    W[j] = W[j] - learning_rate * np.dot(Y[j].T, multi)
                    B[j] = B[j] - learning_rate * multi

            try:
                rmse = mean_squared_error(self.y_train, y_predicted, squared = False)
                error_points[step] = rmse
                step += 1
#                if step % 5 == 0:
                print("Step ", step, " done with RMSE = ", rmse)
            except:
                for i in range(len(W)):
                    W[i] = np.random.normal(0, 1, size = (s_iter[i], s_iter[i + 1]))
                    B[i] = np.random.normal(0, 1, size = (1, s_iter[i + 1]))
                print("[ERROR]: RESET at step ", step, " you should reduce the learning rate")
                # step = 0
        
        plt.plot(np.arange(max_step), error_points)
        plt.show()

        self.W = W
        self.B = B
        self.active_func = active_func
        self.hddn = hddn




    def test(self):
        size_X = np.shape(self.X_train)
        self.y_pred: np = np.array([None for _ in range(size_X[0])])
        
        for i in range(size_X[0]):

            Y = [None for _ in range(len(self.hddn) + 2)]
            Y[0] = np.reshape(self.X_train[i], (1, size_X[1]))

            for j in range(len(self.hddn) + 1):
                z        = np.dot(Y[j], self.W[j]) + self.B[j]
                Y[j + 1] = np.array(self.active_func[j](z))
            
            self.y_pred[i] = Y[-1].tolist()[0]
        
        return self.y_pred



    def graph(self, x_title: str, y_title: str, write_arr: list):

        print(self.y_pred[:5])
        print(self.y_test[:5])
        print(type(self.y_pred[:5]))
        print(type(self.y_test[:5]))

        for write in write_arr:

            id_x: int = self.schema_X.index(x_title)
            id_y: int = self.schema_y.index(y_title)
            if (id_x == -1) or (id_y == -1):
                raise("!!!!!!!!!!!!!!!!!!!")
            

            if write == "both":
                plt.scatter(self.df.loc[:, x_title], self.df.loc[:, y_title], color = "black")

            if write == "train":
                plt.scatter(self.X_train[:, id_x], self.y_train[:, id_y], color = "blue")

            elif write == "test":
                plt.scatter(self.X_test[:, id_x], self.y_test[:, id_y], color = "orange")
            
            elif write == "pred":
                plt.scatter(self.X_test[:, id_x], self.y_pred[:, id_y], color = "red")

        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()



    def info(self, write: str):

        print_type: str = ""
        parameter = None

        if write == "train":
            print_type = "TRAIN"
            parameter = self.df_train.head()

        elif write == "test":
            print_type = "TEST"
            parameter = self.df_test.head()

        elif (write == "train_X") or (write == "X_train"):
            print_type = "TRAIN X"
            parameter = self.X_train[:5, :]

        elif (write == "train_y") or (write == "y_train"):
            print_type = "TRAIN Y"
            parameter = self.y_train[:5, :]
        
        elif (write == "test_X") or (write == "X_test"):
            print_type = "test X"
            parameter = self.X_test[:5, :]

        elif (write == "test_y") or (write == "y_test"):
            print_type = "test Y"
            parameter = self.y_test[:5, :]
        
        print(print_type, ":")
        print(" >  shape: ", parameter.shape)
        print(parameter)
        

