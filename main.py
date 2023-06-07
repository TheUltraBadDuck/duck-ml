from model.regression import RegressionGraph, RegressionSoftmax
from model.active_func import *



if __name__ == "__main__":

    # model = Regression("a0.05_b0_c-3_d0.csv")

    # model.modify("X_coord", lambda x: x / 10)
    # model.modify("y_coord", lambda y: y / 20 + 1)

    # model.split(0.5, 0.5)
    # model.giveTheory(["X_coord"], ["y_coord"])

    # model.train([lambda x: x ** 3 - 3 * x, LeakyReLU], [6], 0.01, 1500, print_error=False)
    # model.test()

    # model.graph("X_coord", "y_coord", ["test", "pred"])

    model = RegressionSoftmax("Heart.csv")

    model.modify("RestBP", lambda x: x / 10 - 1)
    model.modify("Age", lambda x: x / 5)
    model.modify("AHD", lambda x: 1 if x == "Yes" else 0)
    
    model.split(0.5, 0.5)
    model.giveTheory(["RestBP", "Age"], ["AHD"])

    model.train([], 0.03, 2000, print_error=False)
    model.test()

    model.graph("RestBP", "AHD", ["test", "pred"])
    model.graph("Age", "AHD", ["test", "pred"])

    # model = RegressionSoftmax("logistic.csv")
    
    # model.split(0.5, 0.5, -1)
    # model.giveTheory(["X_coord"], ["y_coord"])

    # model.train([], 0.05, 200, print_error=False)
    # model.test()

    # model.graph("X_coord", "y_coord", ["test", "pred"])





# import numpy as np
# import csv
# import matplotlib.pyplot as plt

# data = np.zeros((200, 2), dtype=float)

# for i in range(200):
#     val = i * 0.1 - 10
#     data[i, 0] = val
#     data[i, 1] = 0.05 * val ** 3 - 3 * val + np.random.normal(0, 1)

# with open('a0.05_b0_c-3_d0.csv', 'w', newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["X_coord", "y_coord"])
#     for row in data:
#         writer.writerow(row)
#     f.close()

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()