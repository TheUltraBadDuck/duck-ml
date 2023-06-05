from model.regression import Regression
from model.active_func import *



if __name__ == "__main__":
    model = Regression("Heart.csv")
#    model.numerize(["xedgey"])
    model.split(0.7, 0.3)

    model.giveTheory(["Age"], ["Chol"])

#     model.title(["x2ybar"], "xedgey")
#     model.train([ReLU, ReLU, ReLU],
# #                hddn=[, 6],
#                 learning_rate=0.1,
#                 max_step=100)

    model.train([Linear], [], 0.01, 25)

    model.test()
    model.graph("Age", "Chol", ["test", "pred"])