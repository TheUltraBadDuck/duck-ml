import numpy as np



def ReLU(x):
    size = np.shape(x)
    for i in range(size[0]):
        for j in range(size[1]):
            if x[i][j] < 0:
                x[i][j] = 0
    return x

def Linear(x):
    return x

