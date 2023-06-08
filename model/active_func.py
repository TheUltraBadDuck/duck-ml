import numpy as np



def Linear(x):
    return x

def ReLU(x):
    size = np.shape(x)
    for i in range(size[0]):
        for j in range(size[1]):
            if x[i][j] < 0:
                x[i][j] = 0
    return x

def LeakyReLU(x, scale: float = 0.1):
    size = np.shape(x)
    for i in range(size[0]):
        for j in range(size[1]):
            if x[i][j] < 0:
                x[i][j] *= scale
    return x

def Sinh(x):
    return np.sinh(x)



def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(float)))

def Softmax(x):
    e_z = np.exp(x)
    return e_z / np.sum(e_z)



