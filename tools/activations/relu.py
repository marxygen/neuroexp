import numpy as np


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def relu_deriv(x: np.array):
    x = x.copy()
    x[x <= 0] = 0
    return x
