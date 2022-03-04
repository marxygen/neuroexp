import numpy as np


def tanh(x, deriv=False):
    if deriv:
        return 1 - (tanh(x) ** 2)
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
