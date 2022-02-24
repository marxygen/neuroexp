"""ReLU activation function implementation"""
import numpy as np


def relu(x, deriv=False):
    """Calculate and return the result of applying ReLU activation function"""
    if deriv:
        act = np.zeros_like(x)
        act[x > 0] = 1
        return act

    act = x.copy()
    act[x <= 0] = 0
    return act
