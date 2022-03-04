import numpy as np


def linear(x, deriv=False):
    if deriv:
        return np.ones_like(x)

    return x
