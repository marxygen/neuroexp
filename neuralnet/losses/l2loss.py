import numpy as np


def l2loss(predicted, targets, deriv=False):
    if deriv:
        return -2 * predicted * (targets - predicted)
    return np.square(targets - predicted)
