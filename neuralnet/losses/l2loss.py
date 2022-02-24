"""L2 Loss (MSE)"""
import numpy as np


def l2loss(predicted, targets, deriv=False):
    """Calculate and return L2 Loss (Mean Squared Error)"""
    if deriv:
        return -2 * predicted * (targets - predicted)
    return np.square(targets - predicted)
