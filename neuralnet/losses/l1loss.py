"""L1 Loss"""
import numpy as np


def l1loss(predicted, targets, deriv=False):
    """Calculate and return L1 Loss (Mean Absolute Error)"""
    predicted = predicted.astype(np.longfloat)
    targets = targets.astype(np.longfloat)

    if deriv:
        return (predicted-targets)/np.abs(predicted - targets)
    return np.abs(targets - predicted)
