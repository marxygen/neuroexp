import numpy as np


def categorical_crossentropy(X: np.array, Y: int, *args, **kwargs):
    X = np.clip(X, 1e-7, 1 - 1e-7)
    m = Y.shape[0]
    log_likelihood = -np.log(X[range(m), Y])
    loss = np.sum(log_likelihood) / m
    return loss


def categorical_crossentropy_deriv(X, Y):
    X = np.clip(X, 1e-7, 1 - 1e-7)
    return -Y / X
