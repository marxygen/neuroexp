"""Sigmoid activation function implementation"""
import numpy as np


def sigmoid(x: np.array, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
