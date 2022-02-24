import numpy as np
from typing import Tuple


class SGD(object):
    """Stochastic gradient descent optimizer"""

    def __init__(self, learning_rate: float):
        """Initialize the optimizer"""
        self.learning_rate = learning_rate

    def apply(
        self,
        *,
        weights: np.array,
        dweights: np.array,
        biases: np.array,
        dbiases: np.array,
        **kwargs
    ) -> Tuple[np.array, np.array]:
        """Apply the optimizer on weights and biases or any other parameters specified"""
        weights -= (self.learning_rate * dweights).T
        biases -= (self.learning_rate * dbiases).T

        return weights, biases
