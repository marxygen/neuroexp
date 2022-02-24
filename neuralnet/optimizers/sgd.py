"""SGD implementation"""
from typing import Tuple
import numpy as np


class SGD:
    """Stochastic gradient descent optimizer"""

    ALLOWED_DECAY_BASES = ["epoch", "batch"]

    def __init__(
            self,
            learning_rate: float,
            lr_decay_rate: float = None,
            decay_basis="epoch"):
        """Initialize the optimizer"""
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        if decay_basis not in self.ALLOWED_DECAY_BASES:
            raise ValueError(
                f'Basis "{decay_basis}" is not allowed. Choose from {", ".join(self.ALLOWED_DECAY_BASES)}'
            )
        self.decay_basis = decay_basis

    def apply(
        self,
        *,
        weights: np.array,
        dweights: np.array,
        biases: np.array,
        dbiases: np.array,
        **kwargs,
    ) -> Tuple[np.array, np.array]:
        """Apply the optimizer on weights and biases or any other parameters specified"""
        # Apply LR decay if requested
        learning_rate = (
            self.learning_rate
            * 1
            / (1 + kwargs.pop(self.decay_basis + "s", 1) * self.lr_decay_rate)
            if self.lr_decay_rate
            else self.learning_rate
        )
        weights -= (learning_rate * dweights).T
        biases -= (learning_rate * dbiases).T

        return weights, biases
