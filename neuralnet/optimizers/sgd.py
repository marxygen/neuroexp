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
            decay_basis="epoch",
            momentum=0
    ):
        """Initialize the optimizer"""
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        if decay_basis not in self.ALLOWED_DECAY_BASES:
            raise ValueError(
                f'Basis "{decay_basis}" is not allowed. Choose from {", ".join(self.ALLOWED_DECAY_BASES)}'
            )
        self.decay_basis = decay_basis
        self.momentum = momentum

    def apply(
            self,
            *,
            weights: np.array,
            dweights: np.array,
            biases: np.array,
            dbiases: np.array,
            **kwargs,
    ) -> Tuple[np.array, np.array, np.array]:
        """Apply the optimizer on weights and biases or any other parameters specified"""
        # Apply LR decay if requested
        # learning_rate = self.learning_rate * 1 / \
        #     (1 + kwargs.pop(self.decay_basis + "s",
        #                     1) * self.lr_decay_rate) if self.lr_decay_rate else self.learning_rate
        #
        # weight_momentums = kwargs.pop('weight_momentums')
        # weight_updates = self.momentum * weight_momentums - \
        #     (self.learning_rate * dweights).T
        learning_rate = self.learning_rate
        weight_updates = (self.learning_rate * dweights).T

        weights -= weight_updates
        biases -= (learning_rate * dbiases).T

        return weights, biases, weight_updates
