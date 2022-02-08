from layers.layer import Layer
import numpy as np


class Dense(Layer):
    """A Dense layer object"""

    def forward(self, inputs: np.array):
        """Perform a forward pass"""
        self.inputs = inputs
        self.product = np.dot(inputs, self.weights) + self.biases
        self.activated = self.activation(self.product)
        return self.activated

    def backwards(self, dvalue, learning_rate, next):
        ...


