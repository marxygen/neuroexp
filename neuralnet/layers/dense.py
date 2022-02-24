"""Dense layer"""
import numpy as np
from utils.imports import import_function


class Dense:
    """Dense layer"""

    def __init__(self, *, neurons: int, inputs: int, activation="sigmoid"):
        """Instantiate a new Dense layer"""
        self.inputs = None
        self.values = None
        self.outputs = None
        self.neurons_num = neurons
        self.inputs_num = inputs
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.random.randn(1, neurons)
        self.activation = (
            activation
            if callable(activation)
            else import_function(activation, "activations")
        )

    def forward(self, inputs) -> np.array:
        """Perform forward pass"""
        self.inputs = inputs
        self.values = inputs @ self.weights + self.biases
        self.outputs = self.activation(x=self.values)
        return self.outputs

    def backward(
            self,
            dvalues,
            next_layers: list,
            optimizer,
            epoch=None,
            batch=None) -> None:
        """Perform backward pass"""
        # We received dvalues - its dimensions are neurons x samples
        # Now we have to calculate the derivative of activation function
        # Its dimensions are neurons x samples
        dact = self.activation(x=self.values, deriv=True).T
        dact = dvalues * dact

        # How much does each input affect the output of the neuron
        # This will be sent to the next layer
        dinputs = np.dot(self.weights, dact)
        # How much does the change in weight affect the output of the neuron
        dweights = np.dot(dact, self.inputs)
        dbiases = dact.sum(axis=1, keepdims=True)

        self.weights, self.biases = optimizer.apply(
            weights=self.weights, dweights=dweights, biases=self.biases, dbiases=dbiases,
            epoch=epoch, batch=batch
        )

        if next_layers:
            next_layers[0].backward(
                dvalues=dinputs,
                next_layers=next_layers[1:],
                optimizer=optimizer,
            )
