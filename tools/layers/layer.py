import numpy as np
from typing import Union
from importlib import import_module
from abc import abstractmethod


class Layer(object):
    weights: np.array
    biases: np.array
    product: np.array
    activated = np.array

    def __init__(self, neurons: int, inputs: int, activation: Union[str, callable], **kwargs):
        """Create a new Layer

        Args:
            neurons (int): Number of neurons
            inputs (int): Number of inputs each neuron has
        """
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.zeros(neurons)
        self.activation = (getattr(import_module(f'activations.{activation}'), activation)
                           if not callable(activation) else activation
                           )
        self.activation_deriv = (getattr(import_module(f'activations.{activation}'), activation + '_deriv')
                           if not callable(activation) else kwargs['activation_deriv']
                           )
        @abstractmethod
        def forward(self, *args, **kwargs):
            ...

        @abstractmethod
        def backwards(self, *args, **kwargs):
            ...
