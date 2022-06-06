from abc import ABC

import numpy as np


class BaseLayer(ABC):
    """Base class for a network layer"""
    layer: np.array
    
    def __init__(self, neurons: int, inputs: int):
        """Initialize a new neural network layer.

        Args:
            neurons (int): Number of neurons in the layer
            inputs (int): Number of inputs for each neuron in the layer
        """
        # Generate an array of neurons (inputs x number of neurons)
        self.layer = np.random.rand(inputs, neurons)
    