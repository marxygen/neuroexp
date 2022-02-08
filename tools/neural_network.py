import numpy as np
from typing import Union
from importlib import import_module


class NeuralNetwork(object):
    def __init__(self,
                 loss: Union[str, callable],
                 layers: list,
                 learning_rate=0.01,
                 **kwargs
                 ):
        """Create a neural network

        Args:
            layers (list): List of neural network layers
        """
        self.layers = layers
        self.loss = (getattr(import_module(f'losses.{loss}'), loss)
                     if not callable(loss) else loss
                     )
        self.loss_deriv = (getattr(import_module(f'losses.{loss}'), loss + '_deriv')
                           if not callable(loss) else kwargs['loss_deriv']
                           )
        self.learning_rate = learning_rate

    def perform_forward(self, inputs: np.array, desired_outputs: np.array):
        """Perform forward pass"""
        data = inputs

        for layer in self.layers:
            data = layer.forward(data)

        loss = self.loss(X=data, Y=desired_outputs)
        print('Loss:', loss)
        return data

    def perform_backward(self, inputs, desired_outputs):
        """Perform backward pass"""
        # Initialize with the derivative of loss function
        dvalue = self.loss_deriv(X=inputs, Y=desired_outputs)

        self.layers[0].backwards(dvalue=dvalue, learning_rate=self.learning_rate, next=self.layers[1:])
