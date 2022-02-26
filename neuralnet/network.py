"""Neural Network implementation"""
import numpy as np
from typing import List, Callable
from matplotlib import pyplot as plt
from utils.imports import import_function


class NeuralNetwork:
    """Neural Network class"""
    values: np.array
    targets: np.array
    layers: List['Layer']
    loss: Callable
    optimizer: 'Optimizer'

    def __init__(self, layers: list, optimizer, loss="l2loss"):
        """Initialize the neural network with layers as a list of instances"""
        self.values = None
        self.targets = None
        self.layers = layers
        self.loss = loss if callable(loss) else import_function(loss, "losses")
        self.optimizer = optimizer

    def predict(self, inputs: np.array):
        """Predict the outputs for given inputs"""
        value = inputs.copy()

        for layer in self.layers:
            value = layer.forward(value)

        return value

    def validate(self, predictions, targets, verbose=True):
        """Validate predicted against targets"""
        loss = self.loss(predictions, targets)
        if verbose:
            print(f"Loss: {loss:.5f}")
        return loss

    def forward(self, inputs, targets, verbose=True):
        """Perform forward pass"""
        self.values = self.predict(inputs)
        self.targets = targets

        loss = self.validate(self.values, targets, verbose)

        if verbose:
            print(f"Loss: {loss:.5f}")

    def backward(self, epoch=None, batch=None):
        """Perform backward pass"""
        # Loss derivative shows how much each neuron affect the function across samples
        # So, the dimensions of the loss_deriv must be neurons x samples
        loss_deriv = self.loss(
            self.values, self.targets, deriv=True
        ).T  # neurons x samples
        self.layers[-1].backward(
            dvalues=loss_deriv,
            next_layers=self.layers[:-1][::-1],
            optimizer=self.optimizer,
            epoch=epoch,
            batch=batch
        )

    def measure_error(self, sample: np.array, targets: np.array):
        """Measure and return the error"""
        return self.validate(
            self.predict(sample),
            targets,
            verbose=False).mean()

    def fit(
            self,
            inputs,
            targets,
            validation_split=0.1,
            epochs=100,
            batch_size=1000):
        """Display report based on performance of the network"""

        training_count = int(len(inputs) * (1 - validation_split))
        self.train_x = inputs[:training_count]
        self.train_y = targets[:training_count]

        self.test_x = inputs[training_count:]
        self.test_y = targets[training_count:]

        print('[Inputs]')
        print(f'\tMean: {inputs.mean():.5f}')
        print(f'\tMin: {inputs.mean():.5f}')
        print(f'\tMax: {inputs.mean():.5f}')

        print('[Targets]')
        print(f'\tMean: {targets.mean():.5f}')
        print(f'\tMin: {targets.mean():.5f}')
        print(f'\tMax: {targets.mean():.5f}')
        print()

        self.before_training = self.measure_error(
            self.test_x, self.test_y)
        print(f"Loss before training: {self.before_training:.5f}")

        self.epochs_loss_change = []

        for epoch in range(epochs):
            for batch, (t_x, t_y) in enumerate(zip(np.array_split(
                    self.train_x, batch_size), np.array_split(self.train_y, batch_size))):
                self.forward(t_x, t_y, verbose=False)
                self.backward(epoch=epoch, batch=batch)
            loss = self.measure_error(self.test_x, self.test_y)
            self.epochs_loss_change.append(loss)
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.5f}", end="\r")

        self.after_training = self.measure_error(
            self.test_x, self.test_y)
        print(f"Loss after training: {self.after_training:.5f}")
        self.increase = (self.before_training - self.after_training) * \
            100 / self.before_training
        print(
            f'Performance: {abs(self.increase):.3f}% {"better" if self.increase > 0 else "worse"}'
        )

    def visualize(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.epochs_loss_change)
        ax1.set_title(
            f"Loss change across epochs (LR {self.optimizer.learning_rate}). Overall increase: {self.increase:.5f}%"
        )

        ax2.plot(self.train_x, self.train_y, c="g", label="correct")
        predictions = self.predict(self.train_x)
        ax2.plot(
            self.train_x,
            predictions,
            c="r",
            label="predicted",
            scalex=False,
            scaley=False)
        ax2.set_title(
            f"Comparison (Avg diff {abs(predictions.mean() - self.test_y.mean()):5f}, Loss {self.after_training:.5f})"
        )
        ax2.legend()
        plt.show()
