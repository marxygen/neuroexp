import numpy as np
from matplotlib import pyplot as plt
from utils.imports import import_function


class NeuralNetwork(object):
    def __init__(self, layers: list, optimizer, loss="l2loss"):
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
            print(f'Loss: {loss:.5f}')

    def backward(self):
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
        )

    def measure_error(self, sample: np.array, targets: np.array):
        return self.validate(self.predict(sample), targets, verbose=False).mean()

    def fit(self, inputs, targets, validation_split=0.1, epochs=100):
        """Display report based on performance of the network"""

        training_count = int(len(inputs) * (1 - validation_split))
        train_x = inputs[:training_count]
        train_y = targets[:training_count]

        test_x = inputs[training_count:]
        test_y = targets[training_count:]

        before_training = self.measure_error(test_x, test_y)
        print(f"Loss before training: {before_training:.5f}")

        epochs_loss_change = []

        for epoch in range(epochs):
            print(f"Epoch {epoch},", end=" ")
            for t_x, t_y in zip(
                np.array_split(train_x, 1000), np.array_split(train_y, 1000)
            ):
                self.forward(t_x, t_y, verbose=False)
                self.backward()
            loss = self.measure_error(self.predict(t_x), t_y)
            epochs_loss_change.append(loss)
            print(f"Loss: {loss:.5f}", end="\r")

        after_training = self.measure_error(test_x, test_y)
        print(f"Loss after training: {after_training:.5f}")
        increase = (before_training - after_training) * 100 / before_training
        print(
            f'Performance: {abs(increase):.3f}% {"better" if increase > 0 else "worse"}'
        )

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(test_x, test_y)
        ax1.set_title(f"Correct values for testing")
        ax2.plot(test_x, self.predict(test_x))
        ax2.set_title(f"Predicted values for testing")

        ax3.plot(epochs_loss_change)
        ax3.set_title(
            f"Loss change across epochs (LR {self.learning_rate}). Overall increase: {increase:.5f}%"
        )

        ax4.set_ymargin(2.5)
        ax4.plot(test_x, test_y, c="g", label="correct")
        predictions = self.predict(test_x)
        ax4.plot(
            test_x, predictions, c="r", label="predicted", scalex=False, scaley=False
        )
        ax4.set_title(
            f"Comparison (Avg diff {abs(predictions.mean() - test_y.mean()):5f}, Loss {after_training:.5f})"
        )
        ax4.legend()
        plt.show()
