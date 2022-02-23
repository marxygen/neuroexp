import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork(object):
    def __init__(self, layers: list, learning_rate: float, modifier: float = 1.0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.modifier = modifier
        
    @staticmethod
    def l2loss(predicted, targets, deriv=False):
        if deriv:
            return -2*predicted*(targets-predicted)
        return np.square(targets-predicted)
        
    def predict(self, inputs: np.array):
        """Predict the outputs for given inputs"""
        value = inputs.copy()
        
        for layer in self.layers:
            value = layer.forward(value)
            
        return value * self.modifier
    
    def validate(self, predictions, targets, verbose=True):
        """Validate predicted against targets"""
        loss = self.l2loss(predictions, targets)
        if verbose:
            print(f'MSE: {loss:.5f}')
        return loss
    
    def forward(self, inputs, targets, verbose=True):
        """Perform forward pass"""
        self.values = self.predict(inputs)
        self.targets = targets
        
        loss = self.validate(self.values, targets, verbose)
        
    def backward(self):
        """Perform backward pass"""
        # Loss derivative shows how much each neuron affect the function across samples
        # So, the dimensions of the loss_deriv must be neurons x samples
        loss_deriv = self.l2loss(self.values, self.targets, deriv=True).T # neurons x samples
        self.layers[-1].backward(dvalues=loss_deriv,
                                next_layers=self.layers[:-1][::-1],
                                learning_rate=self.learning_rate)

    def measure_error(self, sample: np.array, targets: np.array):
        return self.validate(
            self.predict(sample),
            targets,
            verbose=False).mean()

    def fit(self, inputs, targets, validation_split=0.1, epochs=100):
        """Display report based on performance of the network"""

        training_count = int(len(inputs) * (1-validation_split))
        train_x = inputs[:training_count]
        train_y = targets[:training_count]

        test_x = inputs[training_count:]
        test_y = targets[training_count:]

        before_training = self.measure_error(test_x, test_y)
        print('MSE before training:', before_training)

        epochs_mse_change = []

        for epoch in range(epochs):
            print(f'Epoch {epoch},', end=' ')
            for t_x, t_y in zip(np.array_split(train_x, 1000), np.array_split(train_y, 1000)):
                self.forward(t_x, t_y, verbose=False)
                self.backward()
            mse = self.measure_error(self.predict(t_x), t_y)
            epochs_mse_change.append(mse)
            print(f'MSE: {mse:.5f}', end='\r')

        after_training = self.measure_error(test_x, test_y)
        print(f'MSE after training: {after_training}')
        increase = (before_training-after_training)*100/before_training
        print(f'Performance: {abs(increase):.3f}% {"better" if increase > 0 else "worse"}')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(test_x, test_y)
        ax1.set_title(f'Correct values for testing')
        ax2.plot(test_x, self.predict(test_x))
        ax2.set_title(f'Predicted values for testing')

        ax3.plot(epochs_mse_change)
        ax3.set_title(f'MSE change across epochs (LR {self.learning_rate})')

        ax4.plot(train_x, train_y, c='r', label='correct')
        ax4.plot(train_x, self.predict(train_x), c='b', label='predicted')
        ax4.set_title('Comparison')
        ax4.legend()
        plt.show()