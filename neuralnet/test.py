"""Model testing"""
import numpy as np
from network import NeuralNetwork
from layers.dense import Dense
from optimizers import SGD
from scalers.minmax import min_max_scale


def f(x):
    return 3*(x**2) - 5


inputs = np.array(np.linspace(-1000, 1000, 10000))
inputs = inputs.reshape(len(inputs), 1)
targets = f(inputs)

inputs = min_max_scale(inputs)
targets = min_max_scale(targets)

network = NeuralNetwork(
    layers=[
        Dense(neurons=1, inputs=1, activation='sigmoid'),
        Dense(neurons=200, inputs=1, activation='sigmoid'),
        Dense(neurons=200, inputs=200, activation='sigmoid'),
        Dense(neurons=200, inputs=200, activation='sigmoid'),
        Dense(neurons=1, inputs=200, activation='sigmoid')
    ],
    loss='l1loss',
    optimizer=SGD(learning_rate=0.1, lr_decay_rate=0.001, decay_basis='epoch', momentum=0.1)
)

network.fit(inputs,
            targets,
            validation_split=0.10,
            epochs=10)

network.visualize()
