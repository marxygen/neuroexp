"""Model testing"""
import numpy as np
from network import NeuralNetwork
from layers.dense import Dense
from activations import sigmoid, relu
from losses import l2loss
from optimizers import SGD


def f(x):
    return 3*np.square(x) - 5


inputs = np.array(np.linspace(-1000, 1000, 2000))
inputs = inputs.reshape(len(inputs), 1)
targets = f(inputs)

max_scale = np.maximum(np.amax(inputs), np.amax(targets))

inputs /= max_scale
targets /= targets

network = NeuralNetwork(
    layers=[
        Dense(neurons=1, inputs=1, activation=sigmoid),
        Dense(neurons=4, inputs=1, activation=sigmoid),
        Dense(neurons=4, inputs=4, activation=sigmoid),
        Dense(neurons=1, inputs=4, activation=sigmoid)
    ],
    loss=l2loss,
    optimizer=SGD(learning_rate=0.001, lr_decay_rate=0.05, decay_basis='epoch')
)

network.fit(inputs,
            targets,
            validation_split=0.10,
            epochs=100)

network.visualize()