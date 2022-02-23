import numpy as np
from network import NeuralNetwork
from layers.dense import Dense
from math import pi

def f(x):
    return x * 2 - 3

inputs = np.array(np.linspace(-10000, 10000, 10000)) / 10000
inputs = inputs.reshape(len(inputs), 1)
targets = f(inputs) / 10000

print(f'Mean input: {inputs.mean():.5f}')
print(f'Mean output: {targets.mean():.5f}')
print()

network = NeuralNetwork(
    layers=[
        Dense(neurons=1, inputs=1),
        Dense(neurons=1, inputs=1),
        Dense(neurons=1, inputs=1)
    ],
    learning_rate = 0.7
)

network.fit(inputs,
            targets,
            validation_split=0.20,
            epochs=500)
