import numpy as np
from network import NeuralNetwork
from dense import Dense
from math import pi

def f(x):
    return x * 2 - 3

inputs = np.array(np.linspace(-10000, 10000, 10000))
inputs = inputs.reshape(len(inputs), 1)
targets = f(inputs)

inputs /= np.amax(inputs)
targets_scaled_by = np.amax(targets)
targets /= targets_scaled_by

print(f'Mean input: {inputs.mean():.5f}')
print(f'Mean output: {targets.mean():.5f}')
print(f'Output will be scaled by {targets_scaled_by:.5f}')
print()

network = NeuralNetwork(
    layers=[
        Dense(neurons=1, inputs=1),
        Dense(neurons=5, inputs=1),
        Dense(neurons=1, inputs=5)
    ],
    learning_rate = targets_scaled_by,
    modifier=targets_scaled_by
)

network.fit(inputs,
            targets,
            validation_split=0.05,
            epochs=500)
