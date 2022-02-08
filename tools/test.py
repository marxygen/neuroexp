from neural_network import NeuralNetwork
from layers.dense import Dense
import numpy as np

network = NeuralNetwork(
    loss='categorical_crossentropy',
    layers=[
        Dense(neurons=3, inputs=2, activation='relu'),
        Dense(neurons=3, inputs=3, activation='relu'),
    ])

input_data = np.array([
    [2, 2],
    [1, 3],
    [4, 5]
])
input_data = (input_data - np.min(input_data)) / np.ptp(input_data)

# 0 - first is even
# 1 - second is even
# 2 - both are even or None are even
answers = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [1, 0, 0]
])

data = network.perform_forward(input_data, answers)
network.perform_backward(data, answers)
