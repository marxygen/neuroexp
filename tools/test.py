from neural_network import NeuralNetwork
from dense import Dense
import numpy as np

network = NeuralNetwork([
    Dense(3, 1,'relu'),
    Dense(3, 5,'relu'),
    Dense(5, 5,'relu'),
    Dense(5, 1,'softmax'),
])
