import numpy as np

class Dense(object):
    def sigmoid(self, x: np.array, deriv=False):
        if deriv:
            return self.sigmoid(x) * (1-self.sigmoid(x))
        return 1/(1 + np.exp(-x))
        
    def __init__(self, *, neurons: int, inputs: int, activation='relu'):
        self.neurons_num = neurons
        self.inputs_num = inputs
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.random.randn(1, neurons)
        
    def forward(self, inputs):
        self.inputs = inputs
        self.values = inputs @ self.weights + self.biases
        self.outputs = self.sigmoid(x=self.values)
        return self.outputs
    
    def backward(self, dvalues, next_layers: list, learning_rate):
        # We received dvalues - its dimensions are neurons x samples
        # Now we have to calculate the derivative of activation function
        # Its dimensions are neurons x samples
        dsigm = self.sigmoid(x=self.values, deriv=True).T
        dsigm = dvalues * dsigm

        # How much does each input affect the output of the neuron
        # This will be sent to the next layer
        dinputs = np.dot(self.weights, dsigm)
        # How much does the change in weight affect the output of the neuron
        dweights = np.dot(dsigm, self.inputs)
        dbiases = dsigm.sum(axis=1, keepdims=True)
        
        self.weights -= (learning_rate * dweights).T
        self.biases -= (learning_rate * dbiases).T
        
        if next_layers:
            next_layers[0].backward(dvalues=dinputs, next_layers=next_layers[1:], learning_rate=learning_rate)
