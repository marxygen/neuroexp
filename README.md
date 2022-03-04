# neuroexp
This repo contains my experiments with neural networks.

## Goals
What I hope to get in the end is a Keras-like neural network constructor. Of course, not so functional and cool,
but I believe it would be enough for me to understand how Keras and neural networks work to give me advantage and
insights in NN design.

## Progress
Here's what I have so far:

#### Layers
- Dense

#### Neural Network architectures
- Sequential (as called in Keras)

#### Optimizers
- SGD

#### Loss functions
- L1
- L2

#### Activation functions
- Linear
- ReLU
- Sigmoid
- TanH

#### Features
- Learning rate decay
- Momentum
- Performance tracking

## Example
Here's a simple neural net built using my NN constructor (imports are omitted)
```Python
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

```
You can also call `network.visualize()` to display graphs indicating loss change and predictions comparison.