
class Layer(object):
    def __init__(self, neurons: int, inputs: int):
        """Create a new Layer

        Args:
            neurons (int): Number of neurons
            inputs (int): Number of inputs each neuron has
        """
        # Weights are generated in the required shape
        self.weights = np.random.randn((inputs, neurons))
