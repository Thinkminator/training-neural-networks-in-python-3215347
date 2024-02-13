import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        # np.random.rand(inputs+1): This generates an array of random numbers sampled from a uniform distribution between 0 and 1. 
        # The inputs + 1 indicates the number of weights needed. The +1 is typically for the bias neuron, which is why it's included.
        # * 2: This scales the random numbers to lie between 0 and 2.
        # - 1: This shifts the scaled random numbers to lie between -1 and 1. 
        # This is often done because neural networks typically initialize weights symmetrically around 0, 
        # which helps with convergence during training.
        self.weights = (np.random.rand(inputs+1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x, self.bias), self.weights)