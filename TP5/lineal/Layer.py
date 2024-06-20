import numpy as np
from Optimazer import AdamOptimizer
from Optimazer import GradientDescentOptimizer


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_derivative):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size, learning_rate=0.001, optimizer_type=None):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.learning_rate = learning_rate
        self.time_step = 0
        if optimizer_type == "ADAM":
            self.optimizer = AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = GradientDescentOptimizer(self.learning_rate)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        self.time_step += 1

        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= self.optimizer.update(weights_gradient, self.time_step)
        self.bias -= self.optimizer.update(bias_gradient, self.time_step)

        return input_gradient
