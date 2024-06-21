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

    def backward(self, output_derivative):
        self.time_step += 1

        weights_gradient = np.dot(output_derivative, self.input.T)
        input_gradient = np.dot(self.weights.T, output_derivative)

        self.weights += self.optimizer.update(weights_gradient,self.time_step)

        # self.weights -= learning_rate * weights_gradient
        self.bias -= self.learning_rate * output_derivative

        return input_gradient
