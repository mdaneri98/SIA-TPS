import numpy as np


class OjaPerceptron:

    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size)

    def predict(self, x_value):
        return np.dot(self.weights, x_value)

    def train(self, x_values, epochs):
        for _ in range(epochs):
            for i, _ in enumerate(x_values):
                y = self.predict(x_values[i])

                # Oja
                delta_w = self.learning_rate * ((y * x_values[i]) - (y * y * self.weights))
                self.weights = self.weights + delta_w
        return self.weights
