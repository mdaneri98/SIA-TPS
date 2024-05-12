import numpy as np
from Activation import Activation


class Neuron:
    def __init__(self, weight_num, activation: Activation, learn_rate):
        self.weights = np.random.uniform(-1, 1, weight_num + 1)
        self.output = 0
        self.delta = 0
        self.output_dx = 0
        self.activation = activation
        self.learn_rate = learn_rate

    def calculate_gradient(self, inputs: np.ndarray, error: np.ndarray):
        inputs = np.append(inputs, 1)  # Agregar bias
        self.delta = self.output_dx * error
        gradient = self.delta * inputs  # Element-wise multiplication
        return gradient

    def calculate_output(self, inputs: np.ndarray):
        # Suma bias
        inputs = np.append(inputs, 1)

        # Similar a .dot pero mejor para manejar matrices multidimensionales.
        e = np.inner(inputs, self.weights)

        self.output = self.activation.apply(e)
        self.output_dx = self.activation.apply_dx(e)

    def update_w(self, inputs: np.ndarray, error: np.ndarray):
        inputs = np.append(inputs, 1)
        self.delta = self.output_dx * error
        for i in range(0, len(self.weights)):
            self.weights[i] += self.learn_rate * self.delta * inputs[i]

    def plot(self):
        print("{ Pesos: " + str(self.weights) + " Delta: " + str(self.delta) + " Output: " + str(self.output) + " }\n")
