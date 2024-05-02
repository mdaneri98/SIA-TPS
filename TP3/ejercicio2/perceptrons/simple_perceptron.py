import math
import numpy as np
import sys
from abc import ABC, abstractmethod


class SimplePerceptron(ABC):

    def __init__(self, dim: int, learning_rate: float, limit: int,
                 eps: float, activation_min: float = None, activation_max: float = None):
        self.dim = dim
        self.learning_rate = learning_rate
        self.limit = limit

        # Debe completar la superclase
        self.activation_max = activation_max
        self.activation_min = activation_min

        self.eps = eps
        self.w = self.initialize_weights(dim + 1)  # Contamos el w0.

    def initialize_weights(self, n):
        return np.random.rand(n)

    @abstractmethod
    def activation_function(self, x: float) -> float:
        pass

    @abstractmethod
    def activation_derivative(self, x: float) -> float:
        pass

    def compute_error(self, x_values: list[list[float]], expected_values: list[float]):
        obtained_values = self.compute_for_values(x_values)  # Valores obtenidos por el perceptrón.

        total_error = 0
        for mu in range(0, len(expected_values)):
            scaled_expected = self.activation_function(expected_values[mu])
            total_error += (scaled_expected - obtained_values[mu]) ** 2
        return total_error / 2

    def compute_for_values(self, x_values: list[list[float]]):
        return [self.activation_function(self.compute_excitement(x)) for x in x_values]

    def compute_excitement(self, x: list[float]) -> float:
        extended_x = np.array([1] + x)
        return np.dot(extended_x, self.w)

    def delta_weights(self, excitement, expected, obtained, x_mu):
        x_mu = np.array([1] + x_mu)
        return self.learning_rate * (expected - obtained) * self.activation_derivative(excitement) * x_mu

    def normalize_value(self, x, new_max, new_min):
        """ Normaliza una entrada usando el mínimo y máximo utilizado en el entrenamiento. """
        return ((x - new_min) / (new_max - new_min)) * (self.activation_max - self.activation_min) + self.activation_min

    def train(self, x_train_set: list[list[float]], expected_train_values: list[float], x_test_set: list[list[float]],
              expected_test_values: list[float], scale: bool = False):
        min_error = sys.maxsize
        epoch = 0

        if scale:
            expected_train_values = [self.normalize_value(y, min(expected_train_values), max(expected_train_values)) for
                                     y in
                                     expected_train_values]
            expected_test_values = [self.normalize_value(y, min(expected_test_values), max(expected_test_values)) for
                                    y in
                                    expected_test_values]

        train_errors = []
        test_errors = []

        while min_error > self.eps and epoch < self.limit:
            mu = np.random.randint(len(x_train_set))
            x_mu = x_train_set[mu]
            expected_mu = expected_train_values[mu]

            h_mu = self.compute_excitement(x_mu)
            o_mu = self.activation_function(h_mu)

            delta_w = self.delta_weights(h_mu, expected_mu, o_mu, x_mu)
            self.w = self.w + delta_w

            # Computamos error de train set con la nueva w
            error = self.compute_error(x_train_set, expected_train_values)
            train_errors.append(error)

            # Computamos error de test set con la nueva w
            test_errors.append(self.compute_error(x_test_set, expected_test_values))

            if error < min_error:
                min_error = error

            epoch += 1

        return epoch, train_errors, test_errors

    def predict(self, x_values: list[list[float]], expected_values: list[float]) -> tuple[list[float], float]:
        result = []
        for x_mu in x_values:
            h_mu = self.compute_excitement(x_mu)
            o_mu = self.activation_function(h_mu)
            result.append(o_mu)

        test_mse = self.compute_error(x_values, expected_values)

        return result, test_mse


class LinearPerceptron(SimplePerceptron):
    def activation_function(self, x: float) -> float:
        return x

    def activation_derivative(self, x: float) -> float:
        return 1


class NonLinearPerceptron(SimplePerceptron):

    def __init__(self, dim: int, beta: float, learning_rate: float, limit: int, eps: float):
        super().__init__(dim, learning_rate, limit, eps, -1, 1)
        self.beta = beta

    def activation_function(self, x: float) -> float:
        return math.tanh(self.beta * x)

    def activation_derivative(self, x: float) -> float:
        return self.beta * (1 - self.activation_function(x) ** 2)
