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
        self.w_intermediate = [self.w]

    def initialize_weights(self, n):
        return np.random.rand(n)

    @abstractmethod
    def activation_function(self, x: float) -> float:
        pass

    @abstractmethod
    def activation_derivative(self, x: float) -> float:
        pass

    def compute_error(self, w: list[float], x_values: list[list[float]], expected_values: list[float]):
        # expected_values ya debe estar normalizado.
        obtained_values = self.compute_for_values(w, x_values)  # Valores obtenidos por el perceptrón.

        total_error = 0
        for mu in range(0, len(expected_values)):
            total_error += (expected_values[mu] - obtained_values[mu]) ** 2
        return total_error / len(x_values)

    def compute_for_values(self, w: list[float], x_values: list[list[float]]):
        return [self.activation_function(self.compute_excitement(w, x)) for x in x_values]

    def compute_excitement(self, w: list[float], x: list[float]) -> float:
        extended_x = np.array([1] + x)
        return np.dot(extended_x, w)

    def delta_weights(self, excitement, expected, obtained, x_mu):
        x_mu = np.array([1] + x_mu)
        return self.learning_rate * (expected - obtained) * self.activation_derivative(excitement) * x_mu

    def normalize_value(self, x, new_min, new_max):
        """ Normaliza una entrada usando el mínimo y máximo utilizado en el entrenamiento. """
        return ((x - new_min) / (new_max - new_min)) * (self.activation_max - self.activation_min) + self.activation_min

    def denormalize_value(self, y_norm, new_min, new_max):
        """ Convierte una salida normalizada de vuelta a su escala original. """
        return ((y_norm - self.activation_min) / (self.activation_max - self.activation_min)) * (
                new_max - new_min) + new_min

    def train(self, x_set: list[list[float]], expected_values: list[float], scale: bool = False):
        min_error = sys.maxsize
        epoch = 0

        if scale:
            expected_values = [self.normalize_value(y, min(expected_values), max(expected_values)) for
                               y in expected_values]

        errors = []

        while min_error > self.eps and epoch < self.limit:
            mu = np.random.randint(len(x_set))
            x_mu = x_set[mu]
            expected_mu = expected_values[mu]

            h_mu = self.compute_excitement(self.w, x_mu)
            o_mu = self.activation_function(h_mu)

            delta_w = self.delta_weights(h_mu, expected_mu, o_mu, x_mu)
            w = self.w + delta_w

            # Computamos error de train set con la nueva w
            error = self.compute_error(w, x_set, expected_values)
            errors.append(error)

            if error < min_error:
                min_error = error
                self.w = w
                self.w_intermediate.append(w)

            epoch += 1

        return epoch, self.w, self.w_intermediate, errors

    def predict(self, x_values: list[list[float]], expected_values: list[float], scale: bool = False) -> tuple[
        list[float], float]:
        if scale:
            expected_values = [self.normalize_value(y, min(expected_values), max(expected_values)) for
                               y in expected_values]

        result = []
        for x_mu in x_values:
            h_mu = self.compute_excitement(self.w, x_mu)
            o_mu = self.activation_function(h_mu)
            result.append(o_mu)

        test_mse = self.compute_error(self.w, x_values, expected_values)


        return result, test_mse

    def accuracy_per_epoch(self, x_values, y_values, scale):
        result = []
        total_predictions = len(y_values)

        for w in self.w_intermediate:
            correct_predictions = 0
            self.w = w  # Actualizar el peso del perceptrón para esta época

            predictions, mse = self.predict(x_values, y_values, scale)

            for i in range(len(predictions)):
                if np.abs(y_values[i] - predictions[i]) <= self.eps:
                    correct_predictions += 1

            accuracy = correct_predictions / total_predictions
            result.append(accuracy)

        return result


class LinearPerceptron(SimplePerceptron):
    def activation_function(self, x: float) -> float:
        return x

    def activation_derivative(self, x: float) -> float:
        return 1


class HypPerceptron(SimplePerceptron):

    def __init__(self, dim: int, beta: float, learning_rate: float, limit: int, eps: float):
        super().__init__(dim, learning_rate, limit, eps, -1, 1)
        self.beta = beta

    def activation_function(self, x: float) -> float:
        return math.tanh(self.beta * x)

    def activation_derivative(self, x: float) -> float:
        return self.beta * (1 - (self.activation_function(x) ** 2))
    
class LogPerceptron(SimplePerceptron):
    def __init__(self, dim: int, beta: float, learning_rate: float, limit: int, eps: float):
        super().__init__(dim, learning_rate, limit, eps, 0, 1)
        self.beta = beta

    def activation_function(self, x: float) -> float:
        return 1/(1+math.exp(-2*self.beta*x))


    def activation_derivative(self, x: float) -> float:
        return 2*self.beta *self.activation_function(x)*(1-self.activation_function(x))
