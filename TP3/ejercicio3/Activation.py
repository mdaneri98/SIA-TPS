import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @staticmethod
    @abstractmethod
    def apply(x):
        pass

    @staticmethod
    @abstractmethod
    def apply_dx(x):
        pass


class Sigmoid(Activation):

    def apply(x):
        if -700 < x < 700:
            return np.exp(x) / (1 + np.exp(x))
        return 0 if x < 0 else 1

    def apply_dx(x):
        if -355 < x < 355:
            return np.exp(x) / np.power(np.exp(x) + 1, 2)
        return 0


class Tanh(Activation):
    def apply(excitation):
        return np.tanh(excitation)

    def apply_dx(excitation):
        return 1 - np.tanh(excitation) ** 2