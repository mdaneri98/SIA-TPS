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

    def apply(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def apply_dx(self, x):
        return np.exp(x) / np.power(np.exp(x) + 1, 2)


class Tanh(Activation):
    def apply(self,excitation):
        return np.tanh(excitation)

    def apply_dx(self,excitation):
        return 1 - np.tanh(excitation) ** 2