from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Identity(Activation):
    def apply(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class Sigmoid(Activation):
    def apply(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        return self.apply(x) * (1. - self.apply(x))


class Tanh(Activation):
    def apply(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.power(np.tanh(x), 2)



class ReLU(Activation):
    def apply(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)



