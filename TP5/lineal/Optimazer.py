import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learningRate = learning_rate

    def update(self, gradient, time_step):
        pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, gradient, time_step):
        return -self.learningRate * gradient


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, shape=None):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        if shape is not None:
            self.m = np.zeros(shape)
            self.v = np.zeros(shape)
        else:
            self.m = None
            self.v = None

    def update(self, gradient, time_step):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)

        mHat = self.m / (1 - self.beta1 ** time_step)
        vHat = self.v / (1 - self.beta2 ** time_step)

        return (-self.learningRate * mHat) / (np.sqrt(vHat) + self.epsilon)

