import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        pass


class GradientDescentOptimizer(Optimizer):
    def update(self, weights, gradients):
        return weights + self.learning_rate * gradients


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1, beta2, epsilon):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, weights, gradients):
        self.t += 1
        self.m = np.zeros_like(gradients)
        self.v = np.zeros_like(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        ##theta_t <--theta{t - 1} - alpha*hat_m_t / (sqrt{\hat v_t} + epsilon)
        return weights + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)