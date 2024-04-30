import numpy as np
import sys

class LinearPerceptron:

    def __init__(self, data_set, learning_rate, limit):
        self.data_set = data_set
        self.learning_rate = learning_rate
        self.limit = limit

        self.w_min = None

    def initialize_weights(self, n):
        return np.random.rand(n)

    # sum(xi*wi) + w0
    def compute_excitement(self, x, w):
        return w[0] + np.dot(x[1:], w[1:])

    def threshold_activation(self, h):
        return 1 if h > 0 else -1

    def compute_error(self, data_with_bias, w):
        total_error = 0
        for x, expected in data_with_bias:
            h = self.compute_excitement(x, w)
            o = self.threshold_activation(h)
            total_error += (expected - o) ** 2
        return total_error / 2  # MSE

    def run(self):
        data_with_bias = [(np.array([1] + list(x)), y) for x, y in self.data_set]

        n = len(data_with_bias[0][0])  # Number of features
        w = self.initialize_weights(n)
        min_error = sys.maxsize
        w_min = None
        i = 0

        intermediate_weights = [w]
        while min_error > 0 and i < self.limit:
            mu = np.random.randint(len(data_with_bias))
            x_mu, t_mu = data_with_bias[mu]

            h_mu = self.compute_excitement(x_mu, w)
            o_mu = self.threshold_activation(h_mu)

            delta_w = self.learning_rate * (t_mu - o_mu) * x_mu
            w = w + delta_w

            error = self.compute_error(data_with_bias, w)
            if error < min_error:
                min_error = error
                w_min = w
                intermediate_weights.append(w_min)

            i += 1

        self.w_min = w_min

        return intermediate_weights, w_min

    def predict(self, x):
        if self.w_min is None:
            print("First you have to train the perceptron")
            return None
        else:
            # Agregamos un '1' para poder multiplicar w0*1 en el dot product.
            x_with_bias = np.insert(x, 0, 1)
            return self.compute_excitement(x_with_bias, self.w_min)


