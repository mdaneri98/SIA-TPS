import sys
import numpy as np


def initialize_weights(n):
    return np.random.rand(n)


# sum(xi*wi) + w0
def compute_excitement(x, w):
    return np.dot(x[1:], w[1:]) + w[0]


def threshold_activation(h):
    return 1 if h > 0 else -1


def compute_error(data, w):
    total_error = 0
    for x, expected in data:
        h = compute_excitement(x, w)
        o = threshold_activation(h)
        total_error += (expected - o) ** 2
    return total_error / 2  # MSE


def perceptron_training(data, learning_rate, limit):
    data_with_bias = [(np.array([1] + list(x)), y) for x, y in data]

    n = len(data_with_bias[0][0])  # Number of features
    w = initialize_weights(n)
    min_error = sys.maxsize
    w_min = None
    i = 0

    intermediate_weights = [w]
    while min_error > 0 and i < limit:
        mu = np.random.randint(len(data_with_bias))
        x_mu, t_mu = data_with_bias[mu]

        h_mu = compute_excitement(x_mu, w)
        o_mu = threshold_activation(h_mu)

        delta_w = learning_rate * (t_mu - o_mu) * x_mu
        w = w + delta_w

        error = compute_error(data_with_bias, w)
        if error < min_error:
            min_error = error
            w_min = w
            intermediate_weights.append(w_min)

        i += 1

    return intermediate_weights, w_min


