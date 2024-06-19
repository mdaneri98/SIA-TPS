import numpy as np

def mse(expected, predicted):
    return np.mean(np.power(expected - predicted, 2))

def mse_derivative(expected, predicted):
    return 2 * (predicted - expected) / np.size(expected)