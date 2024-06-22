import numpy as np
from Layer import Layer


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def predict_with_layer_value(network, input, layer_index):
    output = input
    values_at_layer = [output]
    for layer in network:
        output = layer.forward(output)
        values_at_layer.append(output)
    return output, values_at_layer[layer_index]


def encode(network, input, latent_layer_index):
    return predict_with_layer_value(network, input, latent_layer_index)[1]


def decode(network, input, decoder_start_index):
    output = input
    for index, layer in enumerate(network):
        if index >= decoder_start_index:
            output = layer.forward(output)
    return output


def train(network, error_function, error_derivative, x_train, y_train, epochs, verbose=True):
    mse = []
    max_index = len(x_train) - 1

    for e in range(epochs):
        error = 0

        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += error_function(y, output)

            # backward
            grad = error_derivative(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad)

        error /= len(x_train)

        mse.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

    return mse


def train_with_max_error(network, error_function, error_derivative, x_train, y_train, max_epochs, max_error,
                         verbose=True):
    mse = []

    epochs = 0
    for e in range(max_epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += error_function(y, output)

            # backward
            grad = error_derivative(y, output)

            for layer in reversed(network):
                grad = layer.backward(grad)

        error /= len(x_train)

        mse.append(error)
        if verbose:
            print(f"{epochs + 1} epochs, error={error}")

        if error < max_error:
            break

        epochs += 1

    return mse, epochs
