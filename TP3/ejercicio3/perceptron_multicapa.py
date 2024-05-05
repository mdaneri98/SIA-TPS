import numpy as np


class NeuralNetwork:
    # hidden : número de capas ocultas
    def __init__(self, dim, hidden_size, output_size, learning_rate, eps):
        self.input_size = dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.eps = eps

        # Retorna una matriz 'dim'X'hidden_size'
        self.weights_input_hidden = np.random.randn(dim, hidden_size)
        self.bias = np.random.randn(hidden_size)

        # Retorna una matriz 'hidden_size'X'output_size'
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)
        self.hidden_activation = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Propagación hacia adelante
        self.hidden_activation = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias)
        return self.sigmoid(np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output)


    def backward(self, x, y, output):
        # Retropropagación del error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_activation)

        # Calcular los gradientes de los parámetros
        d_weights_hidden_output = self.hidden_activation.T.dot(output_delta)
        d_bias_output = np.sum(output_delta)
        d_weights_input_hidden = x.T.dot(hidden_delta)
        d_bias = np.sum(hidden_delta)

        # Actualizar los pesos y sesgos
        self.weights_hidden_output += d_weights_hidden_output * self.learning_rate
        self.bias_output += d_bias_output * self.learning_rate
        self.weights_input_hidden += d_weights_input_hidden * self.learning_rate
        self.bias += d_bias * self.learning_rate

        # Retornar los gradientes de los pesos y sesgos
        return d_weights_hidden_output, d_bias_output

    def train(self, X, y, epochs):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for x, target in zip(X, y):
                x = np.array([x])  # Convertir a formato de fila
                target = np.array([target])  # Convertir a formato de fila
                output = self.forward(x)
                total_error += np.sum((target - output) ** 2)
                self.backward(x, target, output)
            errors.append(total_error / len(X))
        return errors

    def predict(self, X):
        predictions = []
        for x in X:
            x = np.array([x])  # Convertir a formato de fila
            output = self.forward(x)
            predictions.append(output)
        return np.array(predictions)
