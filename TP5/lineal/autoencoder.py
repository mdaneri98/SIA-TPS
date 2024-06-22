from Activation import Sigmoid
from neural_network import *
from Optimazer import *
import numpy as np
import re


class Autoencoder:

    def __init__(self, layers_dim, optimazer):
        """
            layers_dim contiene la cantidad de neuronas por cada layer.
            El primer layer es el input, y el último será el código latente (Z).
        """
        self.encoder = NeuralNetwork(neurons_per_layer=layers_dim, activation=Sigmoid(), optimizer=optimazer)
        self.decoder = NeuralNetwork(neurons_per_layer=layers_dim.reverse(), activation=Sigmoid(), optimizer=optimazer)

    def train(self, X, epochs=100, learning_rate=0.01):
        for _ in range(epochs):
            # Forward pass
            encoded = self.encoder.forward_propagation(X)
            decoded = self.decoder.forward_propagation(encoded)

            # Backward pass
            error = X - decoded
            # d_decoded = error * learning_rate

            d_encoded = self.decoder.back_propagation(decoded, X)
            self.encoder.back_propagation(d_encoded, X)

            # Update weights
            self.encoder.update_weights(learning_rate)
            self.decoder.update_weights(learning_rate)

    def encode(self, X):
        return self.encoder.forward(X)

    def decode(self, encoded_data):
        return self.decoder.forward(encoded_data)


def read_font3_from_header(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Busca la definición de Font3
    pattern = r'Font3\[\d+\]\[\d+\] = \{(.*?)\};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Font3 not found in the provided header file.")

    # Extrae y procesa los datos
    data_str = match.group(1)
    data_str = data_str.replace('\n', '').replace(' ', '')
    rows = data_str.split('},')
    font3_data = []
    for row in rows:
        if row:
            row_data = re.findall(r'0x[0-9a-fA-F]+', row)
            # Rellenar con ceros si tiene menos de 7 elementos
            while len(row_data) < 7:
                row_data.append('0x00')
            font3_data.append([int(x, 16) for x in row_data])

    # Convertir a un array de NumPy
    Font3 = np.array(font3_data, dtype=np.float32)

    # Normalizar los datos
    Font3 /= 0x1f

    return Font3


if __name__ == "__main__":
    file_path = 'fonts.h'
    Font3 = read_font3_from_header(file_path)

    print(Font3)

    # autoencoder = Autoencoder(32, Optimizer(learning_rate=0.01))
    #
    # autoencoder.train(Font3)

    # # Encode and decode a sample
    # encoded_data = autoencoder.encode(X_train[0])
    # decoded_data = autoencoder.decode(encoded_data)

