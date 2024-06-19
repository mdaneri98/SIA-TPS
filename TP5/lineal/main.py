from Layer import *
from Activation import *
from mse import *
from neural_network import *


def fonts_to_bitmap(fonts: dict):
    bitmaps = {}
    for (character, hexaList) in fonts.items():
        bitmap = []
        for byte in hexaList:
            binary = format(byte, '08b')
            row = [int(bit) for bit in binary[-5:]]
            bitmap.extend(row)
        bitmaps[character] = bitmap
    return bitmaps


# Imprime un bitmap de 7x5
def print_bitmap(bitmap: list):
    for i in range(7):
        for j in range(5):
            print(bitmap[i * 5 + j], end='')
        print()


# Devuelve una matriz de 7x5
def bitmap_as_matrix(bitmap: list):
    return [[bitmap[i * 5 + j] for j in range(5)] for i in range(7)]



def add_noise_to_dataset(dataset, noise_level=0.3):
    noisy_dataset = dataset.astype(np.float64)

    for i in range(len(noisy_dataset)):
        for j in range(len(noisy_dataset[i])):
            if np.random.rand() < noise_level:
                delta = np.random.normal(0, 0.5)
                if noisy_dataset[i, j] == 1.:
                    noisy_dataset[i, j] -= np.abs(delta)
                else:
                    noisy_dataset[i, j] += np.abs(delta)

    return noisy_dataset


def train_different_architectures(optimizer, learning_rate, max_epochs, dataset):
    mse_list = []

    autoencoder = [
        Dense(35, 20, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(20, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 20, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(20, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    autoencoder = [
        Dense(35, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 5, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(5, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 5, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(5, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    autoencoder = [
        Dense(35, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    autoencoder = [
        Dense(35, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]
    error = train(autoencoder, mse, mse_derivative, dataset, dataset, epochs=max_epochs, verbose=False)
    mse_list.append(error)

    return mse_list

