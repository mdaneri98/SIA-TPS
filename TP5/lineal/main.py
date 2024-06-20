from Layer import *
from Activation import *
from mse import *
from neural_network import *
import matplotlib.pyplot as plt
from font import fontDict


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


def plot_bitmap_matrix_2(matrix_list, character_list, title):
    num_plots = len(matrix_list)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(np.ceil(num_plots / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_plots:
                axes[i, j].imshow(matrix_list[index], cmap='binary', interpolation='none', vmin=0, vmax=1)
                axes[i, j].set_title(character_list[index])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                fig.delaxes(axes[i, j])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_autoencoder(optimizer=None, learning_rate=0.001):
    return [
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



def is_same_pixel(pixel1, pixel2):
    return round(pixel1) == round(pixel2)


def compare_bitmaps(input_bitmap, output_bitmap, character, max_wrongs=1):
    wrongs = 0
    for i in range(7 * 5):
        if not is_same_pixel(input_bitmap[i], output_bitmap[i]):
            print(f"Pixel {i} of '{character}' is different: {input_bitmap[i]} != {output_bitmap[i]}")
            wrongs += 1
            if wrongs > max_wrongs:
                return False

    return True


def start():
    bitmapDict = fonts_to_bitmap(fontDict)
    bitmapList = list(bitmapDict.values())

    #Representacion de los caracteres en bits.
    X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

    #Correspondencia de cada caracter con su bitmap X[i]
    characters = list(bitmapDict.keys())
    print(characters)

    # epochs = 5000
    # max_error = 0.02

    # print(f"Entrenando con error maximo permitido de {max_error}")
    autoencoder = generate_autoencoder(AdamOptimizer(learning_rate=0.001), 0.001)
    error = train(autoencoder, mse, mse_derivative, X, X, epochs=15000, verbose=False)
    print(f"Se entrenaron {15000} epochs con un error de {error[-1]}")

    latent_spaces = []
    raw_latent_spaces = []
    input_matrix_list = []
    output_matrix_list = []
    correct = 0
    for c in range(len(characters)):
        input_bitmap = []
        output_bitmap = []

        # X es una lista de listas de -listas con un solo elemento-
        for i in range(len(X[c])):
            input_bitmap.append(X[c][i][0])
        input_matrix_list.append(bitmap_as_matrix(input_bitmap))

        # El espacio latente es la salida de la "capa 8"
        bits, raw_latent_space = predict_with_layer_value(autoencoder, X[c], 6)
        raw_latent_spaces.append(raw_latent_space)
        latent_spaces.append((raw_latent_space[0][0], raw_latent_space[1][0]))

        for bit in bits:
            output_bitmap.append(bit[0])

        if not compare_bitmaps(input_bitmap, output_bitmap, characters[c]):
            print(f"Error en la reconstruccion del caracter '{characters[c]}'")
            # break
        else:
            correct += 1

        output_matrix_list.append(bitmap_as_matrix(output_bitmap))

    # if(correct == len(characters)):
    #     break

    # epochs += 1000 # Entreno 200 epochs mas
    # max_error *= 0.9 # Reduzco el error maximo permitido un 10%

    plot_bitmap_matrix_2(input_matrix_list, characters, "Caracteres Originales")
    plot_bitmap_matrix_2(output_matrix_list, characters, "Caracteres Predichos")


start()