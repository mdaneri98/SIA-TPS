from Layer import *
from Activation import *
from mse import *
from neural_network import *
from font import fontDict
import numpy as np
import matplotlib.pyplot as plt


# Define las funciones de error
def fonts_to_bitmap(fonts: dict):
    bitmaps = {}
    for character, hexaList in fonts.items():
        bitmap = []
        for byte in hexaList:
            binary = format(byte, '08b')
            row = [int(bit) for bit in binary[-5:]]
            bitmap.extend(row)
        bitmaps[character] = bitmap
    return bitmaps

# Función para representar un bitmap como matriz 7x5
def bitmap_as_matrix(bitmap: list):
    return [[bitmap[i * 5 + j] for j in range(5)] for i in range(7)]

# Función para graficar matrices de bitmaps
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

def plot_latent_space(raw_latent_spaces, labels):
    latent_spaces = np.array([latent_space.flatten() for latent_space in raw_latent_spaces])

    plt.figure(figsize=(10, 8))
    plt.scatter(latent_spaces[:, 0], latent_spaces[:, 1], c='blue', alpha=0.5)
    plt.xlabel('Latente 1')
    plt.ylabel('Latente 2')
    plt.title('Datos de Entrada en el Espacio Latente')

    # Agregar etiquetas a cada punto
    for i, (x, y) in enumerate(latent_spaces):
        letra = labels[i]
        plt.annotate(letra, (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    plt.grid(True)
    plt.show()


# Definición del autoencoder
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
# Función para comparar bitmaps
def compare_bitmaps(input_bitmap, output_bitmap, character, max_wrongs=1):
    wrongs = 0
    for i in range(7 * 5):
        if round(input_bitmap[i]) != round(output_bitmap[i]):
            print(f"Pixel {i} of '{character}' is different: {input_bitmap[i]} != {output_bitmap[i]}")
            wrongs += 1
            if wrongs > max_wrongs:
                return False
    return True

def start():
    bitmapDict = fonts_to_bitmap(fontDict)
    bitmapList = list(bitmapDict.values())

    # Representación de los caracteres en bits.
    X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

    # Normalizar los datos entre 0 y 1
    X = X / 1.0

    # Correspondencia de cada carácter con su bitmap X[i]
    characters = list(bitmapDict.keys())
    print(characters)

    autoencoder = generate_autoencoder('ADAM', 0.001)
    error = train(autoencoder, mse, mse_derivative, X, X, epochs=10000, verbose=True)
    print(f"Se entrenaron {10000} epochs con un error de {error[-1]}")

    latent_spaces = []
    raw_latent_spaces = []
    input_matrix_list = []
    output_matrix_list = []
    correct = 0
    for c in range(len(characters)):
        input_bitmap = []
        output_bitmap = []

        for i in range(len(X[c])):
            input_bitmap.append(X[c][i][0])
        input_matrix_list.append(bitmap_as_matrix(input_bitmap))

        bits, raw_latent_space = predict_with_layer_value(autoencoder, X[c], 6)
        raw_latent_spaces.append(raw_latent_space)
        latent_spaces.append((raw_latent_space[0][0], raw_latent_space[1][0]))

        for bit in bits:
            output_bitmap.append(bit[0])

        if not compare_bitmaps(input_bitmap, output_bitmap, characters[c]):
            print(f"Error en la reconstrucción del carácter '{characters[c]}'")
        else:
            correct += 1

        output_matrix_list.append(bitmap_as_matrix(output_bitmap))

    plot_bitmap_matrix_2(input_matrix_list, characters, "Caracteres Originales")
    plot_bitmap_matrix_2(output_matrix_list, characters, "Caracteres Predichos")
    plot_latent_space(raw_latent_spaces, characters)

start()
