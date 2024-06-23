from Layer import *
from Activation import *
from mse import *
from neural_network import *
from font import fontDict
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour convertir les motifs en bitmaps
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

# Fonction pour représenter un bitmap en matrice 7x5
def bitmap_as_matrix(bitmap: list):
    return [[bitmap[i * 5 + j] for j in range(5)] for i in range(7)]

# Fonction pour tracer les matrices de bitmaps
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

# Fonction pour ajouter du bruit sel et poivre
def add_salt_and_pepper_noise(data, salt_prob=0.1, pepper_prob=0.1):
    noisy_data = data.copy()
    num_elements = data.size
    num_salt = np.ceil(salt_prob * num_elements)
    num_pepper = np.ceil(pepper_prob * num_elements)

    # Ajouter du sel (blanc)
    coords = [np.random.randint(0, i, int(num_salt)) for i in data.shape]
    noisy_data[tuple(coords)] = 1

    # Ajouter du poivre (noir)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in data.shape]
    noisy_data[tuple(coords)] = 0

    return noisy_data

# Definición de la arquitectura
def generate_autoencoder_simple(optimizer=None, learning_rate=0.01):
    return [
        Dense(35, 32, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(32, 16, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(16, 8, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(8, 4, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(4, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 4, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(4, 8, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(8, 16, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(16, 32, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(32, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]

def generate_autoencoder_arch1(optimizer=None, learning_rate=0.001):
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

def generate_autoencoder_arch2(optimizer=None, learning_rate=0.001):
    return [
        Dense(35, 25, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(25, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 15, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(15, 25, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(25, 35, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
    ]

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

    # Représentation des caractères en bits
    X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

    # Normaliser les données entre 0 et 1
    X = X / 1.0

    # Correspondance de chaque caractère avec son bitmap X[i]
    characters = list(bitmapDict.keys())
    print(characters)

    # Ajout de bruit aux données
    X_noisy = add_salt_and_pepper_noise(X.copy())

    autoencoder1 = generate_autoencoder_arch1('ADAM', 0.001)
    error = train(autoencoder1, mse, mse_derivative, X_noisy, X, epochs=15000, verbose=True)
    print(f"Se entrenaron {10000} epochs con un error de {error[-1]}")

    # Débruitage des données
    denoised_X = []
    for noisy_data in X_noisy:
        denoised_data = predict(autoencoder1, noisy_data)
        denoised_X.append(denoised_data)

    denoised_X = np.array(denoised_X)

    input_matrix_list = []
    noisy_matrix_list = []
    output_matrix_list = []
    correct = 0
    for c in range(len(characters)):
        input_bitmap = []
        noisy_bitmap = []
        output_bitmap = []

        for i in range(len(X[c])):
            input_bitmap.append(round(X[c][i][0]))
            noisy_bitmap.append(round(X_noisy[c][i][0]))
            output_bitmap.append(round(denoised_X[c][i][0]))

        input_matrix_list.append(bitmap_as_matrix(input_bitmap))
        noisy_matrix_list.append(bitmap_as_matrix(noisy_bitmap))
        output_matrix_list.append(bitmap_as_matrix(output_bitmap))

    plot_bitmap_matrix_2(input_matrix_list, characters, "Caracteres Originales")
    plot_bitmap_matrix_2(noisy_matrix_list, characters, "Caracteres con ruido")
    plot_bitmap_matrix_2(output_matrix_list, characters, "Caracteres sin ruido")

start()
