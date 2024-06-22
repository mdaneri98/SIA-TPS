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

# Définition de l'autoencodeur
def generate_autoencoder(optimizer=None, learning_rate=0.01):
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

    autoencoder = generate_autoencoder('ADAM', 0.001)
    error = train(autoencoder, mse, mse_derivative, X, X, epochs=5000, verbose=True)
    print(f"Se entrenaron {10000} epochs con un error de {error[-1]}")

    # Ajout de bruit aux données
    X_noisy = add_salt_and_pepper_noise(X.copy())

    # Débruitage des données
    denoised_X = []
    for noisy_data in X_noisy:
        denoised_data = predict(autoencoder, noisy_data)
        denoised_X.append(denoised_data)

    denoised_X = np.array(denoised_X)

    latent_spaces = []
    raw_latent_spaces = []
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
    plot_bitmap_matrix_2(noisy_matrix_list, characters, "Caracteres avec Bruit")
    plot_bitmap_matrix_2(output_matrix_list, characters, "Caracteres Débruités")

start()
