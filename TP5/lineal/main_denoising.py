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
        Dense(35, 30, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(30, 20, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(20, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 2, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(2, 10, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(10, 20, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(20, 30, optimizer_type=optimizer, learning_rate=learning_rate),
        Sigmoid(),
        Dense(30, 35, optimizer_type=optimizer, learning_rate=learning_rate),
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
            wrongs += 1
            if wrongs > max_wrongs:
                return False
    return True

def count_corrected_letters(original_X, denoised_X, characters, max_wrongs=1):
    corrected = 0
    for c in range(len(characters)):
        input_bitmap = []
        output_bitmap = []

        for i in range(len(original_X[c])):
            input_bitmap.append(round(original_X[c][i][0]))
            output_bitmap.append(round(denoised_X[c][i][0]))

        if compare_bitmaps(input_bitmap, output_bitmap, characters[c], max_wrongs=max_wrongs):
            corrected += 1

    return corrected

def train_and_evaluate_autoencoder(learning_rate, num_runs=10):
    bitmapDict = fonts_to_bitmap(fontDict)
    bitmapList = list(bitmapDict.values())

    # Représentation des caractères en bits
    X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

    # Normaliser les données entre 0 et 1
    X = X / 1.0

    # Correspondance de chaque caractère avec son bitmap X[i]
    characters = list(bitmapDict.keys())

    total_corrected = 0

    for _ in range(num_runs):
        # Ajout de bruit aux données
        X_noisy = add_salt_and_pepper_noise(X.copy())

        autoencoder = generate_autoencoder_arch1('ADAM', learning_rate)
        error = train(autoencoder, mse, mse_derivative, X_noisy, X, epochs=10000, verbose=False)

        # Débruitage des données
        denoised_X = []
        for noisy_data in X_noisy:
            denoised_data = predict(autoencoder, noisy_data)
            denoised_X.append(denoised_data)

        denoised_X = np.array(denoised_X)

        # Calculate number of corrected letters
        corrected_count = count_corrected_letters(X, denoised_X, characters, max_wrongs=1)
        total_corrected += corrected_count

    average_corrected = total_corrected / num_runs
    return average_corrected

def display_characters(autoencoder, X, X_noisy, characters):
    denoised_X = []
    for noisy_data in X_noisy:
        denoised_data = predict(autoencoder, noisy_data)
        denoised_X.append(denoised_data)
    denoised_X = np.array(denoised_X)

    input_matrix_list = []
    noisy_matrix_list = []
    output_matrix_list = []
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


def evaluate_noise_levels(noise_levels, learning_rate, num_runs=3):
    results = []

    for noise_level in noise_levels:
        total_corrected = 0
        for _ in range(num_runs):
            bitmapDict = fonts_to_bitmap(fontDict)
            bitmapList = list(bitmapDict.values())

            # Représentation des caractères en bits
            X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

            # Normaliser les données entre 0 et 1
            X = X / 1.0

            # Correspondance de chaque caractère avec son bitmap X[i]
            characters = list(bitmapDict.keys())

            # Ajout de bruit aux données
            X_noisy = add_salt_and_pepper_noise(X.copy(), salt_prob=noise_level, pepper_prob=noise_level)

            autoencoder = generate_autoencoder_arch1('ADAM', learning_rate)
            error = train(autoencoder, mse, mse_derivative, X_noisy, X, epochs=10000, verbose=False)

            # Débruitage des données
            denoised_X = []
            for noisy_data in X_noisy:
                denoised_data = predict(autoencoder, noisy_data)
                denoised_X.append(denoised_data)

            denoised_X = np.array(denoised_X)

            # Calculate number of corrected letters
            corrected_count = count_corrected_letters(X, denoised_X, characters, max_wrongs=1)
            total_corrected += corrected_count

        average_corrected = total_corrected / num_runs
        results.append(average_corrected)

    return results


def main():
    learning_rates = [0.1, 0.01, 0.001]
    average_corrected_values = []

    for lr in learning_rates:
        average_corrected = train_and_evaluate_autoencoder(lr, num_runs=10)
        average_corrected_values.append(average_corrected)
        print(f"Learning rate: {lr}, Average corrected letters: {average_corrected}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, average_corrected_values, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Corrected Letters')
    plt.title('Average Corrected Letters vs Learning Rate')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

    #New functionality for evaluating noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    learning_rate = 0.001
    average_corrected_per_noise = evaluate_noise_levels(noise_levels, learning_rate, num_runs=3)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, average_corrected_per_noise, marker='o')
    plt.xlabel('Salt and Pepper Noise Level')
    plt.ylabel('Average Corrected Letters')
    plt.title('Average Corrected Letters vs Noise Level')
    plt.grid(True)
    plt.show()


    # Affichage des lettres originales, bruitées et débruitées
    bitmapDict = fonts_to_bitmap(fontDict)
    bitmapList = list(bitmapDict.values())

    # Représentation des caractères en bits
    X = np.reshape(bitmapList, (len(bitmapList), 35, 1))

    # Normaliser les données entre 0 et 1
    X = X / 1.0

    # Correspondance de chaque caractère avec son bitmap X[i]
    characters = list(bitmapDict.keys())

    # Ajout de bruit aux données
    X_noisy = add_salt_and_pepper_noise(X.copy())

    autoencoder1 = generate_autoencoder_simple('ADAM', 0.001)
    train(autoencoder1, mse, mse_derivative, X_noisy, X, epochs=10000, verbose=False)
    display_characters(autoencoder1, X, X_noisy, characters)

main()
