from emojis import *
from Activation import *
from Layer import *
from variational_ae import *
from neural_network import *
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
LATENT_SIZE = 2
HIDDEN_SIZE = 100
HIDDEN_SIZE2 = 200
HIDDEN_SIZE3 = 300
EMOJIS_CHOSEN = len(emojis_images)
NOISE = None

emojis_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
data = np.array(emojis_images)
dataset_input = data[emojis_indexes]
dataset_input_list = list(dataset_input)

optimizer = Adam(0.001)

def load_image(path):
    # Abre la imagen y conviértela a escala de grises
    image = Image.open(path).convert('L')
    # Convierte la imagen a una matriz NumPy
    matrix = np.array(image)
    return matrix


def save_image(matrix, path):
    # Crea una imagen a partir de la matriz
    image = Image.fromarray(matrix.astype(np.uint8), mode='L')
    # Guarda la imagen en la ruta especificada como formato PNG
    image.save(path, format='PNG')


def get_all_images(path):
    images = []

    for archivo in os.listdir(path):
        # Verifica si el archivo es un archivo PNG
        if archivo.endswith(".png"):
            # Construye la ruta completa del archivo
            full_path = os.path.join(path, archivo)

            # Carga la imagen en una matriz y agrégala a la lista
            matrix = load_image(full_path)
            images.append(matrix)
    return images


def plot_data(original, decoded, input_rows, input_cols):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')
    ax1.imshow(np.array(original).reshape((input_rows, input_cols)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((input_rows, input_cols)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.show()


def generate_new_emoji(vae):
    for _ in range(15):
        n = 10
        images = np.zeros((INPUT_ROWS, INPUT_COLS * n))

        random_index1 = np.random.choice(emojis_indexes)
        input_reshaped1 = np.reshape(emojis_images[random_index1], (len(emojis_images[random_index1]), 1))
        vae.feedforward(input_reshaped1)
        img1 = vae.sampler.sample

        random_index2 = np.random.choice(emojis_indexes)
        while random_index1 == random_index2:
            random_index2 = np.random.choice(emojis_indexes)
        input_reshaped2 = np.reshape(emojis_images[random_index2], (len(emojis_images[random_index2]), 1))
        vae.feedforward(input_reshaped2)
        img2 = vae.sampler.sample

        for i in range(n):
            z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
            output = vae.decoder.feedforward(z)
            output = output.reshape(INPUT_ROWS, INPUT_COLS)
            images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

        plt.figure(figsize=(10, 10))
        plt.title(f"\"{emojis_name[random_index1]}\" "
                  f"-> \"{emojis_name[random_index2]}\"")
        plt.imshow(images, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


def start():
    encoder = MLP()
    encoder.addLayer(Dense(inputDim=INPUT_SIZE, outputDim=HIDDEN_SIZE3, activation=ReLU(), optimizer=optimizer))
    encoder.addLayer(Dense(inputDim=HIDDEN_SIZE3, outputDim=HIDDEN_SIZE2, activation=ReLU(), optimizer=optimizer))
    encoder.addLayer(Dense(inputDim=HIDDEN_SIZE2, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))

    sampler = Sampler(HIDDEN_SIZE, LATENT_SIZE, optimizer=optimizer)

    decoder = MLP()
    decoder.addLayer(Dense(inputDim=LATENT_SIZE, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE, outputDim=HIDDEN_SIZE2, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE2, outputDim=HIDDEN_SIZE3, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE3, outputDim=INPUT_SIZE, activation=Sigmoid(), optimizer=optimizer))

    vae = VAE(encoder, sampler, decoder)

    vae.train(dataset_input=dataset_input_list, loss=MSE(), epochs=500, callbacks={})


    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output = vae.feedforward(input_reshaped)
        plot_data(list(dataset_input)[i], output, INPUT_ROWS, INPUT_COLS)

    generate_new_emoji(vae)

start()