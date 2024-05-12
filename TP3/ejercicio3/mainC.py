import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Activation import Sigmoid
from ejercicio3.Optimazer import AdamOptimizer, GradientDescentOptimizer
from neural_network import NeuralNetwork


def split_data(data, labels, train_ratio):
    indices = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return np.array(data)[train_indices], np.array(data)[test_indices], np.array(labels)[train_indices], \
        np.array(labels)[test_indices]


def min_max_normalize(lista):
    min_val = np.min(lista)
    max_val = np.max(lista)
    return (lista - min_val) / (max_val - min_val)


def read_data(archivo):
    with open(archivo) as file:
        lines = [line.strip() for line in file if line.strip()]

    matrices = []
    for i in range(0, len(lines), 7):
        matrix = [list(map(int, line.split())) for line in lines[i:i + 7]]
        matrices.append(matrix)
    return matrices


# Función para agregar ruido a las imágenes de los dígitos
def add_noise(data, noise_level):
    noisy_data = []
    for image in data:
        noisy_image = image.copy()
        noise = np.random.choice([0, 1], size=image.shape, p=[1 - noise_level, noise_level])
        noisy_image += noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Clip para asegurar que los valores estén entre 0 y 1
        noisy_data.append(noisy_image)
    return np.array(noisy_data)


def graph_confusion_matrix(predictions, y_test, labels=None):
    # Convertir las predicciones a etiquetas discretas
    predicted_labels = [np.argmax(prediction) for prediction in predictions]

    # Construir la matriz de confusión
    cm = confusion_matrix(y_test, predicted_labels, labels=labels)

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()


# Convertir matrices 5x7 a vectores de 35 elementos
archivo = "TP3-ej3-digitos.txt"
matrices = read_data(archivo)

matrices = [np.array(matrix).flatten() for matrix in matrices]
expected_output = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

# Instancia de la red neuronal, corrección en las dimensiones
dim = len(matrices[0])  # Tamaño de entrada correcto
layer1_neurons = 10  # Tamaño de la capa oculta arbitrario
layer2_neurons = 8
output_size = 10
learning_rate = 0.01  # Tasa de aprendizaje

optimizer = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.01)
# optimizer = GradientDescentOptimizer(learning_rate)
model = NeuralNetwork([dim, layer1_neurons, layer2_neurons, output_size], Sigmoid(), optimizer, verbose=False)

epochs = 5000
errors = model.train(matrices, expected_output, epochs)

# Predicción
num_repetitions = 500
matriz_de_confusion = np.zeros((output_size, output_size))
for _ in range(num_repetitions):
    matrices = add_noise(matrices, 0.001)
    predictions = model.predict(matrices)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    real_labels = [np.argmax(label) for label in expected_output]  # One-hot to label

    for real, predicted in zip(real_labels, predicted_labels):
        matriz_de_confusion[real][predicted] += 1

for row in matriz_de_confusion:
    print(" ".join("{:4d}".format(int(x)) for x in row))

# Normalizar la matriz de confusión dividiendo por el número de repeticiones
matriz_de_confusion /= num_repetitions

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
ax = sns.heatmap(matriz_de_confusion, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.show()
