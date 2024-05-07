import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math

from ejercicio3.Optimazer import GradientDescentOptimizer
from perceptron_multicapa import NeuralNetwork

def split_data(data, labels, train_ratio):
    indices = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return np.array(data)[train_indices], np.array(data)[test_indices], np.array(labels)[train_indices], np.array(labels)[test_indices]

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
        noise = np.random.choice([0, 1], size=image.shape, p=[1-noise_level, noise_level])
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

data = read_data("TP3-ej3-digitos.txt")

# Cargar los datos de los dígitos
matrices = [np.array(matrix).flatten() for matrix in data]
expected_output = np.array([i for i in range(len(matrices))])

data = add_noise(matrices, 0.1)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba (80% entrenamiento, 20% prueba)
x_data, _, y_data, _ = split_data(matrices, expected_output, 1)



num_repetitions = 100
all_predicted_labels = []
all_real_labels = []

for _ in range(num_repetitions):
    # Crear instancia de la red neuronal para clasificación de dígitos (10 neuronas de salida)
    model = NeuralNetwork(dim=len(x_data[0]), hidden_size=10, output_size=10, learning_rate=0.01, eps=0.1,
                          optimizer=GradientDescentOptimizer(0.01))

    # Entrenar la red neuronal con los datos de entrenamiento
    epochs = 300
    model.train(x_data, y_data, epochs)

    predictions = model.predict(x_data)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    all_predicted_labels.extend(predicted_labels)
    all_real_labels.extend(y_data)

# Crear la matriz de confusión con todas las predicciones
cm = confusion_matrix(all_real_labels, all_predicted_labels, labels=range(10))

# Calcular el promedio de cada casillero
average_cm = cm / num_repetitions

# Graficar la matriz de confusión con los promedios
plt.figure(figsize=(8, 6))
sns.heatmap(average_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión Promedio')
plt.show()
