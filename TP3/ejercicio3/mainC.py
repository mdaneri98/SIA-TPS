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
    # Convertir las predicciones a etiquetas
    predicted_labels = [np.argmax(prediction) for prediction in predictions]

    print(predicted_labels)

    # Construir la matriz de confusión
    cm = confusion_matrix(y_test, predicted_labels, labels=labels)

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()



# Cargar los datos de los dígitos
matrices = [np.array(matrix).flatten() for matrix in read_data("TP3-ej3-digitos.txt")]
expected_output = np.array([i for i in range(len(matrices))])


# Dividir los datos en conjunto de entrenamiento y conjunto de prueba (80% entrenamiento, 20% prueba)
x_data, y_data, _, _ = split_data(matrices, expected_output, 0.8)

# Crear instancia de la red neuronal para clasificación de dígitos (10 neuronas de salida)
model = NeuralNetwork(dim=len(x_data[0]), hidden_size=10, output_size=10, learning_rate=0.01, eps=0.1, optimizer=GradientDescentOptimizer(0.01))

# Entrenar la red neuronal con los datos de entrenamiento
epochs = 1000
model.train(x_data, y_data, epochs)

# Evaluar el rendimiento de la red neuronal en el conjunto de prueba
predictions = model.predict(x_data)

# Graficar la matriz de confusión
graph_confusion_matrix(predictions, y_data, labels=range(10))

# # Función para visualizar una imagen del dígito con su etiqueta
# def visualize_digit(image, label):
#     plt.imshow(image.reshape(5, 5), cmap='gray')
#     plt.title(f'Etiqueta: {label}')
#     plt.axis('off')
#     plt.show()
#
# # Visualizar algunos ejemplos de imágenes de dígitos y sus etiquetas
# num_examples = 5
# random_indices = np.random.choice(len(test_data), size=num_examples, replace=False)
# for index in random_indices:
#     visualize_digit(test_data[index, :-1], int(test_data[index, -1]))
#
# # Graficar la convergencia del error durante el entrenamiento
# plt.plot(range(epochs), errors)
# plt.xlabel('Épocas')
# plt.ylabel('Error Medio Cuadrado')
# plt.title('Convergencia del Error durante el Entrenamiento')
# plt.show()
