import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ejercicio3.Activation import Sigmoid
from ejercicio3.Optimazer import GradientDescentOptimizer
from ejercicio3.neural_network import NeuralNetwork

def accuracy_for_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    predicted_digits = [np.argmax(output) for output in predictions]  # Removido np.round, no necesario con np.argmax
    y_test_digits = [np.argmax(y) for y in y_test]
    total_rights = np.sum([1 if pred == real else 0 for pred, real in zip(predicted_digits, y_test_digits)])
    return total_rights / len(x_test)

def read_data(archivo):
    with open(archivo) as file:
        lines = [line.strip() for line in file if line.strip()]

    matrices = []
    for i in range(0, len(lines), 7):
        matrix = [list(map(int, line.split())) for line in lines[i:i + 7]]
        matrices.append(matrix)
    return matrices

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
hidden_size = 30  # Tamaño de la capa oculta arbitrario
output_size = 10  # Tamaño de salida corregido para clasificación de 10 clases
learning_rate = 0.01  # Tasa de aprendizaje

# Corrección: Asegurar que las dimensiones de las capas estén correctamente definidas
model = NeuralNetwork([dim, 10, 8, output_size], learning_rate, Sigmoid(), verbose=False)

epochs = 5000  # Número de épocas de entrenamiento
errors = model.train(matrices, expected_output, epochs)  # Asumiendo que devuelve errores

# Predicciones y precisión
predictions = model.predict(matrices)
predicted_digits = [np.argmax(output) for output in predictions]
print("Predicciones:")
for i in range(len(matrices)):
    print(f"Entrada:\n{np.array(matrices[i]).reshape((7, 5))}")
    print(f"Salida Esperada: {expected_output[i]}")
    print(f"Salida Predicha: {predicted_digits[i]}")
    print("-------------")
accuracy = accuracy_for_model(model, matrices, expected_output)  # eps ajustado para clasificación
print(f"Accuracy: {accuracy:.2f}")

# Graficar el error por época
plt.figure(figsize=(10, 5))
plt.plot(errors, label='Error por época')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Error durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.show()
