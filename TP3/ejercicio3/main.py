import pandas as pd
import numpy as np
from perceptron_multicapa import NeuralNetwork

# Paso 1: Instancia la red neuronal
dim = 2  # Tamaño de entrada (dos características)
hidden_size = 4  # Tamaño de la capa oculta
output_size = 1  # Tamaño de salida (una salida)
learning_rate = 0.1  # Tasa de aprendizaje
eps = 0.1  # Término de regularización

# Instancia de la red neuronal
model = NeuralNetwork(dim, hidden_size, output_size, learning_rate, eps)

# Paso 2: Datos de entrada y esperados.
X = np.array([
    [-1, 1],
    [1, -1],
    [-1, -1],
    [1, 1]])  # Entradas
y = np.array([1, 1, -1, -1])

# Paso 3: Entrena la red neuronal
epochs = 1000  # Número de épocas de entrenamiento
model.train(X, y, epochs)

# Paso 4: Predice las salidas para las entradas dadas
predictions = model.predict(X)

# Imprime las predicciones
print("Predicciones:")
for i in range(len(X)):
    print(f"Entrada: {X[i]}, Salida Esperada: {y[i]}, Salida Predicha: {predictions[i]}")



