import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron_multicapa import NeuralNetwork
from Optimazer import *


# Paso 1: Instancia la red neuronal
dim = 2  # Tamaño de entrada (dos características)
hidden_size = 4  # Tamaño de la capa oculta
output_size = 1  # Tamaño de salida (una salida)
learning_rate = 0.1  # Tasa de aprendizaje
eps = 0.1  # Término de regularización

# Instancia de la red neuronal
optimizer = GradientDescentOptimizer(learning_rate=0.01)
model = NeuralNetwork(dim, hidden_size, output_size, learning_rate, eps, optimizer)

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



def superficie_de_decision():
    # Crear un grid de puntos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predecir el resultado para cada punto en el grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar la superficie de decisión
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Graficar los puntos de entrada
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Superficie de Decisión de la Red Neuronal')
    plt.show()

def convergencia_de_error():
    # Paso 3: Entrena la red neuronal y devuelve los errores
    errors = model.train(X, y, epochs)

    # Paso 5: Grafica la convergencia del error
    plt.plot(range(epochs), errors)
    plt.xlabel('Épocas')
    plt.ylabel('Error Medio Cuadrado')
    plt.title('Convergencia del Error durante el Entrenamiento')
    plt.show()


superficie_de_decision()
convergencia_de_error()