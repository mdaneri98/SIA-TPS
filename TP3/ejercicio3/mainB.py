import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron_multicapa import NeuralNetwork
from Optimazer import *

def split_data(data, labels, train_ratio):
    indices = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return np.array(data)[train_indices], np.array(data)[test_indices], np.array(labels)[train_indices], np.array(labels)[test_indices]

def read_data(archivo):
    with open(archivo) as file:
        lines = [line.strip() for line in file if line.strip()]

    matrices = []
    for i in range(0, len(lines), 7):
        matrix = [list(map(int, line.split())) for line in lines[i:i + 7]]
        matrices.append(matrix)
    return matrices


def accuracy_for_model(model, x_test, y_test, eps):
    predictions = model.predict(x_test)
    predicted_digits = [np.round(np.argmax(output)) for output in predictions]
    total_rights = np.sum([np.abs(pred - real) < eps for pred, real in zip(predicted_digits, y_test)])
    return total_rights / len(x_test)

def accuracy_for_with_std(model_params, data, labels, training_percentage, num_trials=10):
    accuracies = []
    X_train, X_test, y_train, y_test = split_data(data, labels, training_percentage)
    for _ in range(num_trials):
        dim, hidden_size, output_size, learning_rate, epochs, eps, optimizer = model_params
        model = NeuralNetwork(dim, hidden_size, output_size, learning_rate, epochs, optimizer)
        model.train(X_train, y_train, epochs)
        acc = accuracy_for_model(model, data, labels, eps)
        accuracies.append(acc)
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)
    return mean_accuracy, std_dev

# Datos y configuración
matrices = [np.array(matrix).flatten() for matrix in read_data("TP3-ej3-digitos.txt")]
expected_output = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

dim = len(matrices[0])
hidden_size = 40
output_size = 2
learning_rate = 0.01
eps = 0.1
epochs = 1000
training_percentages = [0.2, 0.4, 0.6, 0.8]

# Optimizador ADAM
optimizer = AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=0.01)
accuracies_with_std = []
for pct in training_percentages:
    model_params = (dim, hidden_size, output_size, learning_rate, epochs, eps, optimizer)
    accuracy = accuracy_for_with_std(model_params, matrices, expected_output, pct)
    accuracies_with_std.append(accuracy)


means = [acc[0] for acc in accuracies_with_std]
std_devs = [acc[1] for acc in accuracies_with_std]

# Crear gráfico de barras con barras de error
plt.bar(training_percentages, means, yerr=std_devs, capsize=5, width=0.1, align='center')
plt.xlabel("Porcentaje de Entrenamiento con optimizador ADAM.")
plt.ylabel("Accuracy")
plt.title("Accuracy del training vs Porcentaje de Entrenamiento")
plt.grid(axis='y')
plt.show()



# Optimizador Gradiente descendente
optimizer = GradientDescentOptimizer(learning_rate=0.01)
accuracies_with_std = []
for pct in training_percentages:
    model_params = (dim, hidden_size, output_size, learning_rate, epochs, eps, optimizer)
    accuracy = accuracy_for_with_std(model_params, matrices, expected_output, pct)
    accuracies_with_std.append(accuracy)


means = [acc[0] for acc in accuracies_with_std]
std_devs = [acc[1] for acc in accuracies_with_std]

# Crear gráfico de barras con barras de error
plt.bar(training_percentages, means, yerr=std_devs, capsize=5, width=0.1, align='center')
plt.xlabel("Porcentaje de Entrenamiento con optimizador Gradiente descendente.")
plt.ylabel("Accuracy")
plt.title("Accuracy del training vs Porcentaje de Entrenamiento")
plt.grid(axis='y')
plt.show()