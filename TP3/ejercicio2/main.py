from perceptrons.simple_perceptron import LinearPerceptron, NonLinearPerceptron
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import random


def get_config_params(config):
    learning_rate = config["learning_rate"]
    test_percentage = config["test_percentage"]
    epoch_limit = config["max_epochs"]
    beta = config["beta"]
    eps = config["epsilon"]

    return learning_rate, test_percentage, epoch_limit, beta, eps


def split_data(data, test_ratio):
    # Asumiendo que esta función baraja y divide correctamente los datos
    np.random.seed(42)  # Para reproducibilidad en el barajado
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def initialize_data(test_ratio):
    data = pd.read_csv('./datos.csv')

    # Suponiendo que la última columna es la etiqueta
    features = data.columns[:-1]
    label = data.columns[-1]

    train_set, test_set = split_data(data, test_ratio)

    # Dividir los datos en características y etiquetas y convertir a listas
    train_features = train_set[features].values.tolist()  # Lista de listas de floats
    train_labels = train_set[label].values.tolist()  # Lista de floats

    test_features = test_set[features].values.tolist()  # Lista de listas de floats
    test_labels = test_set[label].values.tolist()  # Lista de floats

    # Convertir explícitamente todos los valores a float si es necesario
    train_features = [[float(value) for value in row] for row in train_features]
    train_labels = [float(value) for value in train_labels]

    test_features = [[float(value) for value in row] for row in test_features]
    test_labels = [float(value) for value in test_labels]

    return train_features, train_labels, test_features, test_labels


def graph_mse_per_epoch(config):
    _, test_percentage, epoch_limit, beta, eps = get_config_params(config)

    train_set, train_expected_set, test_set, test_expected_set = initialize_data(test_percentage)
    dim = len(train_set[0])

    # Crear figura para el gráfico de perceptrones lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época - Perceptrón Lineal')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón lineal
    for learning_rate in [0.01, 0.001, 0.0001]:
        linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
        linear_train_output = linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set, False)
        plt.plot(range(1, linear_train_output[0] + 1), linear_train_output[3], label=f'LR={learning_rate}')

    plt.legend()
    plt.grid(True)
    plt.show()

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época - Perceptrón No Lineal')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón no lineal
    for learning_rate in [0.01, 0.001, 0.0001]:
        non_linear_perceptron = NonLinearPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, test_set,
                                                              test_expected_set, True)
        plt.plot(range(1, non_linear_train_output[0] + 1), non_linear_train_output[3], label=f'LR={learning_rate}')

    plt.legend()
    plt.grid(True)
    plt.show()



def cross_validate_perceptron(lineal: bool, k=5, learning_rate=0.01, epoch_limit=300, eps=0.01):
    # learning_rate, test_percentage, epoch_limit, beta, eps = get_config_params(config)
    data = pd.read_csv('./datos.csv')

    # Convertir a listas
    x_values = data[['x1', 'x2', 'x3']].values.tolist()
    y_values = data['y'].values.tolist()


    n = len(x_values)
    indices = list(range(n))
    random.shuffle(indices)  # Barajar los índices

    fold_size = n // k
    accuracies = []

    for i in range(k):
        # Dividir los índices en k pliegues
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        # Obtener conjuntos de entrenamiento y prueba
        x_train = [x_values[j] for j in train_indices]
        y_train = [y_values[j] for j in train_indices]
        x_test = [x_values[j] for j in test_indices]
        y_test = [y_values[j] for j in test_indices]

        # Inicializar y entrenar el perceptrón no lineal
        if lineal:
            perceptron = LinearPerceptron(dim=len(x_train[0]), learning_rate=learning_rate,
                                             limit=epoch_limit,
                                             eps=eps)
            perceptron.train(x_train, y_train, x_test, y_test, scale=False)
        else:
            perceptron = NonLinearPerceptron(dim=len(x_train[0]), beta=1.0, learning_rate=learning_rate, limit=epoch_limit,
                                         eps=eps)
            perceptron.train(x_train, y_train, x_test, y_test, scale=False)

        # Calcular la precisión en el conjunto de prueba
        predictions, _ = perceptron.predict(x_test, y_test, scale=False)
        accuracy = np.mean([np.abs(y - pred) < perceptron.eps for y, pred in zip(y_test, predictions)])
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


def graph_mse_per_train_percentage(config):
    learning_rate, _, epoch_limit, beta, eps = get_config_params(config)

    # Crear figura para el gráfico de perceptrones lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por porcentaje de entrenamiento - Perceptrón Lineal')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    # Inicializar listas para almacenar los errores de entrenamiento
    linear_errors = []

    # Iterar sobre cada porcentaje de entrenamiento para perceptrón lineal
    test_percentages = [0.2, 0.4, 0.6, 0.8]
    bar_width = 0.35  # Ancho de las barras
    index = np.arange(len(test_percentages))  # Índices para las barras
    for i, test_percentage in enumerate(test_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(test_percentage)
        dim = len(train_set[0])

        linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
        linear_train_output = linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set, False)
        linear_errors.append(linear_train_output[3][-1])  # Obtener el último error de la lista

    # Crear gráfico de barras para perceptrones lineales
    plt.bar(index, linear_errors, bar_width, label='Perceptrón Lineal')

    plt.xticks(index, [f'{percent*100}%' for percent in test_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por porcentaje de entrenamiento - Perceptrón No Lineal')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    # Inicializar listas para almacenar los errores de entrenamiento
    non_linear_errors = []

    # Iterar sobre cada porcentaje de entrenamiento para perceptrón no lineal
    for i, test_percentage in enumerate(test_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(test_percentage)
        dim = len(train_set[0])

        non_linear_perceptron = NonLinearPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, test_set,
                                                              test_expected_set, True)
        non_linear_errors.append(non_linear_train_output[3][-1])  # Obtener el último error de la lista

    # Crear gráfico de barras para perceptrones no lineales
    plt.bar(index + bar_width, non_linear_errors, bar_width, label='Perceptrón No Lineal')

    plt.xticks(index + bar_width, [f'{percent*100}%' for percent in test_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.load(f)


    graph_mse_per_train_percentage(config)

    graph_mse_per_epoch(config)


    #graph_mse_per_epoch(config)



