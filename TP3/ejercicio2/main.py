from perceptrons.simple_perceptron import LinearPerceptron, HypPerceptron,LogPerceptron
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import random


def get_config_params(config):
    learning_rate = config["learning_rate"]
    train_percentage = config["train_percentage"]
    epoch_limit = config["max_epochs"]
    beta = config["beta"]
    eps = config["epsilon"]

    return learning_rate, train_percentage, epoch_limit, beta, eps


def split_data(data, train_ratio):
    # Asumiendo que esta función baraja y divide correctamente los datos
    np.random.seed(42)  # Para reproducibilidad en el barajado
    shuffled_indices = np.random.permutation(len(data))
    train_size = int(len(data) * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
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


def graph_mse_per_learning_rate(config):
    _, train_percentage, epoch_limit, beta, eps = get_config_params(config)

    train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
    dim = len(train_set[0])

    # Crear figura para el gráfico de perceptrones lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época para diferentes learning rate  - Perceptrón Lineal')
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
    plt.title('Error de entrenamiento por época para diferentes learning rate - Perceptrón Hiperbolic')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón no lineal
    for learning_rate in [0.01, 0.001, 0.0001]:
        non_linear_perceptron = HypPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, True)
        plt.plot(range(1, non_linear_train_output[0] + 1), non_linear_train_output[3], label=f'LR={learning_rate}')

    plt.legend()
    plt.grid(True)
    plt.show()

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época para diferentes learning rate - Perceptrón Logistic')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón no lineal
    for learning_rate in [0.01, 0.001, 0.0001]:
        log_perceptron = LogPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        log_output = log_perceptron.train(train_set, train_expected_set, test_set,
                                                              test_expected_set, True)
        plt.plot(range(1, log_output[0] + 1), log_output[3], label=f'LR={learning_rate}')

    plt.legend()
    plt.grid(True)
    plt.show()




def graph_mse_test_per_train(config):
    learning_rate, _, epoch_limit, beta, eps = get_config_params(config)
    # Crear figura para el gráfico de perceptrones lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de test por porcentaje de entrenamiento - Perceptrón Lineal')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    linear_errors = []
    train_percentages = [0.2, 0.4, 0.6, 0.8]
    bar_width = 0.35  # Ancho de las barras
    index = np.arange(len(train_percentages))  # Índices para las barras
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        
        dim = len(train_set[0])

        linear_perceptron = LinearPerceptron(dim,learning_rate, epoch_limit, eps)
        linear_train_output = linear_perceptron.train(train_set, train_expected_set, False)
        linear_test_output = linear_perceptron.predict(test_set,test_expected_set,False)
        linear_errors.append(linear_test_output[1])  # Obtener el último error de la lista

    plt.bar(index, linear_errors, bar_width, label='Perceptrón Lineal')

    plt.xticks(index, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.title('Error de test por porcentaje de entrenamiento - Perceptrón Hiperbolic')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    non_linear_errors = []
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        non_linear_perceptron = HypPerceptron(dim, beta,learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, True)
        non_linear_test_output = non_linear_perceptron.predict(test_set,test_expected_set,True)
        non_linear_errors.append(non_linear_test_output[1])  # Obtener el último error de la lista

    plt.bar(index, non_linear_errors, bar_width, label='Perceptrón Hiperbolic')

    plt.xticks(index, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Error de test por porcentaje de entrenamiento - Perceptrón Logistic')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    beta_errors = []
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        beta_perceptron = LogPerceptron(dim, beta,learning_rate, epoch_limit, eps)
        beta_train_output = beta_perceptron.train(train_set, train_expected_set, True)
        beta_test_output = beta_perceptron.predict(test_set,test_expected_set,True)
        beta_errors.append(beta_test_output[1])  # Obtener el último error de la lista

    plt.bar(index, beta_errors, bar_width, label='Perceptrón non Logistic')

    plt.xticks(index, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()






def graph_mse_per_train_percentage(config):
    learning_rate, _, epoch_limit, beta, eps = get_config_params(config)
    train_percentages = [0.2, 0.4, 0.6, 0.8]
    bar_width = 0.35  # Ancho de las barras
    index = np.arange(len(train_percentages))  # Índices para las barras

    # Crear figura para el gráfico de perceptrones lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento agrupado por training percentage - Perceptrón Lineal')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')
    # Inicializar listas para almacenar los errores de entrenamiento
    linear_errors = []
    # Iterar sobre cada porcentaje de entrenamiento para perceptrón lineal
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
        linear_train_output = linear_perceptron.train(train_set, train_expected_set,  False)
        linear_errors.append(linear_train_output[3][-1])  # Obtener el último error de la lista
    # Crear gráfico de barras para perceptrones lineales
    plt.bar(index, linear_errors, bar_width, label='Perceptrón Lineal')
    plt.xticks(index, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()


    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento agrupado por training percentage - Perceptrón Hiperbolic')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')
    # Inicializar listas para almacenar los errores de entrenamiento
    non_linear_errors = []
    # Iterar sobre cada porcentaje de entrenamiento para perceptrón no lineal
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])
        non_linear_perceptron = HypPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, True)
        non_linear_errors.append(non_linear_train_output[3][-1])  # Obtener el último error de la lista

    # Crear gráfico de barras para perceptrones no lineales
    plt.bar(index, non_linear_errors, bar_width, label='Perceptrón Hiperbolic')
    plt.xticks(index, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento agrupado por training percentage - Perceptrón Logistic')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')
    log_errors =[]
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        log_perceptron = LogPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        log_train_output = log_perceptron.train(train_set, train_expected_set, True)
        log_errors.append(log_train_output[3][-1])  # Obtener el último error de la lista

    # Crear gráfico de barras para perceptrones no lineales
    plt.bar(index + bar_width, log_errors, bar_width, label='Perceptrón Log')
    plt.xticks(index + bar_width, [f'{percent*100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()



def graph_mse_per_beta(config):
    learning_rate, train_percentage, epoch_limit, _, eps = get_config_params(config)

    train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
    dim = len(train_set[0])
    

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época para varias beta - Perceptrón Hyperbolic')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón no lineal
    for beta in [0.25, 0.50, 0.75,1]:
        non_linear_perceptron = HypPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, True)
        plt.plot(range(1, non_linear_train_output[0] + 1), non_linear_train_output[3], label=f'beta={beta}')

    plt.legend()
    plt.grid(True)
    plt.show()

    # Crear figura para el gráfico de perceptrones no lineales
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento por época - Perceptrón Logistic')
    plt.xlabel('Época')
    plt.ylabel('Error cuadrático medio - SME')

    # Iterar sobre cada tasa de aprendizaje para perceptrón no lineal
    for beta in [0.25, 0.50, 0.75,1]:
        log_perceptron = LogPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        log_output = log_perceptron.train(train_set, train_expected_set, True)
        plt.plot(range(1, log_output[0] + 1), log_output[3], label=f'beta={beta}')

    plt.legend()
    plt.grid(True)
    plt.show()





if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.load(f)


    graph_mse_per_train_percentage(config)
    #graph_mse_per_beta(config)
    #graph_mse_per_learning_rate(config)
    graph_mse_test_per_train(config)


   



