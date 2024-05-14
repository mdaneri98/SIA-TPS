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
    data = pd.read_csv('./test_set.csv')

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



def normalize_01(errors):
    min_error = min(errors)
    max_error = max(errors)
    if max_error - min_error == 0:
        return [0] * len(errors)  # Evitar división por cero si todos los errores son iguales
    normalized_errors = [(error - min_error) / (max_error - min_error) for error in errors]
    return normalized_errors


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
        linear_train_output = linear_perceptron.train(train_set, train_expected_set, False)
        y_values = normalize_01(linear_train_output[3])
        plt.plot(range(1, linear_train_output[0] + 1), y_values, label=f'LR={learning_rate}')


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
        y_values = normalize_01(non_linear_train_output[3])
        plt.plot(range(1, non_linear_train_output[0] + 1), y_values, label=f'LR={learning_rate}')

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
        log_output = log_perceptron.train(train_set, train_expected_set, True)
        y_values = normalize_01(log_output[3])
        plt.plot(range(1, log_output[0] + 1), y_values, label=f'LR={learning_rate}')

    plt.legend()
    plt.grid(True)
    plt.show()



def calculate_improvement(old_mse, new_mse):
    if old_mse == 0:
        return 0  # Evitar división por cero, adaptar según la lógica necesaria
    return ((old_mse - new_mse) / old_mse) * 100

def graph_mse_test_per_train_percentage(config):
    learning_rate, _, epoch_limit, beta, eps = get_config_params(config)

    train_percentages = [0.2, 0.4, 0.6, 0.8]
    bar_width = 0.2  # Ancho de las barras más pequeño para ajustar tres barras
    index = np.arange(len(train_percentages))  # Índices para las barras

    # Listas para almacenar los errores normalizados
    linear_errors = []
    non_linear_errors = []
    beta_errors = []

    # Recorrer cada porcentaje de entrenamiento y calcular errores para cada modelo
    for train_percentage in train_percentages:
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        # Perceptrón Lineal
        linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
        linear_perceptron.train(train_set, train_expected_set, False)
        linear_test_output = linear_perceptron.predict(test_set, test_expected_set, False)
        linear_errors.append(linear_test_output[1])

        # Perceptrón Hiperbólico
        non_linear_perceptron = HypPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        non_linear_perceptron.train(train_set, train_expected_set, True)
        non_linear_test_output = non_linear_perceptron.predict(test_set, test_expected_set, True)
        non_linear_errors.append(non_linear_test_output[1])

        # Perceptrón Logístico
        beta_perceptron = LogPerceptron(dim, beta, learning_rate, epoch_limit, eps)
        beta_perceptron.train(train_set, train_expected_set, True)
        beta_test_output = beta_perceptron.predict(test_set, test_expected_set, True)
        beta_errors.append(beta_test_output[1])

    all_errors = linear_errors + non_linear_errors + beta_errors
    normalized_all_errors = normalize_01(all_errors)
    normalized_linear_errors = normalized_all_errors[:len(linear_errors)]
    normalized_non_linear_errors = normalized_all_errors[len(linear_errors):len(linear_errors) + len(non_linear_errors)]
    normalized_beta_errors = normalized_all_errors[-len(beta_errors):]

    # Crear figura para el gráfico
    plt.figure(figsize=(10, 5))
    plt.title('Error de test por porcentaje de entrenamiento por Perceptrón')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio normalizado - SME')
    plt.yscale('log')  # Usando escala logarítmica en el eje y

    # Dibujar las barras
    plt.bar(index - bar_width, normalized_linear_errors, bar_width, label='Perceptrón Lineal')
    plt.bar(index, normalized_non_linear_errors, bar_width, label='Perceptrón Hiperbólico')
    plt.bar(index + bar_width, normalized_beta_errors, bar_width, label='Perceptrón Logístico')

    # Configurar las etiquetas del eje x
    plt.xticks(index, [f'{percent * 100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.show()




def graph_mse_per_train_percentage(config):
    learning_rate, _, epoch_limit, beta, eps = get_config_params(config)
    train_percentages = [0.2, 0.4, 0.6, 0.8]
    bar_width = 0.25  # Ancho de las barras, reducido para ajustar 3 barras
    index = np.arange(len(train_percentages))  # Índices para las barras

    # Listas para almacenar los errores de entrenamiento
    linear_errors = []
    non_linear_errors = []
    log_errors = []

    # Iterar sobre cada porcentaje de entrenamiento para cada tipo de perceptrón
    for i, train_percentage in enumerate(train_percentages):
        train_set, train_expected_set, test_set, test_expected_set = initialize_data(train_percentage)
        dim = len(train_set[0])

        # Perceptrón Lineal
        linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, 0.001)
        linear_train_output = linear_perceptron.train(train_set, train_expected_set, False)
        linear_errors.append(linear_train_output[3][-1] / dim)  # Error normalizado por dimensión

        # Perceptrón Hiperbólico
        non_linear_perceptron = HypPerceptron(dim, beta, learning_rate, epoch_limit, 0.00001)
        non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, True)
        non_linear_errors.append(non_linear_train_output[3][-1] / dim)  # Error normalizado por dimensión

        # Perceptrón Logístico
        log_perceptron = LogPerceptron(dim, beta, learning_rate, epoch_limit, 0.00001)
        log_train_output = log_perceptron.train(train_set, train_expected_set, True)
        log_errors.append(log_train_output[3][-1] / dim)  # Error normalizado por dimensión

    print(f"Lineal | Max epocas: {linear_train_output[0]}")
    print(f"Hyperbólica | Max epocas: {non_linear_train_output[0]}")
    print(f"Logística | Max epocas: {non_linear_train_output[0]}")

    all_errors = linear_errors + non_linear_errors + log_errors
    normalized_all_errors = normalize_01(all_errors)
    normalized_linear_errors = normalized_all_errors[:len(linear_errors)]
    normalized_non_linear_errors = normalized_all_errors[len(linear_errors):len(linear_errors) + len(non_linear_errors)]
    normalized_beta_errors = normalized_all_errors[-len(log_errors):]

    # Crear figura para el gráfico
    plt.figure(figsize=(10, 5))
    plt.title('Error de entrenamiento agrupado por training percentage')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Error cuadrático medio - SME')

    # Dibujar las barras
    plt.bar(index - bar_width, normalized_linear_errors, bar_width, label='Perceptrón Lineal', align='center')
    plt.bar(index, normalized_non_linear_errors, bar_width, label='Perceptrón Hiperbólico', align='center')
    plt.bar(index + bar_width, normalized_beta_errors, bar_width, label='Perceptrón Logístico', align='center')

    # Configurar las etiquetas del eje x
    plt.xticks(index, [f'{percent * 100}%' for percent in train_percentages])
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Usando escala logarítmica en el eje y
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
    for beta in [0.25, 0.50, 0.75, 1]:
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


def cross_validate_perceptron(perceptron_type: int = 0, k=5, learning_rate=0.01, epoch_limit=300, eps=0.01):
    data = pd.read_csv('./datos.csv')

    # Convertir a listas
    x_values = data[['x1', 'x2', 'x3']].values.tolist()
    y_values = data['y'].values.tolist()

    n = len(x_values)
    indices = list(range(n))
    random.shuffle(indices)  # Barajar los índices

    fold_size = n // k
    train_accuracies = []
    test_accuracies = []

    for i in range(k):
        # Dividir los índices en k pliegues
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        # Obtener conjuntos de entrenamiento y prueba
        x_train = [x_values[j] for j in train_indices]
        y_train = [y_values[j] for j in train_indices]
        x_test = [x_values[j] for j in test_indices]
        y_test = [y_values[j] for j in test_indices]

        scale = not (perceptron_type == 0)

        """
            Se elige el epsilon, para cada modelo según un error relativo esperado de 0.1
            Según la salida de la función de activación de cada perceptrón, se ajusta el epsilon.
            Por ej. La salida del hyperbólico está entre -1 y 1 => Se multiplica el relativo por 2
            Por ej. La salida del Lineal está entre -10 y 10 => Se multiplica el relativo por 20  
        """
        acceptable_error_percentage = 0.1  # Error relativo
        epsilon_lineal = acceptable_error_percentage * 20  # Suponiendo salida de activación es entre [-10, 10]
        epsilon_hyperbolic = acceptable_error_percentage * 2  # [-1, 1]
        epsilon_logistic = acceptable_error_percentage * 1  # [0, 1]

        # Inicializar y entrenar el perceptrón no lineal
        if perceptron_type == 0:
            perceptron = LinearPerceptron(dim=len(x_train[0]), learning_rate=learning_rate,
                                          limit=epoch_limit, eps=epsilon_lineal)
            _, _, _, errors = perceptron.train(x_train, y_train, scale=scale)
        elif perceptron_type == 1:
            perceptron = HypPerceptron(dim=len(x_train[0]), beta=1.0, learning_rate=learning_rate,
                                       limit=epoch_limit, eps=epsilon_hyperbolic)
            _, _, _, errors = perceptron.train(x_train, y_train, scale=scale)
        else:
            perceptron = LogPerceptron(dim=len(x_train[0]), beta=1.0, learning_rate=learning_rate,
                                       limit=epoch_limit, eps=epsilon_logistic)
            _, _, _, errors = perceptron.train(x_train, y_train, scale=scale)


        # Calcular la precisión en el conjunto de entrenamiento
        if scale:
            y_train = [perceptron.normalize_value(y, min(y_train), max(y_train)) for y in y_train]
            y_test = [perceptron.normalize_value(y, min(y_test), max(y_test)) for y in y_test]

        train_predictions, _ = perceptron.predict(x_train, y_train, scale=scale)
        train_accuracy = np.mean([np.abs(y - pred) < perceptron.eps for y, pred in zip(y_train, train_predictions)])
        train_accuracies.append(train_accuracy)

        # Calcular la precisión en el conjunto de prueba
        test_predictions, _ = perceptron.predict(x_test, y_test, scale=scale)
        test_accuracy = np.mean([np.abs(y - pred) < perceptron.eps for y, pred in zip(y_test, test_predictions)])
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies


def plot_accuracies(perceptron_type: int, train_accuracies, test_accuracies):
    x = np.arange(len(train_accuracies))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, train_accuracies, width, label='Accuracy de train')
    ax.bar(x + width / 2, test_accuracies, width, label='Accuracy de test')

    ax.set_xlabel('Número de partición elegida para training set')
    ax.set_ylabel('Accuracy')

    if perceptron_type == 0:
        ax.set_title('Accuracy para cada iteración sobre el conjunto partido de datos\nLineal')
    elif perceptron_type == 1:
        ax.set_title('Accuracy para cada iteración sobre el conjunto partido de datos\nHyperbólica')
    else:
        ax.set_title('Accuracy para cada iteración sobre el conjunto partido de datos\nLogística')

    ax.set_xticks(x)
    ax.legend()

    plt.show()








if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.load(f)

    for i in range(3):
    
        # 3, 4, 6, 7, 8
        train_accuracies, test_accuracies = cross_validate_perceptron(perceptron_type=i, k=6, learning_rate=0.01,
                                                                      epoch_limit=600)
        plot_accuracies(i, train_accuracies, test_accuracies)

    graph_mse_per_learning_rate(config)

    graph_mse_per_train_percentage(config)
    graph_mse_per_beta(config)
    graph_mse_test_per_train_percentage(config)


   



