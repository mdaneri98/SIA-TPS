from perceptrons.simple_perceptron import LinearPerceptron, NonLinearPerceptron
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


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
    data = pd.read_csv('ejercicio2/datos.csv')

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


def show_output(train_errors: list[float],
                test_obtained: list[float], test_mse: float, test_expected: list[float], eps: float,
                scale_function=None):
    """
     # (epoch, train_errors, test_errors)
    # (result, test_mse)
    """
    correct_predictions = 0
    print("Expected         | Expected_Scaled    | Result            | Delta")
    print("------------------------------------------------------------------")

    for j in range(len(test_obtained)):
        expected_str = f'{test_expected[j]}'.ljust(16)

        expected_scaled = test_expected[j]
        if scale_function:
            expected_scaled = scale_function(test_expected[j], max(test_expected), min(test_expected))

        expected_scaled_str = f'{expected_scaled:.3f}'.ljust(18)
        result_str = f'{test_obtained[j]:.3f}'.ljust(17)
        delta_str = f'{abs(test_obtained[j] - expected_scaled):.3f}'.ljust(15)
        print(f'{expected_str} | {expected_scaled_str} | {result_str} | {delta_str}')

        if abs(test_obtained[j] - expected_scaled) <= eps:
            correct_predictions += 1

    print(f'Correct predictions: {correct_predictions} out of {len(test_expected)}')
    print(f'Training Error: {train_errors[-1]}')
    print(f'Testing Error: {test_mse}')


def generate_data_frame(config):
    """
    Generates the data frame based on the different learning_rates and epochs, for both perceptrons.
    """
    repeats = 10
    train_list = []
    config_aux = config.copy()

    for run in range(1, repeats + 1):
        for test_percentage in [0.2, 0.4, 0.6]:
            config_aux['test_percentage'] = test_percentage
            for learning_rate in [0.0001, 0.001, 0.01]:
                config_aux['learning_rate'] = learning_rate
                results = start(config_aux)

                train_lineal_results = results['train']['linear']
                for i in range(train_lineal_results['epoch']):
                    train_list.append({
                        "perceptron_type": 'Lineal',
                        "test_percentage": test_percentage,
                        "learning_rate": learning_rate,
                        "mse": train_lineal_results['train_errors'][i],
                        "epoch": i + 1,
                    })

                train_non_lineal_results = results['train']['non_linear']
                for i in range(train_non_lineal_results['epoch']):
                    train_list.append({
                        "perceptron_type": 'No lineal',
                        "test_percentage": test_percentage,
                        "learning_rate": learning_rate,
                        "mse": train_non_lineal_results['train_errors'][i],
                        "epoch": i + 1,
                    })
    return pd.DataFrame(train_list)


def start(config):
    learning_rate, test_percentage, epoch_limit, beta, eps = get_config_params(config)

    train_set, train_expected_set, test_set, test_expected_set = initialize_data(test_percentage)
    dim = len(train_set[0])

    linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
    non_linear_perceptron = NonLinearPerceptron(dim, beta, learning_rate, epoch_limit, eps)

    # (epoch, train_errors, test_errors)
    linear_train_output = linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set, False)
    non_linear_train_output = non_linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set,
                                                          True)

    # (result, test_mse)
    linear_test_output = linear_perceptron.predict(test_set, test_expected_set, scale=False)
    non_linear_test_output = non_linear_perceptron.predict(test_set, test_expected_set, scale=True)

    result_denormalized, test_mse = [
        non_linear_perceptron.denormalize_value(y, min(non_linear_test_output[0]), max(non_linear_test_output[0])) for y
        in non_linear_test_output[0]], non_linear_test_output[1]
    non_linear_test_output = (result_denormalized, test_mse)

    return {
        'eps': eps,
        'train': {
            'linear': {
                'epoch': linear_train_output[0],
                'train_errors': linear_train_output[1],
                'test_errors': linear_train_output[2]
            },
            'non_linear': {
                'epoch': non_linear_train_output[0],
                'train_errors': non_linear_train_output[1],
                'test_errors': non_linear_train_output[2]
            }
        },
        'test': {
            'x_values': test_set,
            'y_values': test_expected_set,
            'linear': {
                'result': linear_test_output[0],
                'test_mse': linear_test_output[1]
            },
            'non_linear': {
                'result': non_linear_test_output[0],
                'test_mse': non_linear_test_output[1]
            }
        }
    }


def calculate_accuracy(y_true, y_pred, epsilon):
    """Calcula el accuracy como la proporción de predicciones correctas."""
    correct_predictions = np.sum(np.abs(y_true - y_pred) <= epsilon)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


# Función para normalizar los datos de MSE
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def plot_mse_curves(df, learning_rate=0.0001):
    df = df[df['learning_rate'] == learning_rate]
    df = df[df['test_percentage'] == 0.2]

    # Agrupar por 'perceptron_type' y 'epoch', luego calcular la media de 'mse'
    stats = df.groupby(['perceptron_type', 'epoch'])['mse'].agg(['mean', 'std']).reset_index()
    stats['mse_normalized'] = stats.groupby('perceptron_type')['mean'].transform(normalize)
    # stats['std_normalized'] = stats.groupby('perceptron_type')['std'].transform(normalize)

    # Crear un gráfico para cada tipo de perceptrón
    fig, ax = plt.subplots(figsize=(10, 5))  # Tamaño del gráfico

    # Separar los datos por tipo de perceptrón y graficar
    for label, group_df in stats.groupby('perceptron_type'):
        # group_df.sort_values('epoch', inplace=True)  # Asegurarse de que los datos están ordenados por época
        ax.plot(group_df['epoch'], group_df['mse_normalized'], label=label, marker='o')

        # ax.errorbar(group_df['epoch'], group_df['mse_normalized'], yerr=group_df['std_normalized'], label=label, marker='o', fmt='-o', capsize=5)

    # Añadir título y etiquetas
    ax.set_title('MSE Medio por Época para cada Tipo de Perceptrón con learning_rate = 0.0001 y 20% de test')
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Medio')
    ax.legend(title='Tipo de Perceptrón')  # Añadir leyenda con título

    # Mostrar el gráfico
    plt.show()


def plot_accuracies(cofig):
    test_results = start(config)['test']
    eps = test_results['eps']
    epochs = test_results['epoch']

    x_values = test_results['x_values']
    y_values = test_results['y_values']

    linear_results = test_results['linear']['result']
    non_linear_results = test_results['non_linear']['result']

    total_predictions = len(x_values)

    correct_predictions = [(y_value - obtained) < eps for y_value, obtained in zip(y_values, linear_results)]
    print(correct_predictions)

    fig, ax = plt.subplots()

    ax.set_xlabel('Generation')
    ax.set_ylabel('Accuracy')
    ax.set_title('Evolution of Results\' Accuracy by Generation')

    generations = np.array(range(len(accuracies)))

    ax.set_xlim(0, len(generations))
    ax.set_ylim(0, np.amax(accuracies))

    ax.plot(generations, accuracies, color='b')
    plt.show()


if __name__ == '__main__':
    with open('ejercicio2/config.json', 'r') as f:
        config = json.load(f)

    # learning_rate, test_percentage, epoch_limit, beta, eps = get_config_params(config)

    # show_output(linear_train_output[1], linear_test_output[0], linear_test_output[1], test_expected_set, eps, None)
    # show_output(non_linear_train_output[1], non_linear_test_output[0], non_linear_test_output[1], test_expected_set, eps, ...)

    df = generate_data_frame(config)
    plot_mse_curves(df)
