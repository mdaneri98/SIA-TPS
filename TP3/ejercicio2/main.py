from perceptrons.simple_perceptron import LinearPerceptron, NonLinearPerceptron
import pandas as pd
import numpy as np
import random
import pandas as pd



def split_data(data, test_ratio):
    # Asumiendo que esta función baraja y divide correctamente los datos
    np.random.seed(42)  # Para reproducibilidad en el barajado
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



def initialize_data(test_ratio):
    data = pd.read_csv('datos.csv')

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


def start():
    train_set, train_expected_set, test_set, test_expected_set = initialize_data(0.2)

    dim = len(train_set[0])
    learning_rate = 0.02
    epoch_limit = 100
    eps = 0.5
    beta = 1

    linear_perceptron = LinearPerceptron(dim, learning_rate, epoch_limit, eps)
    non_linear_perceptron = NonLinearPerceptron(dim, beta, learning_rate, epoch_limit, eps)

    linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set, False)
    non_linear_perceptron.train(train_set, train_expected_set, test_set, test_expected_set, True)

    correct_predictions = 0

    print("Expected         | Expected_Scaled    | Result            | Delta")
    print("------------------------------------------------------------------")
    for i, perceptron in enumerate([linear_perceptron, non_linear_perceptron]):
        epochs, train_errors, test_errors = perceptron.train(train_set, train_expected_set, test_set,
                                                             test_expected_set, False)
        result = []
        test_mse = None
        if i == 0:
            result, test_mse = perceptron.predict(test_set, test_expected_set)
        elif i == 1:
            result, test_mse = perceptron.predict(test_set, test_expected_set)

        for j in range(len(result)):
            expected_str = f'{test_expected_set[i]}'.ljust(16)

            expected_scaled = perceptron.normalize_value(test_expected_set[i], max(test_expected_set), min(test_expected_set))

            expected_scaled_str = f'{expected_scaled:.3f}'.ljust(18)
            result_str = f'{result[i]:.3f}'.ljust(17)
            delta_str = f'{abs(result[i] - expected_scaled):.3f}'.ljust(15)
            print(f'{expected_str} | {expected_scaled_str} | {result_str} | {delta_str}')

            if abs(result[i] - expected_scaled) <= eps:
                correct_predictions += 1

        print(f'Correct predictions: {correct_predictions} out of {len(result)}')
        print(f'Training Error: {train_errors[len(train_errors) - 1]}')
        print(f'Testing Error: {test_mse}')


if __name__ == '__main__':
    start()
