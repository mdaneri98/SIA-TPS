from perceptrons.simple_perceptron import LinearPerceptron
from perceptrons.non_linear_perceptron import NonLinearPerceptron
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


# Funciones auxiliares.
def split_data(data, test_ratio):
    data_copy = data[:]
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * (1 - test_ratio))
    train_set = data_copy[:split_idx]
    test_set = data_copy[split_idx:]

    return train_set, test_set




def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def evaluate_non_linear_perceptron_epochs(train_data, test_data, learning_rate, epochs_list):
    results = []
    x_train = [features for features, _ in train_data]
    y_train = [label for _, label in train_data]
    x_test = [features for features, _ in test_data]
    y_test = [label for _, label in test_data]

    for epochs in epochs_list:
        # Entrenar el perceptrón
        perceptron = NonLinearPerceptron(train_data, learning_rate, epochs)
        perceptron.train()

        # Predecir en el conjunto de entrenamiento
        y_train_pred = np.array([perceptron.predict(x) for x in x_train])
        train_mse = mean_squared_error(np.array(y_train), y_train_pred)
        train_r2 = r2_score(np.array(y_train), y_train_pred)

        # Predecir en el conjunto de prueba
        y_test_pred = np.array([perceptron.predict(x) for x in x_test])
        test_mse = mean_squared_error(np.array(y_test), y_test_pred)
        test_r2 = r2_score(np.array(y_test), y_test_pred)

        results.append((epochs, train_mse, train_r2, test_mse, test_r2))

    return results


def evaluate():
    data_set_pd = pd.read_csv('datos.csv')

    data_set = []
    for _, row in data_set_pd.iterrows():
        x_values = row[['x1', 'x2', 'x3']].values
        y_value = row['y']
        data_set.append((x_values, y_value))

    train_set, test_set = split_data(data_set, 0.2)

    epochs_list = [1, 10, 20, 50, 100, 150, 200, 250, 300]
    linear_evaluation = evaluate_linear_perceptron_epochs(train_set, test_set, 0.2, epochs_list)
    non_linear_evaluation = evaluate_non_linear_perceptron_epochs(train_set, test_set, 0.2, epochs_list)

    return linear_evaluation, non_linear_evaluation



def graph_evaluation_non_linear(eval_data):
    # Preparar los datos para graficar
    epochs = [result[0] for result in eval_data]
    train_mse = [result[1] for result in eval_data]
    train_r2 = [result[2] for result in eval_data]
    test_mse = [result[3] for result in eval_data]
    test_r2 = [result[4] for result in eval_data]

    # Crear gráficos
    plt.figure(figsize=(10, 5))

    # Gráfico para MSE
    plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer gráfico
    plt.plot(epochs, train_mse, label='Train MSE')
    plt.plot(epochs, test_mse, label='Test MSE')
    plt.title('Train vs Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Gráfico para R²
    plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segundo gráfico
    plt.plot(epochs, train_r2, label='Train R²')
    plt.plot(epochs, test_r2, label='Test R²')
    plt.title('Train vs Test R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()


linear_eval, non_linear_eval = evaluate()
graph_evaluation_non_linear(non_linear_eval)
graph_evaluation_linear(linear_eval)
