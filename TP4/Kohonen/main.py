import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from KohonenNetwork import *
import seaborn as sns


def get_data():
    data = pd.read_csv('europe.csv')
    countries = data["Country"].tolist()
    labels = data.columns[1:].tolist()
    country_data = data.iloc[:, 1:].values
    return countries, labels, country_data


def standarize_data(input_data):
    data_standardized = np.copy(input_data)
    means = np.mean(data_standardized, axis=0)
    stdevs = np.std(data_standardized, axis=0)
    data_standardized = (data_standardized - means) / stdevs
    return data_standardized


def plot_heatmap(network, input_data, countries):
    k = network.k

    activation_matrix = np.zeros((k, k), dtype=int)
    country_count = {country: 0 for country in countries}

    for i, input_vector in enumerate(input_data):
        bmu_position, similarity = network.predict(input_vector)
        x, y = bmu_position
        activation_matrix[x][y] += 1
        country_count[countries[i]] += 1

    fig, ax = plt.subplots()
    cax = ax.matshow(activation_matrix, cmap='viridis', origin='lower', extent=[0, k, 0, k])
    ax.set_title('Heatmap of Countries')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(np.arange(1, k + 1))  # Ajusta las etiquetas del eje x
    ax.set_yticklabels(np.arange(1, k + 1))  # Ajusta las etiquetas del eje y

    for i in range(k):
        for j in range(k):
            text = activation_matrix[i, j]
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')

    cbar = fig.colorbar(cax)

    plt.show()


def plot_average_neighbor_distances(ud_matrix, k):
    cmap = plt.get_cmap('gray')
    plt.imshow(ud_matrix, cmap=cmap, extent=[0, k, 0, k], origin='lower')

    min_val = min(min(row) for row in ud_matrix)
    max_val = max(max(row) for row in ud_matrix)
    color_thr = min_val + 0.8 * (max_val - min_val)

    for i in range(k):
        for j in range(k):
            color = 'w' if ud_matrix[i][j] < color_thr else 'k'
            plt.text(j + 0.5, i + 0.5, "", ha='center', va='center', color=color)

    cbar = plt.colorbar()

    plt.show()


def analyze_association(network, data):
    association_count = np.zeros(network.grid_shape)
    for sample in data:
        bmu_position, _ = network.predict(sample)
        association_count[bmu_position] += 1
    return association_count


def plot_confusion_matrix(association_count):
    plt.figure(figsize=(8, 6))
    sns.heatmap(association_count, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Columna de Neurona")
    plt.ylabel("Fila de Neurona")
    plt.title("Matriz de Confusión")
    plt.show()


def analyze_variable(network, input_data, variable_index, variables):
    k = network.k
    association_count = analyze_association(network, input_data[:, [variable_index]])

    plt.figure(figsize=(8, 6))
    plt.imshow(association_count, cmap='Blues', origin='lower', extent=[0, k, 0, k], vmin=association_count.min())
    plt.colorbar()
    plt.title(f'Heatmap for "{variables[variable_index]}"')

    plt.show()


def main():
    countries, labels, data = get_data()
    standard_data = standarize_data(data)
    variables = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    k = 4
    learning_rate = 0.01
    initial_radius = 2
    max_epochs = 1000

    # Network 4x4
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    network.train(max_epochs)

    plot_heatmap(network, standard_data, countries)
    plot_average_neighbor_distances(network.calculate_unified_distances(initial_radius), k)
    for i, variable in enumerate(variables):
        analyze_variable(network, standard_data, i, variables)

    # Network 5x5
    k = 5
    network2 = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    network2.train(max_epochs)
    plot_heatmap(network2, standard_data, countries)
    plot_average_neighbor_distances(network2.calculate_unified_distances(initial_radius), k)


if __name__ == "__main__":
    main()
