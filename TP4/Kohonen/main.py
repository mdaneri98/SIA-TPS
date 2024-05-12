import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from KohonenNetwork import *


def get_data():
    data = pd.read_csv('europe.csv')
    countries = data["Country"].tolist()
    labels = data.columns[1:].tolist()
    country_data = data.iloc[:, 1:].values
    return countries, labels, country_data


def standarize_data(input_data):
    data_standardized = np.copy(input_data)  # Copia los datos para no modificar el original
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
    cax = ax.matshow(activation_matrix, cmap='viridis')
    ax.set_title('Heatmap of Countries')
    ax.set_xticks(np.arange(k + 1))
    ax.set_yticks(np.arange(k + 1))
    ax.set_xticklabels(list(range(k + 1)))
    ax.set_yticklabels(list(range(k + 1)))

    for i in range(k):
        for j in range(k):
            country_name = ''
            for country, position in zip(countries, input_data):
                bmu_position, similarity = network.predict(position)
                max_index = similarity.argmax()
                x = max_index // network.k  # División entera para obtener la fila
                y = max_index % network.k
                if x == i and y == j:
                    country_name += f'{country}\n'
            ax.text(j, i, activation_matrix[i, j], ha='center', va='center', color='black')

    cbar = fig.colorbar(cax)
    cbar.set_label('Frequency')

    plt.show()


def plot_average_neighbor_distances(ud_matrix, k):
    cmap = plt.get_cmap('gray')
    plt.imshow(ud_matrix, cmap=cmap)

    min_val = min(min(row) for row in ud_matrix)
    max_val = max(max(row) for row in ud_matrix)
    color_thr = min_val + 0.8 * (max_val - min_val)

    for i in range(k):
        for j in range(k):
            color = 'w' if ud_matrix[i][j] < color_thr else 'k'
            plt.text(j, i, str(f"{ud_matrix[i][j]:.2f}"), ha='center', va='center', color=color)

    cbar = plt.colorbar()
    cbar.set_label('Valor')

    plt.show()



def plot_neighbor_distances(network, radius):
    all_distances = []

    for i in range(network.k):
        for j in range(network.k):
            position = (i, j)
            neighbors = network.get_neighbors(position, radius)
            all_distances.extend(dist for _, dist in neighbors)

    plt.hist(all_distances, bins=20)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Neighboring Neurons')
    plt.show()

def main():
    countries, labels, data = get_data()

    standarized_data = standarize_data(data)

    k = 8
    learning_rate = 0.01
    initial_radius = 3
    max_epochs = 1000

    network = KohonenNetwork(standarized_data, len(standarized_data[0]), k, learning_rate, initial_radius)

    network.train(max_epochs)

    plot_heatmap(network, standarized_data, countries)

    plot_neighbor_distances(network, initial_radius)

    plot_average_neighbor_distances(network.calculate_unified_distances(initial_radius), k)


if __name__ == "__main__":
    main()