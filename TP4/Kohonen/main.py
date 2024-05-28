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


def plot_heatmap(network, input_data, countries,similitud = 'euclidean'):
    k = network.k

    # Dictionnaire pour stocker les pays mappés à chaque neurone
    neuron_countries = { (i, j): [] for i in range(k) for j in range(k) }

    for i, input_vector in enumerate(input_data):
        bmu_position, similarity = network.predict(input_vector,similitud)
        x, y = bmu_position
        neuron_countries[(x, y)].append(countries[i])

    # Création de la matrice d'activation pour l'affichage
    activation_matrix = np.zeros((k, k), dtype=int)
    for position, country_list in neuron_countries.items():
        x, y = position
        activation_matrix[x, y] = len(country_list)

    fig, ax = plt.subplots()
    cax = ax.matshow(activation_matrix, cmap='viridis', origin='lower', extent=[0, k, 0, k])
    ax.set_title('Heatmap of Countries')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(np.arange(1, k + 1))  # Ajuste les étiquettes de l'axe x
    ax.set_yticklabels(np.arange(1, k + 1))  # Ajuste les étiquettes de l'axe y

    for i in range(k):
        for j in range(k):
            country_list = neuron_countries[(i, j)]
            if country_list:
                # Ajuster la taille de la police en fonction du nombre de pays
                font_size = max(10 - len(country_list), 5)
                text = "\n".join(country_list)
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=font_size)

    cbar = fig.colorbar(cax)
    plt.show()


def plot_average_neighbor_distances(ud_matrix, k):
    cmap = plt.get_cmap('gray')
    plt.imshow(ud_matrix, cmap=cmap, extent=[0, k, 0, k], origin='lower')

    min_val = np.min(ud_matrix)
    max_val = np.max(ud_matrix)
    color_thr = min_val + 0.8 * (max_val - min_val)

    for i in range(k):
        for j in range(k):
            color = 'w' if ud_matrix[i][j] < color_thr else 'k'
            plt.text(j + 0.5, i + 0.5, f"{ud_matrix[i][j]:.2f}", ha='center', va='center', color=color)

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


def analyze_variable(network, input_data, labels, variable):
    k = network.k

    activation_matrix = np.zeros((k, k))

    for i in range(len(input_data)):
        # Predict returns a tuple with the position of the BMU
        bmu_position, _ = network.predict(input_data[i])
        x, y = bmu_position
        activation_matrix[x][y] += input_data[i][variable]

    min_val = np.min(activation_matrix)
    if min_val < 0:
        activation_matrix -= min_val


    cmap = plt.get_cmap('Blues')

    plt.xticks(range(k), range(k))
    plt.yticks(range(k), range(k))

    # Create a norm with integer boundaries
    min_val = np.min(activation_matrix)
    max_val = np.max(activation_matrix)
    norm = BoundaryNorm(np.arange(min_val, max_val + 1), cmap.N)

    plt.imshow(activation_matrix, cmap=cmap)

    for i in range(k):
        for j in range(k):
            cell_color = cmap(norm(activation_matrix[i][j]))
            brightness = np.linalg.norm(cell_color[:3])  # Calculate luminance (brightness)
            text_color = 'k' if brightness >= 0.5 else 'w'  # Determine text color
            annotation = f"{activation_matrix[i, j]:.2f}"
            plt.text(j, i, annotation, ha='center', va='center', color=text_color)

    plt.title(f"Heatmap \"{labels[variable]}\"")
    plt.colorbar()
    plt.show()


def main():
    countries, labels, data = get_data()
    standard_data = standarize_data(data)
    variables = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    k = 3
    learning_rate = 0.01
    initial_radius = 3
    max_epochs = 3500

    # Network 3x3
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs)

    plot_heatmap(network, standard_data, countries)

    
    
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs,'exp')
    plot_heatmap(network, standard_data, countries,'exp')

    k = 4
    initial_radius = 4
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs)
    plot_heatmap(network, standard_data, countries)
    plot_average_neighbor_distances(network.calculate_unified_distances(initial_radius), k)
    for i, variable in enumerate(variables):
        analyze_variable(network, standard_data, labels, i)


    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs,'exp')
    plot_heatmap(network, standard_data, countries,'exp')


    k = 5
    initial_radius = 5
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs)
    plot_heatmap(network, standard_data, countries)
    plot_average_neighbor_distances(network.calculate_unified_distances(initial_radius), k)
    network = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius)
    _=network.train(max_epochs,'exp')
    plot_heatmap(network, standard_data, countries,'exp')

    num_epochs = 1000
    num_repeats = 10
    all_similarities_random = []
    all_similarities_data_samples = []

    for _ in range(num_repeats):
        # Réseau avec initialisation aléatoire
        network_random = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius, init_type='random')
        similarities_random = network_random.train(num_epochs, similitud='euclidean')
        all_similarities_random.append(similarities_random)

        # Réseau avec initialisation par échantillons de données
        network_data_samples = KohonenNetwork(standard_data, len(standard_data[0]), k, learning_rate, initial_radius, init_type='data_samples')
        similarities_data_samples = network_data_samples.train(num_epochs, similitud='euclidean')
        all_similarities_data_samples.append(similarities_data_samples)

    # Calculer les similarités moyennes pour chaque méthode d'initialisation
    mean_similarities_random = np.mean(all_similarities_random, axis=0)
    mean_similarities_data_samples = np.mean(all_similarities_data_samples, axis=0)

    # Tracer les similarités moyennes
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, mean_similarities_random, label='Inicialización random - Euclideana')
    plt.plot(epochs, mean_similarities_data_samples, label='Inicialización respecto de los datos - Euclideana')
    plt.xlabel('Épocas')
    plt.ylabel('Similitud media')
    plt.title('Evolución de similitudes durante el entrenamiento (10 ejecuciones)')
    plt.legend()
    plt.show()

    
    


if __name__ == "__main__":
    main()
