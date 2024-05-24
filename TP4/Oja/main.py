from Oja.oja_perceptron import OjaPerceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca_index_by_country(countries, scores):
    plt.figure(figsize=(14, 8))
    plt.bar(countries, scores)
    plt.xlabel('Países')
    plt.ylabel('Índice de la Primera Componente Principal (PC1)')
    plt.title('Índice de la Primera Componente Principal según País')
    plt.xticks(rotation=90)
    plt.show()


def main():
    # Cargar los datos desde un archivo CSV
    data = pd.read_csv('europe.csv')

    # Extraer las características (sin la columna 'Country')
    features = data.iloc[:, 1:].values

    # Normalizar los datos (restar la media y dividir por la desviación estándar)
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features_normalized = (features - features_mean) / features_std

    # Inicializar el perceptrón con la regla de Oja
    perceptron = OjaPerceptron(input_size=features_normalized.shape[1], learning_rate=0.001)

    # Entrenar el perceptrón y # obtenemos la primera componente principal
    principal_component = perceptron.train(features_normalized, epochs=4000)

    # Mostrar la primera componente principal
    print("Primera componente principal:", principal_component)

    # Graficamos
    scores = [np.dot(principal_component, feature_normalized) for feature_normalized in features_normalized]
    plot_pca_index_by_country(countries=data['Country'].values, scores=scores)



if __name__ == "__main__":
    main()
