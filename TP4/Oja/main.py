from Oja.oja_perceptron import OjaPerceptron
import numpy as np
import pandas as pd

def main():
    # Cargar los datos desde un archivo CSV
    data = pd.read_csv('europe.csv')

    # Extraer las características (sin la columna 'Country')
    features = data.iloc[:, 1:].values

    # Normalizar los datos (restar la media y dividir por la desviación estándar)
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features_normalized = (features - features_mean) / features_std

    print(features_normalized)

    # Inicializar el perceptrón con la regla de Oja
    perceptron = OjaPerceptron(input_size=features_normalized.shape[1], learning_rate=0.001)

    # Entrenar el perceptrón y # obtenemos la primera componente principal
    principal_component = perceptron.train(features_normalized, epochs=4000)

    # Mostrar la primera componente principal
    print("Primera componente principal:", principal_component)

if __name__ == "__main__":
    main()
