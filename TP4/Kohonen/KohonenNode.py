import numpy as np


class KohonenNode:
    def __init__(self, weights, learning_rate: float = 0.1):
        # Inicializa los pesos con los valores proporcionados
        self.weights = np.array(weights)
        self.activation = 0.0
        self.learning_rate = learning_rate

    def calculate_similarity(self, input_vector):
        # Calcula la similitud entre el vector de entrada y los pesos del nodo
        return np.exp(-np.linalg.norm(input_vector - self.weights) ** 2)

    def calculate_euclidean_distance(self, input_vector):
        # Calcula la distancia euclidiana entre el vector de entrada y los pesos del nodo
        return np.linalg.norm(input_vector - self.weights)

    def update_weights(self, input_vector):
        # Actualiza los pesos del nodo en función del vector de entrada, la influencia y la tasa de aprendizaje
        self.weights += self.learning_rate * (input_vector - self.weights)
