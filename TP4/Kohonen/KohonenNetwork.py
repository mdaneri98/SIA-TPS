import numpy as np
from KohonenNode import KohonenNode


class KohonenNetwork:
    def __init__(self, input_data, input_size, k, learning_rate, radius):
        self.input_data = input_data
        self.input_size = input_size
        self.k = k
        self.learning_rate = learning_rate
        self.radius = radius
        self.grid_shape = (k, k)
        self.neuron_matrix = self.initialize_neurons()

    def initialize_neurons(self):
        neuron_matrix = [[KohonenNode(self.get_random_sample(), self.learning_rate) for _ in range(self.k)] for _ in
                         range(self.k)]
        return np.array(neuron_matrix)

    def get_random_sample(self):
        return np.random.rand(self.input_size).tolist()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for input_vector in self.input_data:
                bmu_position, bmu_similarity = self.best_matching_neuron(input_vector)
                self.update_neighborhood(bmu_position, input_vector, bmu_similarity, epoch, num_epochs)

    def update_neighborhood(self, bmu_position, input_vector, bmu_similarity, epoch, num_epochs):
        for i in range(self.k):
            for j in range(self.k):
                node_position = np.array([i, j])
                distance_to_bmu = np.linalg.norm(node_position - bmu_position)
                if distance_to_bmu <= self.radius:
                    node = self.neuron_matrix[i, j]
                    influence = bmu_similarity * np.exp(-epoch / num_epochs)
                    node.update_weights(input_vector, influence)

    def best_matching_neuron(self, input_vector):
        distances = np.linalg.norm(input_vector - np.array([node.weights for row in self.neuron_matrix for node in row]), axis=1)
        bmu_index = np.argmin(distances)
        bmu_position = np.unravel_index(bmu_index, self.grid_shape)
        bmu_node = self.neuron_matrix[bmu_position[0], bmu_position[1]]
        bmu_similarity = bmu_node.calculate_similarity(input_vector)
        return bmu_position, bmu_similarity

    def predict(self, input_vector):
        # Encuentra el nodo más cercano (BMN) y calcula la similitud para una muestra de entrada dada
        bmu_position, bmu_similarity = self.best_matching_neuron(input_vector)
        return bmu_position, bmu_similarity

    def get_neighbors(self, position, radius):
        x, y = position
        neighbors = []

        for i in range(self.k):
            for j in range(self.k):
                neighbor_position = (i, j)
                distance = self.neuron_matrix[i][j].calculate_euclidean_distance(self.neuron_matrix[x][y].weights)
                if distance <= radius:
                    neighbors.append((neighbor_position, distance))

        return neighbors

    def calculate_unified_distances(self, radius):
        ud_matrix = np.zeros((self.k, self.k))

        for i in range(self.k):
            for j in range(self.k):
                position = (i, j)
                neighbors = self.get_neighbors(position, radius)
                total_distance = sum(dist for _, dist in neighbors)
                average_distance = total_distance / len(neighbors)
                ud_matrix[i][j] = average_distance

        return ud_matrix
