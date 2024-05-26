import numpy as np
from KohonenNode import KohonenNode


class KohonenNetwork:
    def __init__(self, input_data, input_size, k, learning_rate, radius, init_type='random'):
        self.input_data = input_data
        self.input_size = input_size
        self.k = k
        self.initial_radius = radius
        self.learning_rate = learning_rate
        self.radius = radius
        self.grid_shape = (k, k)
        if init_type == 'data_samples':
            self.neuron_matrix = self.initialize_neurons_with_input_data()
        else:  # default to random initialization
            self.neuron_matrix = self.initialize_neurons()

    def initialize_neurons(self):
        neuron_matrix = [[KohonenNode(self.get_random_sample(), self.learning_rate) for _ in range(self.k)] for _ in
                         range(self.k)]
        return np.array(neuron_matrix)
    
    def initialize_neurons_with_input_data(self):
        data_sample_indices = np.random.choice(len(self.input_data), size=self.k * self.k, replace=False)
        selected_samples = self.input_data[data_sample_indices]
        neuron_matrix = [[KohonenNode(selected_samples[i * self.k + j], self.learning_rate) for j in range(self.k)] for i in range(self.k)]
        return np.array(neuron_matrix)


    def get_random_sample(self):
        return np.random.rand(self.input_size).tolist()

    def train(self, num_epochs, similitud='euclidean'):
        similarities = []
        for epoch in range(num_epochs):
            total_similarity = 0
            for input_vector in self.input_data:
                bmu_position, bmu_similarity = self.best_matching_neuron(input_vector, similitud)
                total_similarity += bmu_similarity
                if self.radius > 1:
                    self.radius = max(1, self.initial_radius / (epoch + 1))
                self.update_neighborhood(bmu_position, input_vector, bmu_similarity, epoch, num_epochs)
            average_similarity = total_similarity / len(self.input_data)
            similarities.append(average_similarity)
        return similarities

    def update_neighborhood(self, bmu_position, input_vector, bmu_similarity, epoch, num_epochs):
        for i in range(self.k):
            for j in range(self.k):
                node_position = np.array([i, j])
                distance_to_bmu = np.linalg.norm(node_position - bmu_position)
                if distance_to_bmu <= self.radius:
                    node = self.neuron_matrix[i, j]
                    node.update_weights(input_vector)

    def best_matching_neuron(self, input_vector,similitud):
        if (similitud == 'euclidean'):
            distances = np.linalg.norm(input_vector - np.array([node.weights for row in self.neuron_matrix for node in row]), axis=1)
        else : 
            distances = np.linalg.norm(input_vector - np.array([node.weights for row in self.neuron_matrix for node in row]), axis=1)
            distances = np.square(distances)
            distances = np.exp(-distances)
        bmu_index = np.argmin(distances)
        bmu_position = np.unravel_index(bmu_index, self.grid_shape)
        bmu_node = self.neuron_matrix[bmu_position[0], bmu_position[1]]
        bmu_similarity = bmu_node.calculate_similarity(input_vector)
        return bmu_position, bmu_similarity

    def predict(self, input_vector,similitud ='euclidean'):
        # Encuentra el nodo más cercano (BMN) y calcula la similitud para una muestra de entrada dada
        bmu_position, bmu_similarity = self.best_matching_neuron(input_vector,similitud)
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
