import numpy as np
from Neuron import Neuron


class Layer:
    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = np.array([Neuron(prev_num_neurons, activation, learn_rate) for _ in range(num_neurons)])

    def get_all_outputs(self):
        return [neuron.output for neuron in self.neurons]

    def get_neuron_delta(self, num_neuron):
        return sum(neuron.weights[num_neuron] * neuron.delta for neuron in self.neurons)

    def propagation(self, inputs):
        for neuron in self.neurons:
            neuron.calculate_output(inputs)