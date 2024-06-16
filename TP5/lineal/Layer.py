import numpy as np
from Neuron import Neuron


class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = np.array([Neuron(prev_num_neurons, activation, learn_rate) for _ in range(num_neurons)])

    def get_all_outputs(self):
        outputs = []
        for current_neuron in self.neurons:
            outputs.append(current_neuron.output)
        return outputs

    def get_neuron_delta(self, num_neuron):
        delta = 0
        for current_neuron in self.neurons:
            delta += (current_neuron.weights[num_neuron] * current_neuron.delta)
        return delta

    def propagation(self, inputs):
        for neuron in self.neurons:
            neuron.calculate_output(inputs)

    def plot(self):
        print("Layer " + " : ")
        for neuron in self.neurons:
            neuron.plot()
        print("------------------------")
