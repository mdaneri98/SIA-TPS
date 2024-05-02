from Layer import Layer
import numpy as np

class MultiPerceptron:
    ERROR_MIN = 0.01

    def __init__(self, net_config, learn_rate, activation):

        self.layers = np.array(
            [Layer(net_config[i], net_config[i - 1], activation, learn_rate) for i in range(1, len(net_config))])

    def get_inputs(self, current_layer, inputs):
        index = current_layer - 1
        if index < 0:
            return inputs
        else:
            return self.layers[index].get_all_outputs()

    def get_expected(self, current_layer, current_neuron, expected_value):
        if current_layer + 1 < len(self.layers):
            return self.layers[current_layer + 1].get_neuron_delta(current_neuron)
        else:
            neuron = self.layers[current_layer].neurons[current_neuron]
            return expected_value[current_neuron] - neuron.output

    def plot(self):
        for layer in self.layers:
            layer.plot()

    def forward_propagation(self, inputs):
        layer_index = 0
        for layer in self.layers:
            layer.propagation(self.get_inputs(layer_index, inputs))
            layer_index += 1

    def back_propagation(self, data, expected_value):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            inputs = self.get_inputs(layer_index, data)
            neuron_index = 0
            for neuron in self.layers[layer_index].neurons:
                neuron_error = self.get_expected(layer_index, neuron_index, expected_value)
                neuron.update_w(inputs, neuron_error)
                neuron_index += 1

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].output) ** 2
        return aux_sum

    def train(self, training, expected_output, max_gen):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        errors = []
        positions = np.arange(0, len(training))

        while self.error_min > self.ERROR_MIN and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(training[i])
                self.back_propagation(training[i], expected_output[i])

                error = self.calculate_error(expected_output[i])
                if error < self.error_min:
                    self.error_min = error
                    errors.append(float(error))
                    if self.error_min < self.ERROR_MIN:
                        print("Terminado en %d generaciones" % current_gen)
                        print("Error de %d" % self.error_min)
                        return errors

            current_gen += 1

        return errors

    def save(self, filepath):
        with open(filepath, "w+") as file:
            for layer in self.layers:
                for neuron in layer.neurons:
                    wcount = 1
                    for weight in neuron.weights:
                        file.write("w%d: %d" % (wcount, weight))
                        wcount += 1
                    file.write("\n")
        print("¡Terminado!")

    def test(self, test_set):
        to_return = []
        for data in test_set:
            self.forward_propagation(data)
            to_return.append(self.layers[-1].get_all_outputs())
        return to_return