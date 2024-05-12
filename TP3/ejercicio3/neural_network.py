import numpy as np

from Layer import Layer


class NeuralNetwork:
    ERROR_MIN = 0.0001

    def __init__(self, neurons_per_layer: list, learning_rate, activation, verbose=False):
        self.verbose = verbose
        self.error_min = np.inf
        self.layers = np.array(
            list(
                Layer(neurons_per_layer[i], neurons_per_layer[i - 1], activation, learning_rate)
                for i in range(1, len(neurons_per_layer))))

    def get_inputs(self, current_layer, inputs):
        """
        Retornamos la salida de la anterior Layer, o en caso que no haya anterior Layer, de la entrada a la red.
        """
        index = current_layer - 1
        if index < 0:
            return inputs
        else:
            return self.layers[index].get_all_outputs()

    def get_neuron_error(self, current_layer, current_neuron, expected_value):
        """
        Determinamos la diferencia entre el valor esperado y la salida real de una neurona, que se utilizará para ajustar los pesos de la red.
        """
        if current_layer + 1 < len(self.layers):
            # Capa oculta => Error calculado en base a las neuronas siguientes.
            return self.layers[current_layer + 1].get_neuron_delta(current_neuron)
        else:
            # Si es la última layer => Es la capa de salida => Error calculado como la diferencia entre lo esperado y obtenido.
            neuron = self.layers[current_layer].neurons[current_neuron]
            
            return expected_value - neuron.output

    def plot(self):
        for layer in self.layers:
            layer.plot()

    def forward_propagation(self, inputs):
        for index, layer in enumerate(self.layers):
            layer.propagation(self.get_inputs(index, inputs))

    def back_propagation(self, data, expected_value):
        # Recorremos desde la capa de salida hasta la de inicio.
        for layer_index in range(len(self.layers) - 1, -1, -1):
            # Inputs de la actual Layer.
            inputs = self.get_inputs(layer_index, data)
            for index, neuron in enumerate(self.layers[layer_index].neurons):
                
                neuron_error = self.get_neuron_error(layer_index, index, expected_value)
                if self.verbose:
                    print(f"Layer index: {layer_index} | Neuron Index: {index} | Neuron.output: {neuron.output} | Expected value: {expected_value} | Neuron error: {neuron_error}")
                neuron.update_w(inputs, neuron_error)

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].output) ** 2
        return aux_sum

    def train(self, x_values, expected_output, epoch_limit, verbose=False):
        epoch = 0
        errors = []
        positions = np.arange(0, len(x_values))

        while self.error_min > self.ERROR_MIN and epoch < epoch_limit:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(x_values[i])
                self.back_propagation(x_values[i], expected_output[i])

                error = self.calculate_error(expected_output)
                if self.verbose:
                    print(f"{epoch}/{epoch_limit} | Error: {error}")

                if error < self.error_min:
                    self.error_min = error
                    errors.append(float(error))
                    #if self.error_min < self.ERROR_MIN:
                        #return errors

            epoch += 1

        return errors

    def save(self, filepath):
        file = open(filepath, "w+")
        for layer in self.layers:
            for neuron in layer.neurons:
                wcount = 1
                for weight in neuron.weights:
                    file.write("w%d: %d" % wcount % weight)
                    wcount += 1
                file.write("\n")
        print("termine!")

        file.close()

    def predict(self, test_set):
        to_return = []
        for data in test_set:
            self.forward_propagation(data)
            to_return.append(self.layers[-1].get_all_outputs())
        return to_return
