import numpy as np
from Layer import Layer
from Optimazer import GradientDescentOptimizer
from Activation import Sigmoid

class Autoencoder:
    def __init__(self, encoder_layers, decoder_layers, learning_rate=0.01, activation=Sigmoid()):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = GradientDescentOptimizer(learning_rate)
        self.build_model()

    def build_model(self):
        self.encoder = []
        prev_neurons = self.encoder_layers[0]
        for neurons in self.encoder_layers[1:]:
            self.encoder.append(Layer(neurons, prev_neurons, self.activation, self.learning_rate))
            prev_neurons = neurons
        
        self.decoder = []
        prev_neurons = self.decoder_layers[0]
        for neurons in self.decoder_layers[1:]:
            self.decoder.append(Layer(neurons, prev_neurons, self.activation, self.learning_rate))
            prev_neurons = neurons

    def forward_propagation(self, inputs):
        for layer in self.encoder:
            layer.propagation(inputs)
            inputs = layer.get_all_outputs()
        encoded_output = inputs

        for layer in self.decoder:
            layer.propagation(inputs)
            inputs = layer.get_all_outputs()
        return encoded_output, inputs

    def back_propagation(self, data, expected_output):
        encoded_output, outputs = self.forward_propagation(data)
        error = np.array(expected_output) - np.array(outputs)

        # Backpropagation through decoder
        for layer_index in range(len(self.decoder) - 1, -1, -1):
            inputs = data if layer_index == 0 else self.decoder[layer_index - 1].get_all_outputs()
            for index, neuron in enumerate(self.decoder[layer_index].neurons):
                neuron_error = error[index] if layer_index == len(self.decoder) - 1 else self.decoder[layer_index + 1].get_neuron_delta(index)
                gradient = neuron.calculate_gradient(inputs, neuron_error)
                updated_weights = self.optimizer.update(neuron.weights, gradient)
                neuron.weights = updated_weights

        # Backpropagation through encoder
        for layer_index in range(len(self.encoder) - 1, -1, -1):
            inputs = data if layer_index == 0 else self.encoder[layer_index - 1].get_all_outputs()
            for index, neuron in enumerate(self.encoder[layer_index].neurons):
                neuron_error = error[index] if layer_index == len(self.encoder) - 1 else self.encoder[layer_index + 1].get_neuron_delta(index)
                gradient = neuron.calculate_gradient(inputs, neuron_error)
                updated_weights = self.optimizer.update(neuron.weights, gradient)
                neuron.weights = updated_weights

    def train(self, data, epochs=10000):
        for epoch in range(epochs):
            for i in range(len(data)):
                self.forward_propagation(data[i])
                self.back_propagation(data[i], data[i])
            if epoch % 1000 == 0:
                print(f'Epoch {epoch} complete.')

    def encode(self, data):
        for layer in self.encoder:
            layer.propagation(data)
            data = layer.get_all_outputs()
        return data

    def decode(self, encoded_data):
        for layer in self.decoder:
            layer.propagation(encoded_data)
            encoded_data = layer.get_all_outputs()
        return encoded_data
