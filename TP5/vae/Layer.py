import numpy as np
import copy

from Activation import Sigmoid
from Optimizer import Adam

class Dense:
    def __init__(self, inputDim = 1, outputDim = 1, activation = Sigmoid(), optimizer = Adam()):
        self.inputDim = inputDim
        self.outputDim = outputDim

        # set the activation function
        self.activation = activation

        # set optimizer
        self.weightOptimizer = copy.copy(optimizer)
        self.biasOptimizer = copy.copy(optimizer)

        # randomly initialize the weight and biases
        limit = np.sqrt(6 / (inputDim + outputDim)) # xavier uniform initializer
        self.weight = np.random.uniform(-limit, limit,(outputDim, inputDim))
        self.bias   = np.zeros(outputDim)

        # trainable decides whether weight and biases are trained in backward pass
        self.trainable = True
        

    def feedforward(self, input):
        if input.ndim == 1:
            input = np.squeeze(input).reshape((input.shape[0], self.batchSize))

        self.input = input

        self.z = np.dot(self.weight, self.input) + np.tile(self.bias, (self.input.shape[1], 1)).T
        self.a = self.activation.apply(self.z)
        return self.a


    def backward(self, lastGradient, outputLayer = False, updateParameters = True):
        oldWeight = np.copy(self.weight)
        if not outputLayer:
            lastGradient *= self.activation.derivative(self.z)

        if self.trainable and updateParameters:
            # layer is NOT frozen, weights and biases should be changed
            gradWeight = np.dot(lastGradient, self.input.T)
            gradBias   = np.sum(lastGradient, axis=1)

            # update weights and biases with optimizer
            self.weightOptimizer.optimize(self.weight, gradWeight)
            self.biasOptimizer.optimize(self.bias, gradBias)

        self.gradient = np.dot(oldWeight.T, lastGradient)
        return self.gradient


    def numParameters(self):
        weightShape = self.weight.shape
        return weightShape[0]*weightShape[1] + self.bias.shape[0]


    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
        self.weightOptimizer.setLearningFactor(self.batchSize)
        self.biasOptimizer.setLearningFactor(self.batchSize)