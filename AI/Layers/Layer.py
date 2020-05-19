import random
import numpy as np
from AI.NeuralNetworkPackage.NeuralNetworkConstants import NeuralNetworkConstants as Const
from AI.NeuralNetworkPackage.ActivationFunction import sigmoidFunc
from AI.NeuralNetworkPackage.ActivationFunction import dSigmoidFunc

class layer:
    weights = list()
    deltaWeights = list()
    def weightsInit(self, inputSize, outputSize):
        weights = list()
        for i in range((inputSize+1)*outputSize):
            weights.append(random.uniform(-2, 2))
        return weights

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize+1
        self.outputSize = outputSize
        self.weights = self.weightsInit(inputSize, outputSize)
        self.deltaWeights = np.zeros((inputSize+1)*outputSize)


    def run(self, input):
        self.input = input.copy()
        self.output = []
        self.input.append(1.0)
        for i in range(self.outputSize):
            tmp = 0.0
            for j in range(self.inputSize):
                tmp += self.input[j]*self.weights[j+i*self.inputSize]
            self.output.append(sigmoidFunc(tmp))
        return self.output.copy()


    def train(self, error):
        nextError = []
        for i in range(self.inputSize):
            nextError.append(0.0)
        for i in range(self.outputSize):
            delta = dSigmoidFunc(self.output[i])*error[i]
            for j in range(self.inputSize):
                nextError[j] = nextError[j] + delta*self.weights[j + i * self.inputSize]
                deltaWeight = NeuralNetworkConstants.learningRate * self.input[j] * delta
                self.weights[j + i * self.inputSize] += \
                    deltaWeight + NeuralNetworkConstants.momentum * self.deltaWeights[j + i * self.inputSize]
                self.deltaWeights[j + i * self.inputSize] = deltaWeight
        return nextError

    def save(self, n):
        name = 'Layer'
        name += n
        name += '_weights.npy'
        name1 = 'Layer'
        name1 += n
        name1 += '_deltaWeights.npy'
        np.save(name, np.array(self.weights))
        np.save(name1, np.array(self.deltaWeights))

    def load(self,n):
        name = 'Layer'
        name += n
        name += '_weights.npy'
        name1 = 'Layer'
        name1 += n
        name1 += '_deltaWeights.npy'
        self.weights = np.load(Const.layers_folder+"/"+name)
        self.deltaWeights = np.load(Const.layers_folder+"/"+name1)
