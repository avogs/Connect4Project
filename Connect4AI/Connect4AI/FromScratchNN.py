import numpy as np

class FromScratchNN(object):
    """A class which can be used to make and train a neural network. Built using minimal external libraries.
    
    This is being built as a project to review/ dive deeper into python coding and deep learning internals."""

    def __init__(self, layerSizes):
        """Initializes a list of layers representing a neural network.
       
        Goes from from an input layer of length layerSizes[0] to an
        output layer of length of layerSizes[len(layerSizes)]. These
        layers are initialized with 0 as all their parameters."""
        self.layers = [] # A list of layer objects, ordered from input layer to output layer. Will always have at least 1 element.
        for x in range(1,len(layerSizes)):
            self.layers[x-1] = layer(layerSizes[x-1], layerSizes[x])


    def predict(self, input):
        """Takes in a data point and runs it through the current model, returns the output of that."""
        nextOutput = input
        for x in range(len(self.layers)):
            nextOutput = self.layers[x].evaluate(nextOutput)
        return nextOutput


    #def trainGradientDescent(self, data, labels, gradientSpacing, learningRate):
    #    """ """
    #    # Note that this uses a numerical gradient which can give fairly bad results and will usually be highly time intensive.
    #    # A more advanced method should use automatic differentiation or something similar but that is out of the scope of this project (for now).
    #    np.gradient(self.parameterSpaceCostField(gradientSpacing))


    #def parameterSpaceCostField(self, gradientSpacing, size):
    #    """ """
    #    numParams = 0
    #    for layer in self.layer:
    #        numParams += len(layer.biases)
    #        numParams += len(layer.weights)
    #    field = np.zeros((3, numParams))
    #    
    #    for layer in self.layers:
    #        for bias in layer.bias:
    #            bias += gradientSpacing
    #            cost(data, labels)


    def cost(self, data, labels):
        """A function which takes in labelled data and outputs the sum of squared errors on that for the current model."""
        SSE = 0
        for x in range(len(data)):
            SSE += (labels[x] - self.predict(data[x]))**2
        return (SSE/len(data))


    def ReLu(self, input):
        """Puts the input through a Rectified Linear Unit. (converts each input element to 0.0 if negative.)"""
        for x in range(len(input)):
            if input[x] < 0:
                input[x] = 0


    class layer(object):
        """A subclass which holds a layer of a neural network."""

        __init__(self, inputLen, outputLen):
            """Initializes a layer's weights and biases. (maybe should be random instead of zero?)"""
            self.weights = np.zeros((inputLen, outputLen))
            self.biases = np.zeros(inputLen)

        def evaluate(self, input):
            """Takes an input to the layer and returns the ReLu(output of the layer)."""
            return ReLu(self.weights @ input + self.biases)


    # relu(W1A1 + W2A2 + W3A3... + WnAn - bias)
    # A's are a long column vector of activiations, w's are from a KxN matrix, each row
    # of which represents the weights to multiply each a by to get output for the next layer
    # [w00, w01, ..., w0n][a0]  + [b0]
    # [w10, w11, ..., w1n][a1]  + [b1]
    # [..., ..., ..., ...][...] + [...]
    # [wk0, wk1, ..., wkn][an]  + [bn]
    # k is number of neurons in next layer
    # multiplying these matrices then add the bias and apply sigmoid to get the next activation layer.
    # 
    # Cost will be squared sum of differences between outputs and desired outputs
    # Consider average cost over training data, minimize that.
    # Compute the gradient of the cost function and increase/decrease each weight and bias proportionally to it's value in the negative gradient.


    # This is a basic multilayer perceptron