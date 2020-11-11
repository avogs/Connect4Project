import numpy as np

class FromScratchNN(object):
    """A class which can be used to make and train a neural network. Built using minimal external libraries.
    
    This is being built as a project to review/dive deeper into python coding and deep learning internals."""

    def __init__(self, layerSizes):
        """Initializes a list of layers representing a neural network.
       
        Goes from from an input layer of length layerSizes[0] to an
        output layer of length of layerSizes[len(layerSizes)]. These
        layers are initialized with 0 as all their parameters."""
        self.layers = [] # A list of layer objects, ordered from input layer to output layer. Should always have at least 1 element.
        for x in range(1,len(layerSizes)):
            self.layers.append(self.layer(layerSizes[x-1], layerSizes[x]))


    def predict(self, input):
        """Takes in a data point and runs it through the current model, returns the output of that."""
        nextOutput = input
        for x in range(len(self.layers)):
            nextOutput = self.layers[x].evaluate(nextOutput)
        return nextOutput


    def trainGradientDescent(self, data, labels, gradientSpacing, learningRate, epochs):
        """A simple gradient descent implementation, takes data, labels, a spacing scalar for the numeric gradient evaluation, a learning rate scalar, and a number of iterations.
        
        A FromScratchNN object should first be made and given layer parameters, then this can be invoked and the parameters will be procedurally updated
        for each epoch. Note that the numerical gradient used is time intensive and may yield subpar results on very small spacing due to rounding errors.
        This uses a sum of squared errors cost function by default, others may be added later."""
        for x in range(epochs):
            # Print status at start of epoch
            print("Beginning epoch: " + str(x))
            print("Current cost: " + str(self.cost(data, labels)))
            # Arrange parameters into an array and take the gradient in parameter space to see which change will decrease the average cost the most.
            negativeGradients = -self.numericGradient(self.parameterSpaceCostField(gradientSpacing, data, labels), gradientSpacing)
            paramToChange = np.argmax(negativeGradients)
            # Change the parameter with the highest negative gradient
            for layer in self.layers:
                paramToChange -= len(layer.biases)
                if (paramToChange < 0):
                    paramToChange += len(layer.biases)
                    layer.biases[paramToChange] += negativeGradients[paramToChange]*learningRate
                    break
                for i in range(len(layer.weights)):
                    paramToChange -= len(layer.weights[i])
                    if (paramToChange < 0):
                        paramToChange += len(layer.weights[i])
                        layer.weights[i][paramToChange] += negativeGradients[paramToChange]*learningRate
                        break

    def cost(self, data, labels):
        """A function which takes in labelled data and outputs the average squared error on that for the current model."""
        SSE = 0
        for x in range(len(data)):
            SSE += np.sum((labels[x] - self.predict(data[x]))**2)
        return (SSE/len(data))

    def parameterSpaceCostField(self, gradientSpacing, data, labels):
        """Iterates through each parameter in the neural network and canges it be gradientSpacing then evaluates the cost on given data and labels.
        
        Returns an array with 3 columns, cost with parameter - gradientSpacing, cost with current value, and cost with parameter + gradientSpacing.
        These rows are arranged bias of first layer, weights of first layer, then the same for second layer, etc."""
        numParams = 0
        for layer in self.layers:
            numParams += len(layer.biases)
            for row in layer.weights:
                numParams += len(row)
        field = np.zeros((numParams, 3))
        
        # Begin iterating through parameters and evaluating costs.
        x = 0
        for layer in self.layers:
            for i in range(len(layer.biases)):
                layer.biases[i] -= gradientSpacing
                field[x][0] = self.cost(data, labels)
                layer.biases[i] += gradientSpacing
                field[x][1] = self.cost(data, labels)
                layer.biases[i] += gradientSpacing
                field[x][2] = self.cost(data, labels)
                layer.biases[i] -= gradientSpacing
                x += 1
            for weightRow in layer.weights:
                for i in range(len(weightRow)):
                    weightRow[i] -= gradientSpacing
                    field[x][0] = self.cost(data, labels)
                    weightRow[i] += gradientSpacing
                    field[x][1] = self.cost(data, labels)
                    weightRow[i] += gradientSpacing
                    field[x][2] = self.cost(data, labels)
                    weightRow[i] -= gradientSpacing
                    x += 1
        return field


    def numericGradient(self, input, spacing):
        """A basic numeric gradient calculator. Takes a 3 column input and a spacing scalar and outputs a single column representing the gradients of the input."""
        gradients = np.zeros(len(input))
        x = 0
        for row in input:
            gradients[x] = (((row[1] - row[0]) + (row[2] - row[1]))*0.5)/spacing
            x += 1
        return gradients


    class layer(object):
        """A subclass which can hold a single layer of a neural network."""

        def __init__(self, inputLen, outputLen):
            """Initializes a layer's weights and biases. (maybe should be random instead of zero?)"""
            self.weights = np.zeros((outputLen, inputLen))
            self.biases = np.zeros(outputLen)

        def ReLu(self, input):
            """Puts the input through a Rectified Linear Unit. (converts each input element to 0.0 if negative.)"""
            for x in range(len(input)):
                if input[x] < 0:
                    input[x] = 0

        def evaluate(self, input):
            """Takes an input to the layer and returns the ReLu(output of the layer)."""
            output = self.weights @ input + self.biases
            self.ReLu(output)
            return output


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