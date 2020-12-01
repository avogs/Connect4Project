import numpy as np

class FromScratchNN(object):
    """A class which can be used to make and train a neural network. Built using minimal external libraries.
    
    This is being built as a project to review/dive deeper into python coding and deep learning internals."""

    def __init__(self, layerSizes):
        """Initializes a list of layers representing a neural network.
       
        Goes from from an input layer of length layerSizes[0] to an
        output layer of length of layerSizes[len(layerSizes)]. These
        layers are initialized with 0 as all their parameters."""
        self.layers = [] # A list of layer objects, ordered from input layer to output layer.  Should
                         # always have at least 1 element.
        for x in range(1,len(layerSizes)):
            self.layers.append(self.layer(layerSizes[x - 1], layerSizes[x]))


    def modelOutput(self, input):
        """Takes in a data point and runs it through the current model, returns the output of that."""
        nextOutput = input
        for x in range(len(self.layers)):
            nextOutput = self.layers[x].evaluate(nextOutput)
        return nextOutput

    def predict(self, input):
        """Returns the index of the max element in the array."""
        return np.argmax(modelOutput(self, input))


    def trainGradientDescent(self, data, labels, gradientSpacing, learningRate, epochs, batchSize=1):
        """Trains the neural network using a stochastic batch gradient descent."""
        # Note that this uses a numerical gradient, which can have significant
        # rounding errors and will usually be highly time intensive.
        # A more advanced method should use automatic differentiation or
        # something similar but that is out of the scope of this project (for
        # now).
        rand = np.random.default_rng()
        for x in range(epochs):
            batch_indices = rand.choice(len(data), size = batchSize)
            batch_data = data[batch_indices]
            batch_labels = labels[batch_indices]
            print("Beginning epoch: " + str(x))
            print("Current cost: " + str(self.cost(batch_data, batch_labels)))
            gradients = self.numericGradient(self.parameterSpaceCostField(gradientSpacing, batch_data, batch_labels), gradientSpacing)
            counter = 0
            for layer in self.layers:
                for index in range(len(layer.biases)):
                    layer.biases[index] -= gradients[counter] * learningRate
                    counter += 1
                for row in range(len(layer.weights)):
                    for entry in range(len(layer.weights[row])):
                        layer.weights[row][entry] -= gradients[counter] * learningRate
                        counter += 1

    def cost(self, data, labels):
        """A function which takes in labelled data and outputs the average squared error on that for the current model."""
        SSE = 0
        label = np.zeros(len(self.layers[len(self.layers) - 1].biases))
        for x in range(len(data)):
            label[labels[x]] = 1
            SSE += np.sum((label - self.modelOutput(data[x])) ** 2)
            label[labels[x]] = 0
        return (SSE / len(data))

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
        middle_cost = self.cost(data, labels)
        for layer in self.layers:
            for i in range(len(layer.biases)):
                layer.biases[i] -= gradientSpacing
                field[x][0] = self.cost(data, labels)
                layer.biases[i] += gradientSpacing
                field[x][1] = middle_cost
                layer.biases[i] += gradientSpacing
                field[x][2] = self.cost(data, labels)
                layer.biases[i] -= gradientSpacing
                x += 1
            for weightRow in layer.weights:
                for i in range(len(weightRow)):
                    weightRow[i] -= gradientSpacing
                    field[x][0] = self.cost(data, labels)
                    weightRow[i] += gradientSpacing
                    field[x][1] = middle_cost
                    weightRow[i] += gradientSpacing
                    field[x][2] = self.cost(data, labels)
                    weightRow[i] -= gradientSpacing
                    x += 1
        return field


    def numericGradient(self, input, spacing):
        """A basic numeric gradient calculator. Takes a 3 column input and a spacing scalar and outputs a single column representing the gradients of the input."""
        gradients = np.zeros(len(input))
        x = 0
        spacing_2 = 2 * spacing
        for row in input:
            gradients[x] = (row[2] - row[0]) / spacing_2
            x += 1
        return gradients


    class layer(object):
        """A subclass which holds a layer of a neural network."""

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