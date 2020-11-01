class FromScratchNN(object):
    """description of class"""




    def sigmoid(self): # Or ReLu???
        """A normalizing function"""

    # sigmoid(W1A1 + W2A2 + W3A3... + WnAn - bias)
    # A's are a long column vector of activiations, w's are from a KxN matrix, each row
    # of which represents the weights to multiply each a by to get output for the next layer
    # [w00, w01, ..., w0n][a0]  + [b0]
    # [w10, w11, ..., w1n][a1]  + [b1]
    # [..., ..., ..., ...][...] + [...]
    # [wk0, wk1, ..., wkn][an]  + [bn]
    # k is number of neurons in next layer
    # multiplying these matrices then add the bias and appy sigmoid to get the next activation layer.
    # 
    # Cost will be sum of differences between output and desired output
    # Consider average cost over training data, minimize that.
    # Compute the gradient of the cost function and increase/decrease each weight and bias proportionally it's value in the negative gradient.


    # This is a basic multilayer perceptron