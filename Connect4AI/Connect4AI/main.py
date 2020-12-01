# Main: this is being used to test and run implemented classes.
import tensorflow as tf
from tensorflow import keras
from Connect4Board import Connect4Board
import numpy as np
from dataGenerator import dataGenerator
from FromScratchNN import FromScratchNN as FSNN

def makeConnect4Data():
    # Randomly generate 5000 games worth of data
    test = dataGenerator(5)
    test.produceRandomData()
    
    # Reformat the data to something which the neural net can be trained on.
    features = np.empty((len(test.data),43))
    for x in range(len(test.data)):
        features[x] = np.append(test.data[x][0].flatten(), test.data[x][2])
    labels = np.zeros((len(test.data)))
    for x in range(len(test.data)):
        labels[x] = [test.data[x][1]]
    return(features, labels)

def makeTensorFlowModel(features, labels):
    # Make a model with layers of neurons, 1 input layer which takes a board and
    # playerID, one hidden layer which goes to 43 neurons,
    # then a final layer where it goes to 7 cells representing each column where a
    # move can be made.
    model = keras.Sequential([
        keras.Input(shape=(43,)),
        keras.layers.Dense(43, activation=tf.nn.relu),
        keras.layers.Dense(7, activation=tf.nn.softmax)
        ])

    # Pick a loss function and optimizer that determines how the model learns.
    model.compile(optimizer="Adam",
                  loss='MSE') # Use the Adam algorithm for better computation
                              # times and (possibly) better results than SGD
    # Train the model
    model.fit(features, labels)
    return model

def makeBasicModel(features, labels, layers):
    newAI = FSNN(layers)
    newAI.trainGradientDescent(features, labels, 0.1, 0.5, 20)
    return newAI

#(features, labels) = makeConnect4Data()
#model = makeBasicModel(features, labels, [43,7,7])
#model = makeTensorFlowModel(features, labels)

# Make a test board and feed the model that to see how it reacts.
input = np.array([0,1,2,0,0,0,0,
                  0,1,2,0,0,0,0,
                  0,1,2,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0])
input = np.append(input, 1)
input2 = np.empty((1,43))
input2[0] = input

#a = model.predict(input2)
#print(a)
#print(model.predict(input))

# Currently since the model is being trained on entirely random moves it is
# equally likely to pick any column.



def getMnistData():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train, x_test, y_test)


(x_train, y_train, x_test, y_test) = getMnistData()

x_train_flattened = np.zeros((10000, 28*28))
for x in range(len(x_train[0:10000])):
    x_train_flattened[x] = np.ndarray.flatten(x_train[x])

# Train an AI to read handwritten numbers from the mnist dataset
mnistAI = FSNN([784, 10])
mnistAI.trainGradientDescent(x_train_flattened, y_train[0:10000], 0.05, 0.01, 2000)

# Print the prediction for the first number in the set
print(mnistAI.predict(np.ndarray.flatten(x_train[0])))
# Print the confidence levels for each category
print(mnistAI.modelOutput(np.ndarray.flatten(x_train[0])))
# Print the actual first number in the set
print(y_train[0])