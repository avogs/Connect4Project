# Main: this is being used to test and run implemented classes.

#import tensorflow as tf
#from tensorflow import keras
from Connect4Board import Connect4Board
import numpy as np
from dataGenerator import dataGenerator

# Randomly generate 5000 games worth of data
test = dataGenerator(20)
test.produceRandomData()

# Make a model with layers of neurons, 1 input layer which takes a board and playerID, one hidden layer which goes to 100 neurons,
# then a final layer where it goes to 7 cells representing each column where a move can be made.
#model = keras.Sequential([
#    keras.Input(shape=(43,)),
#    keras.layers.Dense(43, activation=tf.nn.relu),
#    keras.layers.Dense(7, activation=tf.nn.softmax)
#    ])

# Pick a loss function and optimizer that determines how the model learns. 
#model.compile(optimizer="Adam",
#              loss='MSE') # Use the Adam algorithm for better computation times and (possibly) better results than SGD

# Reformat the data to something which the neural net can be trained on.
features = np.empty((len(test.data),43))
for x in range(len(test.data)):
    features[x] = np.append(test.data[x][0].flatten(), test.data[x][2])
labels = np.zeros((len(test.data), 7))
for x in range(len(test.data)):
    labels[x][test.data[x][1]] = 1.0

# Train the model
#model.fit(features, labels)

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

# Currently since the model is being trained on entirely random moves it should come out equally likely to pick any column.



from FromScratchNN import FromScratchNN as FSNN

newAI = FSNN([43, 7])
newAI.trainGradientDescent(features, labels, 0.1, 0.5, 20)
print(newAI.predict(input))

# Once again since the model is being trained on entirely random moves it should come out equally likely to pick any column.