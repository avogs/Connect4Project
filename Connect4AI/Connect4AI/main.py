# Main: this is being used to test and run implemented classes.

import tensorflow as tf
from tensorflow import keras
from Connect4Board import Connect4Board
import numpy as np
from dataGenerator import dataGenerator

#RunGame = Connect4Board(1)
#RunGame.game()
test = dataGenerator(2000)
test.produceRandomData();

print(np.shape(test.data))

model = keras.Sequential([
    keras.Input(shape=(43,)),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(7, activation=tf.nn.softmax)
    ])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.compile(optimizer="Adam",
              loss='MSE') # Use the Adam algorithm for better computation times and (arguably) results than SGD

features = np.empty((len(test.data),43))
for x in range(len(test.data)):
    features[x] = np.append(test.data[x][0].flatten(), test.data[x][2])
labels = np.zeros(len(test.data))
for x in range(len(test.data)):
    labels[x] = test.data[x][1]
model.fit(features, labels)

input = np.array([0,1,2,0,0,0,0,
                  0,1,2,0,0,0,0,
                  0,1,2,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0])
input = np.append(input, 1)
input2 = np.empty((1,43))
input2[0] = input
a = model.predict(input2)

print(a)

# This should get a board and a winner and train to output the move
