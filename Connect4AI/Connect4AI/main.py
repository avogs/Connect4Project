# Main: this is being used to test and run implemented classes.

from Connect4Board import Connect4Board
from dataGenerator import dataGenerator

#RunGame = Connect4Board(1)
#RunGame.game()
test = dataGenerator(1)
test.produceRandomData();
for x in range(len(test.data)):
    print(test.data[x])