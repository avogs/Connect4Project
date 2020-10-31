import random
import numpy as np
from Connect4Board import Connect4Board
import copy

# Neural net should take as inputs every square on the board and a playerID. It should run on that then output the column
# most likely to result in a win for its playerID.
# Therefore the data it should take to train on could take a few formats:
    # It could take game in progress paired with a move and if that game was won.
    # It could get data from watching games between 2 random players
    # It could get data by playing against a random player.
    # It could get data by playing against another AI that is training (could result in weird behavior?)
# Training data will (for now) be made by random players and will take the form of an array of tuples
# with a board, a move, and a winner

class dataGenerator:
    def __init__(self, gamesToGenerate):
        self.numBoards = gamesToGenerate*6*7 # Number of boards generated in this.
        self.data = [] # array of arrays containing the board array, a move, and who won

    def produceRandomData(self):
        gameBoard = Connect4Board(random.randint(1,2))
        winner = 0
        finished = 0
        listItem = [np.zeros((6,7)),0,0]
        for x in range(1, self.numBoards):
            gameMoves = np.array([(random.sample(range(7),7)) for x in range(6)]).flatten()
            np.random.shuffle(gameMoves)
            winner = gameBoard.move(gameMoves[x])
            listItem[1] = gameMoves[x]
            self.data.append(copy.deepcopy(listItem))
            if winner > 0:
                finished = x
                break
            listItem[0] = gameBoard.board.copy()
        for x in range(0, finished):
            self.data[x][2] = winner
            
    #def generateRandomGame(self): # Randomly generates an array of 42 integers between 0 and 6 with no more than 6 repeats of each number.
    #    [(random.sample(range(7),7)) for x in range(6)]