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
    """A class which uses Connect4Board to generate training data for a connect 4 AI"""
    def __init__(self, gamesToGenerate):
        """Takes an argument for how many games to generate and initializes a list of data."""
        self.gamesToGenerate = gamesToGenerate
        self.data = [] # list of arrays containing the board array, a move, and who won

    def produceRandomData(self):
        """Generates games with 2 randomly players facing each other and records the data and the victor in self.data."""
        offset = 0
        for game in range(self.gamesToGenerate):
            gameBoard = Connect4Board(random.randint(1,2))
            winner = 0
            finished = 0
            listItem = [np.zeros((6,7)),0,0]
            for x in range(0, 42):
                gameMoves = np.array([(random.sample(range(7),7)) for x in range(6)]).flatten() # Generates 42 moves with no more than 6 of each column.
                np.random.shuffle(gameMoves)
                winner = gameBoard.move(gameMoves[x])
                listItem[1] = gameMoves[x]
                self.data.append(copy.deepcopy(listItem))
                offset += 1
                if winner > 0:
                    finished = x
                    break
                listItem[0] = gameBoard.board.copy()
            for x in range(offset-finished, offset):
                self.data[x][2] = winner