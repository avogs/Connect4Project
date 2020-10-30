import random

# Neural net should take as inputs every square on the board and its playerID. It should run on that then output the column
# most likely to result in a win for its playerID.
# Therefore the data it should take to train on could take a few formats:
    # It could take game in progress paired with a move and if that game was won.
    # It could get data from watching games between 2 random players
    # It could get data by playing against a random player.
    # It could get data by playing against another AI that is training (could result in weird behavior?)
# Training data will (for now) be made by random players and will take the form of an array of tuples
# with a board, a move, and a winner

class randomVSrandomData(gamesToGenerate):
    def __init__(self):
        data = [[[],[],[]] in range(0,gamesToGenerate)] # array of tuples containing the board array, a move, and who won
        
    def produceData(self, numGames):
        gameBoard = Connect4Board(random.randint(1,2))
        for x in range(0, numGames):
            # setupMoves = generateRandomGame()
            datas = gameBoard.loadGame([(random.sample(range(6),6)) for x in range(6)])
            boardHistory = data
            winner = datas[1]
            for board in range(0,len(boardHistory)):
                boardHistory[board][2] = winner
            
    def generateRandomGame(self): # Randomly generates an array of 42 integers between 0 and 6 with no more than 6 repeats of each number.
        [(random.sample(range(6),6)) for x in range(6)]