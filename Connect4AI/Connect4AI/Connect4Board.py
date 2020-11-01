## Connect 4
import matplotlib.pyplot as plt
import numpy as np

class Connect4Board:
    """Class Description..."""
    def __init__(self, startingPlayer):
        self.board = np.array([[0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0]]) # array of ints representing board
        self.displayBoard = np.array([[0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0]]) # array of ints representing board in a display friendly manner
        self.heights = [0,0,0,0,0,0,0] # Array of ints showing how full each column is
        self.playerID = startingPlayer # ID of player who moves first
        
    def move(self, column): # On win returns player ID, otherwise returns 0 "public method"
        if (not (column > 6) and not (column < 0)) and (self.heights[column] < 6): # If the move is invalid do nothing
            self.board[self.heights[column]][column] = self.playerID # Place the token
            if (self._isWin(column, self.heights[column])):
                return self.playerID # If the token resulted in a win return playerID
            self.heights[column] += 1 # Update number of tokens in the column
            self._swapPlayers() # If the move has completeted switch turns to the other player.
        return 0 # The move is complete and playerID was swapped or the move was invalid and nothing was done
        
    def viewBoard(self): # "public method"
        for x in range(0, 6): # updates displayBoard (self.board flipped)
            self.displayBoard[5-x] = self.board[x]
        plt.imshow(self.displayBoard, cmap='hot', interpolation='nearest') # show the board as a heatmap lol :P
        plt.show()
        
    def _isWin(self, column, row): # "private helper method"
        if ((self._checkLeft(column, row) + self._checkRight(column, row)) > 4):
            return True # Since the placed piece gets counted twice we check for "5+" in a row instead of "4+"
        elif ((self._checkDown(column, row)) > 3): # checkUp is unnessecary since any placed token will be on top of its col
            return True # Returns on a vertical 4 in a row
        elif ((self._checkUpRight(column, row) + self._checkDownLeft(column, row)) > 4):
            return True # Returns on 4 in a row in an upwards diagonal slope
        elif ((self._checkDownRight(column, row) + self._checkUpLeft(column, row)) > 4):
            return True # Returns on 4 in a row in an downwards diagonal slope
        else:
            return False
    
    # Each of these methods checks in the specified direction until it hits an edge of the board
    # or runs out of matching tokens and returns the number of tokens in a line in that direction. 
    
    def _checkLeft(self, column, row): # "private helper method"
        if ((column != 0) and self.board[row][column-1] == self.playerID):
            return 1 + self._checkLeft(column-1, row)
        else:
            return 1
            
    def _checkRight(self, column, row): # "private helper method"
        if ((column != 6) and self.board[row][column+1] == self.playerID):
            return 1 + self._checkRight(column+1, row)
        else:
            return 1
        
    def _checkDown(self, column, row): # "private helper method"
        if ((row != 0) and self.board[row-1][column] == self.playerID):
            return 1 + self._checkDown(column, row-1)
        else:
            return 1
        
    def _checkUpRight(self, column, row): # "private hellper method"
        if ((column != 6) and (row != 5) and self.board[row+1][column+1] == self.playerID):
            return 1 + self._checkUpRight(column+1, row+1)
        else:
            return 1
            
    def _checkDownLeft(self, column, row): # "private helper method"
        if ((column != 0) and (row != 0) and self.board[row-1][column-1] == self.playerID):
            return 1 + self._checkDownLeft(column-1, row-1)
        else:
            return 1
        
    def _checkDownRight(self, column, row): # "private helper method"
        if ((column != 6) and (row != 0) and self.board[row-1][column+1] == self.playerID):
            return 1 + self._checkDownRight(column+1, row-1)
        else:
            return 1
        
    def _checkUpLeft(self, column, row): # "private helper method"
        if ((column != 0) and (row != 5) and self.board[row+1][column-1] == self.playerID):
            return 1 + self._checkUpLeft(column-1, row+1)
        else:
            return 1
    
    def _swapPlayers(self): # "private helper method" Swaps players :O
        if (self.playerID == 1):
            self.playerID = 2
        else:
            self.playerID = 1
    
    def loadGame(self, inputBoard, column): # Runs a game from an input array of moves. Returns the winner or 0 if draw or game not over.
        self.board = inputBoard
        return move(column)
        gameOver = False
        i = 0
        while (not gameOver and i < len(moves)):
            boardHistory.append((self.board, moves[i]))
            # If board is full declare a draw
            gameOver = True
            for col in range(0,7): # If there are any empty slots left
                if (self.board[5][col] == 0):
                    gameOver = False
                    break
            # Game is a draw
            if (gameOver):
                return 0
                break
            column = -1
            while ((column < 0) or (column > 6)):
                column = moves[i]
                i += 1
            if (self.move(column) != 0):
                gameOver = True
                return (boardHistory, self.playerID)
        return (boardHistory, 0)
            
    def game(self): # game loop
        gameOver = False
        while (not gameOver):
            self.viewBoard()
            # If board is full declare a draw
            gameOver = True
            for col in range(0,7): # If there are any empty slots left
                if (self.board[5][col] == 0):
                    gameOver = False
                    break
            # Game is a draw
            if (gameOver):
                print("No more moves possible, the game is a draw.")
                break
            column = -1
            while ((column < 0) or (column > 6)):
                column = int(input())
            if (self.move(column) != 0):
                gameOver = True
                self.viewBoard()
                print("Player " + str(self.playerID) + " won!")
                return self.board