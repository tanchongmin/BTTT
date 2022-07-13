import numpy as np
import logging

class Game:
    
    def __init__(self, board = []):        
        self.currentPlayer = 1
        self.grid_shape = (7,7)
        self.input_shape = (3,7,7)
        if board==[]:
            self.b = np.array([0 for i in range(49)], dtype=np.int)
            
            # Variant 1 (D4) training
            ind = 24
            # Brick starts randomly at D3, D4
#             ind = [23, 24][np.random.randint(2)]
            # Brick starts randomly at C3, D3, D4
#             ind = [16, 23, 24][np.random.randint(3)]
            
            self.b[ind]=100
            
        else:
            self.b = board
            
        self.gameState = GameState(self.b, 1)
        self.actionSpace = np.array([0 for i in range(49)], dtype=np.int)
        self.pieces = {'1':'O', '0': '-', '-1':'X', '100':'B'}
        self.name = 'bttt'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(self.b, 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state,actionValues)]

                                   
        currentBoard = state.board.reshape(self.grid_shape)
        currentAV = actionValues.reshape(self.grid_shape)

        #learn faster by accounting for symmetry

        identities.append((GameState(np.flip(currentBoard,0).flatten(), state.playerTurn), np.flip(currentAV,0).flatten()))
        identities.append((GameState(np.flip(currentBoard,1).flatten(), state.playerTurn), np.flip(currentAV,1).flatten()))
        identities.append((GameState(np.flip(currentBoard,(0,1)).flatten(), state.playerTurn), np.flip(currentAV,(0,1)).flatten()))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.grid_shape = (7,7)
        self.winlength = 4
        self.pieces = {'1':'O', '0': ' ', '-1':'X', '100':'B'}
        self.winners = []
        # add in list of winners
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                # horizontal
                if j<=self.grid_shape[1]-self.winlength:
                    self.winners.append([i*self.grid_shape[1]+j+k for k in range(self.winlength)])
                # vertical
                if i<=self.grid_shape[0]-self.winlength:
                    self.winners.append([(i+k)*self.grid_shape[1]+j for k in range(self.winlength)]) 
                # diagnoals
                if i<=self.grid_shape[0]-self.winlength and j<=self.grid_shape[1]-self.winlength:
                    self.winners.append([(i+k)*self.grid_shape[1]+j+k for k in range(self.winlength)]) 
                    self.winners.append([(i+self.winlength-1-k)*self.grid_shape[1]+j+k for k in range(self.winlength)])                                  
                                   
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        # only those empty cells can be filled
        return np.where(self.board == 0)[0]

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board==self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-self.playerTurn] = 1
        
        brick_position = np.zeros(len(self.board), dtype=np.int)
        brick_position[self.board==100] = 1

        position = np.concatenate((currentplayer_position,other_position,brick_position))

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board==1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-1] = 1
        
        brick_position = np.zeros(len(self.board), dtype=np.int)
        brick_position[self.board==100] = 1

        position = np.concatenate((player1_position,other_position,brick_position))

        id = ''.join(map(str,position))

        return id

    def _checkForEndGame(self):
        if np.count_nonzero(self.board) == self.grid_shape[0]*self.grid_shape[1]:
            return 1

        for winner in self.winners:
            if sum([self.board[x] for x in winner]) == self.winlength * -self.playerTurn:
                return 1
        return 0


    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for winner in self.winners:
            if sum([self.board[x] for x in winner]) == self.winlength * -self.playerTurn:
                return (-1, -1, 1)
        return (0, 0, 0)


    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])


    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action]=self.playerTurn
        
        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done) 


    def printgame(self):
        l='ABCDEFGHIJKLMNOPQRSTUVWYZ'
        for r in range(self.grid_shape[0]):
            print(l[r],*[self.pieces[str(x)] for x in self.board[self.grid_shape[1]*r : 
                (self.grid_shape[1]*r + self.grid_shape[1])]],sep='|')
        print(' ',*['__' for i in range(1,8)],sep='')
        print(' ',*[str(i) for i in range(1,8)],sep='|')
        
    def render(self, logger):
        for r in range(self.grid_shape[0]):
            logger.info([self.pieces[str(x)] for x in self.board[self.grid_shape[1]*r : 
                (self.grid_shape[1]*r + self.grid_shape[1])]])
        logger.info(['_' for i in range(1,8)])
        logger.info([str(i) for i in range(1,8)])